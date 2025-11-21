import argparse
import heapq
import collections
import itertools
import json
import numpy
import os
#import re

import pandas
import sentencepiece
import torch
import torchviz

# model params
vocabulary_size = 3169
vocabulary_size_all_tokens = vocabulary_size + 2 # +2 for UNK and BOS

key_value_heads = 4  # 4 key and value heads
key_value_to_query_heads_ratio = 2  # end up with 2 queries per 1 key

attention_heads = key_value_heads * key_value_to_query_heads_ratio  # grouped query attention (GQA) - 8 query heads (rotary) for 4 key (rotary) and 4 value
dimensions_per_head = 8  # how many dimensions from the token each query head will be assigned with
vocabulary_dimensions = attention_heads * dimensions_per_head  # size of vector associated with each token

token_limit = 50  # limit of the context token window
transformer_layers = 6  # how many copies of the whole transformer stack are there
normalization_epsilon = 1e-5 # for RootMeanSquareNormalization

hidden_feed_forward_network_multiply_of = 64
hidden_feed_forward_network_dimensions = (8 * vocabulary_dimensions) // 2
#round up so hidden dimensions will be whole `multiply_of`
ffn_rounding_up = (hidden_feed_forward_network_multiply_of - (hidden_feed_forward_network_dimensions % hidden_feed_forward_network_multiply_of)) % hidden_feed_forward_network_multiply_of
hidden_feed_forward_network_dimensions += ffn_rounding_up

# training
batch_size = 500
learning_rate = 0.001
dropout = 0.05
weight_decay = 0.01
multiple_of = 1

# files
input_data_json_file_name = os.path.join("data-input", "sentences.json")
plain_text_data_file_name = os.path.join("data-work", "plain_text_sentences.txt")
vocabulary_file_name =  os.path.join("data-work", "vocabulary")
vocabulary_file_model_name = vocabulary_file_name + ".model"
token_data_file_name = os.path.join("data-work", "sentences.tokens")
model_visualization_dot_name = os.path.join("data-work", "model-visualization-dot")
model_visualization_onnx_name = os.path.join("data-work", "model-visualization-onnx")

# global variables
word_counts_total = collections.Counter()
first_word_total = collections.Counter()
letter_counts_total = collections.Counter()


class RootMeanSquareNormalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(vocabulary_dimensions))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + normalization_epsilon)
        return out.type_as(x) * self.weight


class SwiGlu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w    = torch.nn.Linear(vocabulary_dimensions, hidden_feed_forward_network_dimensions, bias=False)
        self.v    = torch.nn.Linear(vocabulary_dimensions, hidden_feed_forward_network_dimensions, bias=False)
        self.back = torch.nn.Linear(hidden_feed_forward_network_dimensions, vocabulary_dimensions, bias=False)

    def forward(self, x):
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        return self.back(torch.nn.functional.silu(self.w(x)) * self.v(x))

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.calculate_pre_populated_theta_outer_cosinus_sinus(debug)
        print(self.theta_outer_cos.shape)

    def calculate_pre_populated_theta_outer_cosinus_sinus(self, debug=False):
        # https://www.youtube.com/watch?v=GQPOtyITy54
        # example each token has 128 dimensions, with 4 heads, it's 32 dimensions per head

        # when dimensions_per_head==32 (0,2,4...30) and when dimensions_per_head==33 (0,2,4...30,32)
        even_dimensions = torch.arange(0, dimensions_per_head, 2)

        # when dimensions_per_head even then no change.
        # when dimensions_per_head odd then truncate last dimension: dimensions_per_head==33 (0,2,4...30)
        # so we get even amount (aligned for cos and sin) of even values
        aligned_dimensions = even_dimensions[: (dimensions_per_head // 2)]

        # rescaled from 0 to almost 1.0 (non-inclusive)
        power_normalized = aligned_dimensions.float() / dimensions_per_head

        # base on which the inverse power will be calculated (the theta)
        # https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1626899/full
        baseline=10000.0

        # example 10000.0^0.5=100 and 1.0 / 100 => 0.01
        theta = 1.0 / (baseline ** power_normalized)

        tokens_range = torch.arange(token_limit)

        # https://docs.pytorch.org/docs/stable/generated/torch.outer.html
        # https://en.wikipedia.org/wiki/Outer_product
        theta_outer = torch.outer(tokens_range, theta).float()

        # https://docs.pytorch.org/docs/stable/generated/torch.cos.html
        theta_outer_cos = torch.cos(theta_outer)
        theta_outer_sin = torch.sin(theta_outer)

        if debug:
            print(f"vocabulary dimensions {vocabulary_dimensions} spread to {attention_heads} attention heads => {dimensions_per_head} dimensions per head")
            print(even_dimensions)
            print(aligned_dimensions)
            print(power_normalized)
            print(theta)
            print(theta_outer)
            print(theta_outer_cos)
            print(theta_outer_sin)

        self.register_buffer("theta_outer_cos", theta_outer_cos, persistent=False)
        self.register_buffer("theta_outer_sin", theta_outer_sin, persistent=False)

    def reshape_for_broadcast(cosine_or_sine, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert cosine_or_sine.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return cosine_or_sine.view(shape)

    def apply_on_embedding_query_and_key(self, query, key):
        # https://pub.towardsai.net/llama-architecture-a-deep-dive-into-efficiency-and-mathematics-9c95c0e4bf8b
        # https://pypi.org/project/rotary-embedding-tensorflow/
        # Query rotation
        query_real, query_imaginary = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)

        rotation_cos = self.reshape_for_broadcast(self.theta_outer_cos, query_real)
        rotation_sin = self.reshape_for_broadcast(self.theta_outer_sin, query_real)

        query_out_real      = query_real * rotation_cos - query_imaginary * rotation_sin
        query_out_imaginary = query_real * rotation_sin + query_imaginary * rotation_cos

        query_out = torch.stack([query_out_real, query_out_imaginary], dim=-1).flatten(3)

        # Key rotation
        key_real, key_imaginary = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

        key_out_real      = key_real * rotation_cos - key_imaginary * rotation_sin
        key_out_imaginary = key_real * rotation_sin + key_imaginary * rotation_cos

        key_out = torch.stack([key_out_real, key_out_imaginary], dim=-1).flatten(3)

        return query_out.type_as(query), key_out.type_as(key)


class Model(object):

    @staticmethod
    def reshape_key_value(x, scale_ratio) -> torch.Tensor:
        if scale_ratio == 1:
            # will not need to change shape when value heads == key/value heads, just return as is
            return x

        return (
            x[:, :, :, None, :]
            .expand(batch_size, token_limit, key_value_heads, scale_ratio, dimensions_per_head)
            .reshape(batch_size, token_limit, key_value_heads * scale_ratio, dimensions_per_head)
        )


def _panda_split_to_words(text):
    # text = re.sub(r"[^\w\s']", '', text)  # remove punctuation
    words = text.split()
    return collections.Counter(words)


def _panda_first_word(text):
    words = text.split()
    if len(words) > 0:
        return words[0]
    else:
        print('ERROR, bad sentence encountered', text)
        return "_ERROR_"


def _header(*args, **kwargs):
    print()
    print("-"*70)
    print(*args, **kwargs)


def _load_dataframe():
    _header('Loading input json file as panda dataframe', input_data_json_file_name)
    df = pandas.read_json(input_data_json_file_name)
    return df


def _word_count(df, debug):
    _header("Counting words")

    print('Splitting texts to words')
    df['words'] = df['text'].apply(_panda_split_to_words)

    print('Counting words in each text')
    df['word_counts'] = df['words'].apply(collections.Counter)

    print('Adding up words for all texts')
    for count_in_one_text in df['words']:
        word_counts_total.update(count_in_one_text)

    if debug:
        i=1
        for word in sorted(word_counts_total, key=word_counts_total.get, reverse=True):
            print(i, word, ' => ', word_counts_total[word])
            i=i+1

    print('Words used in texts', len(word_counts_total))
    if vocabulary_size == len(word_counts_total):
        print('Matching hard-coded vocabulary size')
    else:
        print('Warning: Not matching the hard-coded vocabulary size', vocabulary_size)


def _first_word_of_text(df, debug):
    _header('Counting first word of each text')
    df['first'] = df['text'].apply(_panda_first_word)

    for first in df['first']:
        first_word_total.update([first])

    if debug:
        for word in sorted(first_word_total, key=first_word_total.get, reverse=True):
            print(word, ' => ', first_word_total[word])


def _letter_usage_statistics(debug):
    _header('Counting letter usage of all words')
    for word in word_counts_total:
        # there is ~12 bytes of savings in huffman tables if i would use single
        # escape/shortcut characters for these 'once upon a time,' cases, but would
        # spend more bytes in firmware to handle these edge cases. So it's better to
        # have them inside the huffman table correctly without extra conditions in
        # firmware
        if word == 'onceuponatime,':
            word = 'once upon a time,'
            # word = '1'
        elif word == 'oneday':
            word = 'one day'
            # word = '2'

        # not multiplying by the letter usage because that will use indexes to whole vocabulary tokens
        # count = word_counts_total[word]
        # letters = Counter(word)
        # letters *= count
        # letter_counts_total.update(letters)
        # letter_counts_total.update({'0': count})

        letter_counts_total.update(word.upper())
        letter_counts_total.update('0')

    total_char_used_for_vocabulary = 0
    for letter in sorted(letter_counts_total, key=letter_counts_total.get, reverse=True):
        total_char_used_for_vocabulary += letter_counts_total[letter]
        if debug:
            print(letter, ' => ', letter_counts_total[letter])

    print(f"Total amount of characters used to store whole vocabulary: {total_char_used_for_vocabulary}")
    return total_char_used_for_vocabulary


def _build_huffman_table(counter):
    heap = []
    _count = itertools.count()

    # leaf nodes
    for ch, freq in counter.items():
        heap.append((freq, next(_count), ch))
    heapq.heapify(heap)

    # merge to single tree
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, next(_count), (left, right)))

    # recursively assign codes
    [(freq, _, root)] = heap
    table = {}

    def assign_codes(node, prefix=""):
        if isinstance(node, str):
            table[node] = prefix or "0"  # single-character case
        else:
            new_left, new_right = node
            assign_codes(new_left, prefix + "0")
            assign_codes(new_right, prefix + "1")

    assign_codes(root)
    return table

def _huffman_stream_as_padded_cpp(debug, prefix, stream_payload, codeword_max_len_of_bits, character_ord_min):
    pad_len = (8 - len(stream_payload) % 8) %8  # %8 handles exact multiples of bytes
    print(f"Stream needs to be padded with {pad_len} zeros for byte aligned")
    stream_payload = stream_payload + "0" * pad_len

    hex_str = hex(int(stream_payload, 2))[2:]  # remove '0x' prefix

    if debug:
        print(f"Final combined streamed payload in bits {stream_payload}")
        print(f"As one whole hex string {hex_str}")

    hex_chunks = []
    for i in range(0, len(stream_payload), 8):
        chunk = stream_payload[i:i + 8]
        hex_chunk = hex(int(chunk, 2))[2:].zfill(2)  # 2 hex digits = 8 bits
        hex_chunks.append(hex_chunk)

    if debug:
        print(f"As {len(hex_chunks)} 8-bit hex chunks:")

        print(f"static constexpr std::uint8_t {prefix}_max_codeword_bits_length_bits = {codeword_max_len_of_bits.bit_length()};"
              f"// codewords have different sizes (of bits), figure out how many bits needed to store the biggest one")
        print(f"static constexpr std::uint8_t {prefix}_vocabulary_huffman_character_offset = {character_ord_min};")
        print(f"static constexpr std::array<std::uint8_t,{len(hex_chunks)}> {prefix}_vocabulary_huffman_stream{{", end="")
        comma = ''
        for chunk in hex_chunks:
            print(f"{comma}0x{chunk}", end="")
            comma = ', '
        print("};")

    return len(hex_chunks)


def words_counts_and_stats(debug=False):
    df = _load_dataframe()
    _word_count(df, debug)
    _first_word_of_text(df, debug)
    total_char_used_for_vocabulary = _letter_usage_statistics(debug)

    _header('Huffman table creation')
    huffman_table = _build_huffman_table(letter_counts_total)
    total_huffman_table_codeword_bits_used = 0
    codeword_max_len_of_bits = 0
    character_ord_min = 255
    character_ord_max = 0
    codeword_lengths_count = collections.Counter()
    codeword_lengths_sum = 0
    for ch, code in huffman_table.items():
        bits_used = len(code) * letter_counts_total[ch]

        if ord(ch) < character_ord_min:
            character_ord_min = ord(ch)
        if ord(ch) > character_ord_max:
            character_ord_max = ord(ch)

        if codeword_max_len_of_bits < len(code):
            codeword_max_len_of_bits = len(code)

        codeword_lengths_count.update([str(len(code))])
        codeword_lengths_sum += len(code)
        if debug:
            print(f"{ch!r} (ord {ord(ch):>3}): {code:>14}, length {len(code):>2} x frequency {letter_counts_total[ch]:>4} = {bits_used:>5} bits used in "
                  f"total whole vocabulary to address this codeword")

        total_huffman_table_codeword_bits_used += bits_used

    if debug:
        print(f"Codeword length statistics", codeword_lengths_count)

    codeword_lengths_max = int(max(codeword_lengths_count, key=codeword_lengths_count.get))-1 # we can offset by -1 as that is the minimum we would move anyway
    print(f"Largest codeword length {codeword_lengths_max} (but needs to be offset by +1) ({codeword_lengths_max.bit_length()} bits to store this number)")

    delta_characters_ord = character_ord_max - character_ord_min
    print(f"Output character min ord {character_ord_min}, max ord {character_ord_max}, delta {delta_characters_ord}, "
          f"needing bits {delta_characters_ord.bit_length()}")
    total_bits_used_in_huffman_table_for_output_letter = delta_characters_ord.bit_length() * len(huffman_table.items())

    # 2 bytes for codeword (upto 14bits) + 1 byte for codeword mask (only 4bits needed) + 1 byte for the output character
    huffman_table_bytes_aligned = 4 * len(huffman_table.items())
    total_huffman_bytes_used = int( (total_huffman_table_codeword_bits_used + 7) / 8 )
    print(f"Huffman (aligned) Codeword max length {codeword_max_len_of_bits}, total bits used as indexes to codewords "
          f"{total_huffman_table_codeword_bits_used}. Bytes used as indexes to codewords {total_huffman_bytes_used} + "
          f"huffman table {huffman_table_bytes_aligned} => total {huffman_table_bytes_aligned + total_huffman_bytes_used} "
          f"bytes")

    huffman_table_bits_streamed = \
        len(huffman_table.items()) * (codeword_max_len_of_bits.bit_length() + delta_characters_ord.bit_length()) + \
        codeword_lengths_sum

    huffman_table_bytes_streamed = int( (huffman_table_bits_streamed+7) / 8)

    print(f"Huffman (streamed) huffman table size {len(huffman_table.items())} * (codeword bits length (encoded in bits) "
          f"{codeword_max_len_of_bits.bit_length()} + output character delta ord in bits {delta_characters_ord.bit_length()})"
          f" + all code words of the whole table summarized {codeword_lengths_sum} = "
          f"{huffman_table_bits_streamed} bits ({huffman_table_bytes_streamed} bytes)")

    huffman_saves = total_char_used_for_vocabulary - (huffman_table_bytes_aligned + total_huffman_bytes_used)

    print(f"Non huffman approach {total_char_used_for_vocabulary} bytes - huffman (aligned table + indexes)  "
          f"{huffman_table_bytes_aligned + total_huffman_bytes_used} bytes = aligned huffman saves ~{huffman_saves} bytes (ignoring "
          f"the fact that huffman decoder needs to be implemented in the firmware)")

    print(f"Streamed huffman table saves further {huffman_table_bytes_aligned - huffman_table_bytes_streamed} "
           f"bytes ignoring the firmware differences to support this")

    sorted_huffman_table = sorted(huffman_table.items(), key=lambda x: (len(x[1])))

    if debug:
        _header('Huffman (aligned) table ordered for the firmware')
        for ch, code in sorted_huffman_table:
            print(f"{{ .code=0'b{code.ljust(16, '0')}, .bits={str(len(code)).zfill(2)} .character='{ch}' }}, // {code}")

    _header('Huffman (streamed) payload ')
    stream_payload = ''
    max_entry_len = 0
    for ch, code in sorted_huffman_table:
        current_entry = f"{len(code):0{codeword_max_len_of_bits.bit_length()}b}{code}{(ord(ch)-character_ord_min):0{delta_characters_ord.bit_length()}b}"

        if max_entry_len < len(current_entry):
            max_entry_len = len(current_entry)

        if debug:
            print(f"Entry for {ch} = {current_entry:<24} (len={len(code):0{codeword_max_len_of_bits.bit_length()}b} "
                  f"code={code:<{codeword_max_len_of_bits}} "
                  f"char_delta={(ord(ch)-character_ord_min):0{delta_characters_ord.bit_length()}b} + "
                  f"char_offset={character_ord_min})")

        stream_payload += current_entry

    print(f"Max entry length = {max_entry_len} (firmware needs to hold at least this amount of bits)");
    hex_chunks_len = _huffman_stream_as_padded_cpp(debug,"stream", stream_payload, codeword_max_len_of_bits, character_ord_min)

    _header('Huffman (streamed-packed) payload ')
    stream_payload = ''
    length = 0
    for ch, code in sorted_huffman_table:
        if length != len(code):
            changing_size_payload = (f"{len(code):0{codeword_max_len_of_bits.bit_length()}b}"
                                     f"{int(codeword_lengths_count[str(len(code))])-1:0{codeword_lengths_max.bit_length()}b}")
            if debug:
                print(f"Size change payload:{changing_size_payload} len:{len(code)} repeated(-1):{codeword_lengths_count[str(len(code))]-1}")

            stream_payload += changing_size_payload
            length = len(code)

        current_entry = f"{code}{(ord(ch)-character_ord_min):0{delta_characters_ord.bit_length()}b}"

        if debug:
            print(f"Entry for {ch} = {current_entry:<20} ("
                  f"code={code:<{codeword_max_len_of_bits}} "
                  f"char_delta={(ord(ch)-character_ord_min):0{delta_characters_ord.bit_length()}b} + "
                  f"char_offset={character_ord_min})")

        stream_payload += current_entry

    # Calculate how many bits to pad to reach multiple of 8
    print()
    pad_len = (8 - len(stream_payload) % 8) %8  # %8 handles exact multiples of bytes
    print(f"Stream needs to be padded with {pad_len} zeros for byte aligned")
    stream_payload = stream_payload + "0" * pad_len
    print(f"Stream size in {int(len(stream_payload)/8)} bytes")
    if debug:
        print(f"Final combined stream payload in bits {stream_payload}")

    hex_chunks_len = _huffman_stream_as_padded_cpp(debug,"packed_stream", stream_payload, codeword_max_len_of_bits, character_ord_min)

    return df


def plain_text():
    _header('Loading input json file', input_data_json_file_name)
    input_json_file = open(input_data_json_file_name, 'r')
    input_sentences = json.load(input_json_file)
    input_json_file.close()

    _header('Parsing to a plain text file', plain_text_data_file_name)
    plain_file = open(plain_text_data_file_name, 'w')
    for text in input_sentences:
        text = text["text"]
        plain_file.write(text)
        plain_file.write("\n")
    plain_file.close()
    print('Processed', len(input_sentences), 'sentences')


def vocabulary_creation():
    _header("Preparing vocabulary tokens")
    # https://github.com/google/sentencepiece
    # https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
    # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.tokenizers.SentencePiece.html
    # https://medium.com/ai-enthusiast/unlocking-the-power-of-sentencepiece-encoding-a-comprehensive-guide-c181bb3d75ee
    # https://github.com/google/sentencepiece/issues/412
    # https://github.com/google/sentencepiece/issues/121
    # https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L193
    # https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb
    # https://github.com/google/sentencepiece/issues/636
    # https://stackoverflow.com/questions/77036828/some-doubts-about-sentencepiece

    sentencepiece.SentencePieceTrainer.train(input=plain_text_data_file_name,
                                             model_prefix=vocabulary_file_name,
                                             model_type="word",
                                             vocab_size=vocabulary_size_all_tokens,
                                             self_test_sample_size=0,
                                             input_format="text",
                                             character_coverage=1.0,
                                             num_threads=os.cpu_count(),
                                             split_digits=True,
                                             allow_whitespace_only_pieces=True,
                                             byte_fallback=False,
                                             unk_surface="__UNKNOWN__",
                                             bos_id=1,  # maybe i will need the beginning
                                             eos_id=-1,  # maybe i will get away without the end
                                             normalization_rule_name="identity")


def to_tokens():
    _header("Loading vocabulary tokenizer model")
    sentencepiece_model = sentencepiece.SentencePieceProcessor(model_file = vocabulary_file_model_name)

    _header('Loading input json file to parse json into tokens', input_data_json_file_name)
    input_json_file = open(input_data_json_file_name, 'r')
    all_texts = json.load(input_json_file)

    _header('Tokenizing input text and saving it as byte data to', token_data_file_name)
    final_tokens = []
    for entry in all_texts:
        tokens = [1] + sentencepiece_model.encode(entry["text"])
        # print(tokens)
        final_tokens.extend(tokens)

    final_tokens = numpy.array(final_tokens, dtype=numpy.uint16)

    with open(token_data_file_name, "wb") as f:
        f.write(final_tokens.tobytes())

def train(debug=False):

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    # https://docs.pytorch.org/docs/stable/generated/torch.set_printoptions.html
    torch.set_printoptions(threshold=40000, sci_mode=False, precision=8, linewidth=120)

    rope = RotaryPositionEmbedding(debug)
    print(rope.state_dict())

    model = torch.nn.Sequential()
    model.add_module('W0', torch.nn.Linear(8, 16))
    model.add_module('tanh', torch.nn.Tanh())
    model.add_module('W1', torch.nn.Linear(16, 1))

    x = torch.randn(1, 8)
    y = model(x)

    if debug:
        torchviz.make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(model_visualization_dot_name, format="png")
        onnx_export = torch.onnx.export(model, x, dynamo=True)
        onnx_export.save(model_visualization_onnx_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, choices=["words_counts_and_stats", "plain_text", "vocabulary_creation","to_tokens","train"])
    args = parser.parse_args()

    if args.function == "words_counts_and_stats":
        words_counts_and_stats(debug=False)
    elif args.function == "plain_text":
        plain_text()
    elif args.function == "vocabulary_creation":
        vocabulary_creation()
    elif args.function == "to_tokens":
        to_tokens()
    elif args.function == "train":
        #todo: debug as parameter
        train(debug=True)
    else:
        raise ValueError(f"Unknown argument {args.function}")

