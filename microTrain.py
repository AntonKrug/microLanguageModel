import argparse
import heapq
from collections import OrderedDict, Counter
from itertools import count
import json
import os
import re

import pandas
import sentencepiece

vocabulary_size = 4008

input_data_json_file_name = os.path.join("data-input", "sentences.json")
plain_text_data_file_name = os.path.join("data-work", "plain_text_sentences.txt")
vocabulary_file_name =  os.path.join("data-work", "vocabulary")

word_counts_total = Counter()
first_word_total = Counter()
letter_counts_total = Counter()

def panda_split_to_words(text):
    # text = re.sub(r"[^\w\s']", '', text)  # remove punctuation
    words = text.split()
    return Counter(words)


def panda_first_word(text):
    words = text.split()
    if len(words) > 0:
        return words[0]
    else:
        print('ERROR, bad sentence encountered', text)
        return "_ERROR_"


def header(*args, **kwargs):
    print()
    print("-"*70)
    print(*args, **kwargs)


def build_huffman_table(counter):
    heap = []
    _count = count()

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
            left, right = node
            assign_codes(left, prefix + "0")
            assign_codes(right, prefix + "1")

    assign_codes(root)
    return table


def count_words():
    header('Loading input json file as panda dataframe', input_data_json_file_name)
    df = pandas.read_json(input_data_json_file_name)

    #---------------------- word count -------------------------------
    header("Counting words")

    print('Splitting texts to words')
    df['words'] = df['text'].apply(panda_split_to_words)

    print('Counting words in each text')
    df['word_counts'] = df['words'].apply(Counter)

    print('Adding up words for all texts')
    for count_in_one_text in df['words']:
        word_counts_total.update(count_in_one_text)

    i = 1
    for word in sorted(word_counts_total, key=word_counts_total.get, reverse=True):
        print(i, word, ' => ', word_counts_total[word])
        i=i+1

    print('Words used in texts', len(word_counts_total))
    if vocabulary_size == len(word_counts_total):
        print('Matching hard-coded vocabulary size')
    else:
        print('Not matching the hard-coded vocabulary size', vocabulary_size)

    # --------------------- first words ------------------------------------
    header('Counting first word of each text')
    df['first'] = df['text'].apply(panda_first_word)

    for first in df['first']:
        first_word_total.update([first])

    for word in sorted(first_word_total, key=first_word_total.get, reverse=True):
        print(word, ' => ', first_word_total[word])

    # --------------------- letter statistics ------------------------------------
    header('Counting letter usage of all words')
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

        letter_counts_total.update(word)
        letter_counts_total.update('0')

    total_char_used_for_vocabulary = 0
    for letter in sorted(letter_counts_total, key=letter_counts_total.get, reverse=True):
        total_char_used_for_vocabulary += letter_counts_total[letter]
        print(letter, ' => ', letter_counts_total[letter])
    print(f"Total amount of characters used to store whole vocabulary: {total_char_used_for_vocabulary}")

    header('Huffman table creation')
    huffman_table = build_huffman_table(letter_counts_total)
    total_huffman_bits_used = 0
    max_len_of_bits = 0
    for ch, code in huffman_table.items():
        bits_used = len(code) * letter_counts_total[ch]
        total_huffman_bits_used += bits_used
        if max_len_of_bits < len(code):
            max_len_of_bits = len(code)
        print(f"{ch!r}: {code}, length {len(code)} x frequency {letter_counts_total[ch]} = {bits_used} bits used")

    # 2 bytes for codeword (upto 14bits) + 1 byte for codeword mask (only 4bits needed) + 1 byte for the output character
    huffman_table_bytes = 4 * len(huffman_table.items())
    total_huffman_bytes_used = int( (total_huffman_bits_used + 4) / 8 )
    print(f"Codeword max length {max_len_of_bits}, total bits used as indexes to codewords {total_huffman_bits_used}. "
          f"Bytes used as indexes to codewords {total_huffman_bytes_used} + huffman table {huffman_table_bytes} => "
          f"total {huffman_table_bytes + total_huffman_bytes_used} bytes")

    huffman_saves = total_char_used_for_vocabulary - (huffman_table_bytes + total_huffman_bytes_used)
    print(f"Non huffman approach {total_char_used_for_vocabulary} bytes - huffman "
          f"{huffman_table_bytes + total_huffman_bytes_used} bytes = huffman saves ~{huffman_saves} bytes (ignoring "
          f"the fact that huffman decoder needs to be implemented in the firmware)")

    header('Huffman table ordered for the firmware')
    for ch, code in sorted(huffman_table.items(), key=lambda x: (len(x[1]))):
        print(f"{{ .code=0'b{code.ljust(16, '0')}, .bits={str(len(code)).zfill(2)} .character='{ch}' }}, // {code}")

    return df


def plain_text():
    header('Loading input json file', input_data_json_file_name)
    input_json_file = open(input_data_json_file_name, 'r')
    input_sentences = json.load(input_json_file)
    input_json_file.close()

    header('Parsing to a plain text file', plain_text_data_file_name)
    plain_file = open(plain_text_data_file_name, 'w')
    for text in input_sentences:
        text = text["text"]
        plain_file.write(text)
        plain_file.write("\n")
    plain_file.close()
    print('Processed', len(input_sentences), 'sentences')


def vocabulary_creation():
    header("Preparing vocabulary tokens")
    # https://github.com/google/sentencepiece
    # https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
    # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.tokenizers.SentencePiece.html
    # https://medium.com/ai-enthusiast/unlocking-the-power-of-sentencepiece-encoding-a-comprehensive-guide-c181bb3d75ee
    # https://github.com/google/sentencepiece/issues/412
    # https://github.com/google/sentencepiece/issues/121
    # https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L193
    # https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb

    sentencepiece.SentencePieceTrainer.train(input=plain_text_data_file_name,
                                             model_prefix=vocabulary_file_name,
                                             model_type="word",
                                             vocab_size=vocabulary_size + 1, # UNK and BOS
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, choices=["count_words", "plain_text", "vocabulary_creation"])
    args = parser.parse_args()

    if args.function == "count_words":
        count_words()
    elif args.function == "plain_text":
        plain_text()
    elif args.function == "vocabulary_creation":
        vocabulary_creation()
    else:
        raise ValueError(f"Unknown argument {args.function}")

