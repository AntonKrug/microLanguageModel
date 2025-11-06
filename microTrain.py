import os
import argparse
import json

import sentencepiece

vocabulary_size = 4445

input_data_json_file_name = os.path.join("data-input", "sentences.json")
plain_text_data_file_name = os.path.join("data-work", "plain_text_sentences.txt")
vocabulary_file_name =  os.path.join("data-work", "vocabulary")

def plain_text():
    print("Loading input json file")
    input_json_file = open(input_data_json_file_name, "r")
    input_sentences = json.load(input_json_file)
    input_json_file.close()

    print("Parsing to a plain text file")
    plain_file = open(plain_text_data_file_name, "w")
    for text in input_sentences:
        text = text["text"]
        plain_file.write(text)
        plain_file.write("\n")
    plain_file.close()

def vocabulary():
    print("Preparing vocabulary tokens")
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
                                             vocab_size=vocabulary_size,
                                             self_test_sample_size=0,
                                             input_format="text",
                                             character_coverage=1.0,
                                             num_threads=os.cpu_count(),
                                             split_digits=True,
                                             allow_whitespace_only_pieces=True,
                                             byte_fallback=False,
                                             unk_surface="__UNKNOWN__",
                                             bos_id=-1,  # maybe i will need the beginning
                                             eos_id=-1,  # maybe i will get away without the end
                                             normalization_rule_name="identity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, choices=["plain_text", "vocabulary"])
    args = parser.parse_args()

    if args.function == "plain_text":
        plain_text()
    elif args.function == "vocabulary":
        vocabulary()
    else:
        raise ValueError(f"Unknown argument {args.function}")

