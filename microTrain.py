import os
import argparse
import json

def plain_text():
    print("Loading input json file")
    input_file = open(os.path.join("data-input", "sentences.json"), "r")
    input_sentences = json.load(input_file)
    input_file.close()

    print("Parsing to a plain text file")
    plain_file = open(os.path.join("data-work", "plain_text_sentences.txt"), "w")
    for text in input_sentences:
        text = text["text"]
        plain_file.write(text)
        plain_file.write("\n")
    plain_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, choices=["plain_text"])
    args = parser.parse_args()

    if args.function == "plain_text":
        plain_text()
    else:
        raise ValueError(f"Unknown argument {args.function}")

