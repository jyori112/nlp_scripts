#!/usr/bin/env python3
import argparse
from gensim.models import keyedvectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    model = keyedvectors.KeyedVectors.load_word2vec_format(args.input)
    model.save(args.output)

if __name__ == '__main__':
    main()
