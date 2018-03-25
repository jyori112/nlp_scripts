#!/usr/bin/env python3
import jieba

import argparse
import sys
from multiprocessing.pool import Pool

def tokenize(line):
    try:
        return ' '.join(jieba.cut(line))
    except:
        print('Error in tokenize', file=sys.stderr)

def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--lang', '-l', type=str, default='en')
    parser.add_argument('--poolsize', '-p', type=int, default=10)

    args = parser.parse_args()

    if args.input:
        input_file = open(args.input)
    else:
        input_file = sys.stdin

    if args.output:
        output_file = open(args.output, 'w')
    else:
        output_file = sys.stdout

    with Pool(args.poolsize) as p:
        for line in p.imap(tokenize, input_file):
            print(line, file=output_file)

if __name__ == '__main__':
    main()
