#!/usr/bin/env python3

import argparse, joblib
from collections import Counter
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--min_count', '-min', type=int, default=5)

    args = parser.parse_args()

    logger.info('Load Data')
    with open(args.input) as f:
        c = Counter()
        for i, line in enumerate(f):
            c.update(line.split())

            if i % 100000 == 0:
                logger.info('Prog={}'.format(i))

    logger.info('Filter Low Freq Words')
    word2count = {word: count for word, count in c.items() if count >= args.min_count}

    logger.info('Save')
    joblib.dump(word2count, args.output)

if __name__ == '__main__':
    main()
