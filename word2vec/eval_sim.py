#!/usr/bin/env python3
import argparse
from gensim.models import keyedvectors
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('wordsim', type=str)

    args = parser.parse_args()

    emb = keyedvectors.KeyedVectors.load_word2vec_format(args.input)

    pearson, spearman, oov_rate = emb.evaluate_word_pairs(args.wordsim)

    logger.info('Coverage={:.3}; Pearson={:.3f}; Spearman={:.3f}'.format(1-oov_rate/100, pearson[0], spearman[0]))

if __name__ == '__main__':
    main()

