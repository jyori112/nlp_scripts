#!/usr/bin/env python3
import argparse
from gensim.models import keyedvectors
from embedding import Embedding
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

    logger.info('Loading embedding')
    emb = Embedding.load(args.input)

    logger.info('Evaluating')
    pearson = emb.eval_sim(args.wordsim)

    logger.info('Pearson={:.3f}'.format(pearson))

if __name__ == '__main__':
    main()

