#!/usr/bin/env python3
import argparse
from collections import defaultdict
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
    parser.add_argument('src', type=str)
    parser.add_argument('trg', type=str)
    parser.add_argument('--dictionary', '-d', type=str)

    args = parser.parse_args()

    logger.info('Load src embedding')
    src_emb = Embedding.load(args.src)

    logger.info('Load trg embedding')
    trg_emb = Embedding.load(args.trg)

    logger.info('Load dictionary')
    dictionary = defaultdict(list)

    with open(args.dictionary) as f:
        for line in f:
            src, trg = line.split()
            dictionary[src].append(trg)

    logger.info('Evaluate')
    total = 0
    hit = 0
    oov = 0

    for src, trgs in dictionary.items():
        try:
            src_vec = src_emb.get_emb(src)
            trans = trg_emb.similar_by_vector(src_vec)
            total += 1
            if any(tran_word in trgs for tran_word, score in trans):
                hit += 1
        except KeyError:
            oov += 1

    logger.info('Accuracy={:.3f}'.format(hit / total))

if __name__ == '__main__':
    main()

