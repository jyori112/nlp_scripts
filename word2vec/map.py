#!/usr/bin/env python3
import argparse
from gensim.models import word2vec, keyedvectors
import logging, sys
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)

def write_embedding(f, vocab, matrix):
    f.write('{} {}\n'.format(*matrix.shape))

    for word, item in vocab.items():
        vec_str = ' '.join('{:.6}'.format(v) for v in matrix[item.index])

        f.write('{} {}\n'.format(word, vec_str))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('trg', type=str)
    parser.add_argument('src_output', type=str)
    parser.add_argument('trg_output', type=str)
    parser.add_argument('--dictionary', '-d', type=str)

    args = parser.parse_args()

    src_emb = keyedvectors.KeyedVectors.load_word2vec_format(args.src)
    trg_emb = keyedvectors.KeyedVectors.load_word2vec_format(args.trg)

    src_nrm = src_emb.syn0 / np.linalg.norm(src_emb.syn0, axis=1)[:, None]
    trg_nrm = trg_emb.syn0 / np.linalg.norm(trg_emb.syn0, axis=1)[:, None]

    src_idx = []
    trg_idx = []

    oov = 0
    with open(args.dictionary) as f:
        for line in f:
            src, trg = line.split()

            if src in src_emb.vocab and trg in trg_emb.vocab:
                src_idx.append(src_emb.vocab[src].index)
                trg_idx.append(trg_emb.vocab[trg].index)
            else:
                oov += 1

    src_idx = np.array(src_idx, dtype=np.int32)
    trg_idx = np.array(trg_idx, dtype=np.int32)

    src_matrix = src_nrm[src_idx]
    trg_matrix = trg_nrm[trg_idx]

    u, s, vt = np.linalg.svd(np.dot(trg_matrix.T, src_matrix))
    w = np.dot(vt.T, u.T)
    src_matrix_w = src_nrm.dot(w)
    trg_matrix_w = trg_nrm

    with open(args.src_output, 'w') as f:
        write_embedding(f, src_emb.vocab, src_matrix_w)

    with open(args.trg_output, 'w') as f:
        write_embedding(f, trg_emb.vocab, trg_matrix_w)


if __name__ == '__main__':
    main()

