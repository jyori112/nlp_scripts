import argparse
from gensim.models import word2vec
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
    parser.add_argument('--dim', '-d', type=int, default=100)
    parser.add_argument('--window', '-w', type=int, default=5)
    parser.add_argument('--negative', '-n', type=int, default=5)
    parser.add_argument('--skipgram', '-sg', action='store_true')
    parser.add_argument('--worker', type=int, default=4)

    args = parser.parse_args()

    corpus = word2vec.LineSentence(args.input)

    logger.info('Train')
    sg = 1 if args.skipgram else 0
    model = word2vec.Word2Vec(corpus, size=args.dim, window=args.window, negative=5, sg=sg)

    logger.info('Save')
    model.wv.save_word2vec_format(args.output)

if __name__ == '__main__':
    main()

