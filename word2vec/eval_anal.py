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
    parser.add_argument('emb', type=str)
    parser.add_argument('anal', type=str)

    args = parser.parse_args()

    
    model = keyedvectors.KeyedVectors.load_word2vec_format(args.emb)

    total_correct, total = 0, 0

    for section in model.accuracy(args.anal):
        section_correct = len(section['correct'])
        section_total = section_correct + len(section['incorrect'])

        total_correct += section_correct
        total += section_total

        logger.info('Section={}; Accuracy={:.3f} ({}/{})'.format(
            section['section'], section_correct / section_total, section_correct, section_total))

    logger.info('Total: Accuracy={:.3f} ({:.3f}/{:.3f})'.format(total_correct / total, total_correct, total))

if __name__ == '__main__':
    main()

