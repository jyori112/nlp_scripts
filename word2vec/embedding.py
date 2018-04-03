import numpy as np
import scipy.stats
import scipy.spatial.distance

class Embedding:
    def __init__(self, word2id, emb):
        self.word2id = word2id
        self.emb = emb
        self.id2word = {idx: word for word, idx in word2id.items()}

    def get_emb(self, word):
        return self.emb[self.word2id[word]]

    def similar_by_vector(self, vec, topk=10):
        nrm_emb = self.emb / np.linalg.norm(self.emb, axis=1)[:, None]
        nrm_vec = vec / np.linalg.norm(vec)
        
        score = nrm_emb.dot(nrm_vec)
        return [(self.id2word[idx], score[idx]) for idx in np.argsort(-score)[:topk]]

    def eval_sim(self, wordsim):
        model = []
        gold = []
        with open(wordsim) as f:
            f.readline()
            for line in f:
                if not line:
                    continue

                word1, word2, sim = line.lower().split()

                try:
                    idx1 = self.word2id[word1]
                    idx2 = self.word2id[word2]

                    emb1 = self.emb[idx1]
                    emb2 = self.emb[idx2]

                    score = 1-scipy.spatial.distance.cosine(emb1, emb2)

                    model.append(score)
                    gold.append(float(sim))
                except KeyError:
                    pass
                
        return scipy.stats.pearsonr(model, gold)[0]

    def save_text(self, path):
        with open(path, 'w') as f:
            f.write('{} {}\n'.format(*self.emb.shape))

            for word, idx in self.word2id.items():
                vec_str = ' '.join('{:.6f}'.format(v) for v in self.emb[idx])
                f.write('{} {}\n'.format(word, vec_str))
    
    def save(self, path):
        joblib.dump(dict(word2id=self.word2id, emb=self.emb), path)

    @staticmethod
    def load(path):
        with open(path) as f:
            n_lines, dim = tuple(map(int, f.readline().split()))
            emb = np.zeros((n_lines, dim), dtype=np.float32)
            word2id = {}

            lines = [line for line in f.read().split('\n') if line]

            for i, line in enumerate(lines):
                word, vec_str = line.split(' ', 1)
                word2id[word] = i
                emb[i] = np.fromstring(vec_str, sep=' ')

        return Embedding(word2id, emb)
