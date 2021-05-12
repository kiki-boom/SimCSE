import tensorflow as tf
import numpy as np
import scipy
import tokenization
from utils import *


def l2_normalize(vecs):
    """标准化
    """
    norms = np.linalg.norm(vecs, axis=1)
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def similarity(d0, d1, tokenizer, max_sequence_length, model):
    token_ids0, mask0 = convert_to_ids(d0, tokenizer, max_sequence_length)
    token_ids1, mask1 = convert_to_ids(d1, tokenizer, max_sequence_length)
    segment_ids = [0] * len(token_ids0)
    vecs = model.predict([np.array([token_ids0, token_ids1]),
                          np.array([mask0, mask1]),
                          np.array([segment_ids, segment_ids])])
    vecs = l2_normalize(vecs)
    sim = (vecs[0] * vecs[1]).sum(axis=1)
    return sim


def evaluate(args):
    model = tf.keras.models.load_model(args.model_dir)
    tokenizer = tokenization.FullTokenizer(args.vocab_file)
    max_len = args.max_sequence_length
    data = load_data(args.input_file)
    sims = []
    labels = []
    for d0, d1, l in data:
        sim = similarity(d0, d1, tokenizer, max_len, model)
        sims.append(sim)
        labels.append(l)
    corrcoef = compute_corrcoef(labels, sims)
    print(corrcoef)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input dir for training data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="The dir where to save model weights")
    parser.add_argument("--vocab_file", type=str,
                        help="The vocab file")
    parser.add_argument("--max_sequence_length", type=int, default=128,
                        help="Maximum sequence length")
    args = parser.parse_args()
    evaluate(args)
