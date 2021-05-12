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


def encode(texts, tokenizer, max_sequence_length, model):
    def _format_fun(_ids, _mask, _segment):
        return (_ids, _mask, _segment), 0
    ids, masks = [], []
    for text in texts:
        token_ids, mask = convert_to_ids(text, tokenizer, max_sequence_length)
        ids.append(token_ids)
        masks.append(mask)
    segment_ids = np.zeros_like(ids)
    dataset = tf.data.Dataset.from_tensor_slices((ids, masks, segment_ids))
    dataset = dataset.map(_format_fun).batch(32)
    vecs = model.predict(dataset)
    return vecs


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
    all_data = {
        "%s-%s" % (args.task, f):
        load_data("%s/%s/%s" % (args.data_dir, args.task, f))
        for f in ["train", "dev", "test"]
    }

    model = tf.keras.models.load_model(args.model_dir)
    tokenizer = tokenization.FullTokenizer(args.vocab_file)
    max_len = args.max_sequence_length

    for name, data in all_data.items():
        vecs0 = encode([d0 for d0, _, _ in data], tokenizer, max_len, model)
        vecs1 = encode([d1 for _, d1, _ in data], tokenizer, max_len, model)
        labels = [l for _, _, l in data]
        vecs0 = l2_normalize(vecs0)
        vecs1 = l2_normalize(vecs1)
        sims = (vecs0 * vecs1).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)
        print("%s: %s" % (name, corrcoef))


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
