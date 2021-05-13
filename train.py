import tensorflow as tf
import simcse_model
import logging
from kerasbert import model_loader
from config import encoder_config
from utils import *
from data_processor import *
from pathlib import Path
import tokenization

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_model(args):
    encoder_config["dropout_rate"] = args.dropout_rate
    encoder_config["max_sequence_length"] = args.max_sequence_length

    assert args.pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    if args.pooling == "pooler":
        model = simcse_model.SimCSEPool(
            encoder_config=encoder_config,
        )
    elif args.pooling == "last_avg":
        model = simcse_model.SimCSELastAvg(
            encoder_config=encoder_config,
        )
    elif args.pooling == "last_first_avg":
        model = simcse_model.SimCSEFirstLastAvg(
            encoder_config=encoder_config,
        )
    elif args.pooling == "cls":
        model = simcse_model.SimCSECLS(
            encoder_config=encoder_config,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=simcse_model.SimLoss(),
    )
    return model


def init_model(model, args):
    latest_ckpt = tf.train.latest_checkpoint(args.model_dir)
    if latest_ckpt:
        logging.info("Initial model from %s" % latest_ckpt)
        model.load_weights(latest_ckpt)
    elif args.bert_ckpt:
        logging.info("Load BERT weights from %s" % args.bert_ckpt)
        model_loader.load_block_weights_from_official_checkpoint(
            model.encoder, encoder_config, args.bert_ckpt
        )


def get_format_fun(batch_size, sequence_length):
    def format_data(batch_ids, batch_mask):
        ids = tf.broadcast_to(batch_ids, [2, batch_size, sequence_length])
        ids = tf.transpose(ids, [1, 0, 2])
        ids = tf.reshape(ids, [-1, sequence_length])
        mask = tf.broadcast_to(batch_mask, [2, batch_size, sequence_length])
        mask = tf.transpose(mask, [1, 0, 2])
        mask = tf.reshape(mask, [-1, sequence_length])
        segment_ids = tf.zeros_like(ids)
        labels = tf.zeros_like(ids[:, :1])
        return (ids, mask, segment_ids), labels
    return format_data


def load_dataset(args):
    delimiter = delimiters[args.task]
    all_data = {
        "%s-%s" % (args.task, f):
        load_csv_data("%s/%s/%s.csv" % (args.data_dir, args.task, f), delimiter)
        for f in ["train", "dev", "test"]
    }
    tokenizer = tokenization.FullTokenizer(args.vocab_file)
    idses = []
    masks = []
    for task_name, data in all_data.items():
        for d0, d1, l in data:
            token_ids, mask = convert_to_ids(d0, tokenizer, args.max_sequence_length)
            idses.append(token_ids)
            masks.append(mask)
            token_ids, mask = convert_to_ids(d1, tokenizer, args.max_sequence_length)
            idses.append(token_ids)
            masks.append(mask)
    dataset = tf.data.Dataset.from_tensor_slices((idses, masks))
    dataset = dataset.batch(args.batch_size, drop_remainder=True).map(
        get_format_fun(args.batch_size, args.max_sequence_length)
    ).unbatch()
    return dataset


def load_tf_data(args):
    input_files = [str(f) for f in Path(args.data_dir).iterdir()]
    dataset = read_tf_record(input_files, args.max_sequence_length)
    dataset = dataset.batch(args.batch_size, drop_remainder=True).map(
        get_format_fun(args.batch_size, args.max_sequence_length)
    ).unbatch()
    return dataset

def train(args):
    model = make_model(args)
    #dataset = load_dataset(args)
    dataset = load_tf_data(args)
    dataset = dataset.batch(args.batch_size * 2)
    model.predict(dataset.take(1))
    init_model(model, args)
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs",
            histogram_freq=0,
            embeddings_freq=0,
            update_freq=1,
        )
    ]
    model.fit(dataset,
              epochs=args.epochs,
              callbacks=callbacks)
    model.save(args.model_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input dir for training data")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="The dir where to save model weights")
    parser.add_argument("--bert_ckpt", type=str,
                        help="The Bert checkpoint file")
    parser.add_argument("--vocab_file", type=str,
                        help="The vocab file")
    parser.add_argument("--max_sequence_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epoch to train")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--pooling", type=str, default="pooler",
                        help="Pooling type")
    args = parser.parse_args()
    train(args)
