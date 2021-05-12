import tensorflow as tf
from utils import *
import logging
import tokenization
from pathlib import Path


def create_int_features(values):
    feature = tf.train.Features(int64_list=tf.train.Int64List(values=values))
    return feature


def create_tf_example(token_ids, mask):
    tf_example = tf.train.Example(tf.train.Features(
        {"token_ids": create_int_features(token_ids),
         "mask": create_int_features(mask)}
    ))
    return tf_example


def dump_feature(features, tokenizer):
    dump_str = ""
    words = tokenizer.convert_ids_to_tokens(features["token_ids"].int64_list.value)
    dump_str += "words: " + " ".join(words) + "\n"
    input_mask = features["input_mask"].int64_list.value
    dump_str += "mask: " + " ".join([str(v) for v in input_mask])
    return dump_str


def write_to_tf_files(data, tokenizer, max_sequence_length, output_files):
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    total_written = 0
    for d0, d1, l in data:
        writer = writers[total_written % len(writers)]
        token_ids, mask = convert_to_ids(d0, tokenizer, max_sequence_length)
        example0 = create_tf_example(token_ids, mask)
        writer.write(example0.SerializeToString())
        token_ids, mask = convert_to_ids(d1, tokenizer, max_sequence_length)
        example1 = create_tf_example(token_ids, mask)
        writer.write(example1.SerializeToString())

        total_written += 1
        if total_written < 5:
            logging.info("*** Example ***")
            dump_str0 = dump_feature(example0.features, tokenizer)
            dump_str1 = dump_feature(example1.features, tokenizer)
            logging.info("d0: %s\nd1: %s" % (dump_str0, dump_str1))
        if total_written % 500 == 0:
            logging.info("Writing example %d" % total_written)

    for writer in writers:
        writer.close()


def read_tf_record(file_names, max_sequence_length):
    feature_discription = {
        "token_ids": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "mask": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
    }

    def _parse_fun(record):
        features = tf.io.parse_single_example(record, feature_discription)
        token_ids = tf.cast(features["token_ids"], tf.int32)
        mask = tf.cast(features["mask"], tf.int32)
        return token_ids, mask

    dataset = tf.data.TFRecordDataset(filenames=file_names)
    dataset = dataset.map(_parse_fun)
    return dataset


def get_output_files(output_dir, name, num_writer=1, index=0):
    output_files = []
    for i in range(num_writer):
        output_files.append(output_dir + "/" + name + "_{:0>3d}".format(index * num_writer + i))
    return output_files


def create_tf_record_dataset(args):
    input_files = []
    for f in Path(args.input_dir).iterdir():
        if f.is_file():
            input_files.append(f)

    tokenizer = tokenization.Tokenizer(args.vocab_file)

    for f in input_files:
        data = load_data(str(f))
        name = f.parts[-1]
        output_files = get_output_files(args.output_dir, name=name, num_writer=1)
        write_to_tf_files(data, tokenizer, args.max_sequence_length, output_files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input dir")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output dir where to save tfRecord")
    parser.add_argument("-v", "--vocab_file", type=str, required=True,
                        help="The vocab file for char tokenizer")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Maximum sequence length")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_tf_record_dataset(args)
