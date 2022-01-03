import os
import pickle
import dataclasses
import sys

from typing import List, Optional

import tensorflow as tf
import tensorflow_hub as hub
from official.core import config_definitions as cfg
import tensorflow_text as text  # Imports TF ops for preprocessing.

HOME = os.environ.get('MODEL_EXPERIMENT_HOME', None)
SEQ_LENGTH = 512


@dataclasses.dataclass
class PreprocessorConfig(cfg.DataConfig):
    seq_length: int = SEQ_LENGTH
    input_path: str = os.path.join(HOME, 'Data/raw')
    output_path: str = os.path.join(HOME, 'Data/preprocessed')
    vocab_path: str = os.path.join(HOME, 'Data/raw/save_vocab.pkl')
    preprocess_model_path: Optional[str] = os.path.join(HOME, "Models/BERT-Preprocess")


# TODO: how to add max_context?
class Preprocessor(object):

    def __init__(self, params: PreprocessorConfig, name='literature'):
        self._params = params
        self._splits = ['dev', 'train', 'test']
        self._features = ['input_type_ids', 'input_mask', 'input_word_ids', 'label_ids']
        self._info = {'name': name, 'features': self._features, 'labels': [], 'splits': {}}

        self._build_vocab()
        self.preprocess_model = None
        self.model = self._build_model()

    def process(self, number_of_examples: Optional[int] = None, splits: Optional[List[str]] = None, verbose: int = 0):
        """Preprocess raw data and save them in tfrecords
        Args:
            number_of_examples: max number of examples to include in each split (default: None)
            splits: list of splits to process, e.g. ['test', 'train']
            verbose: print records being processed
        Return:
            None
        """
        splits = self._splits if splits is None else splits
        for split in splits:
            print(f"Processing {split} split.")
            input_file = os.path.join(self._params.input_path, split + '.txt')
            output_file = os.path.join(self._params.output_path, split + '.tfrecord')
            assert os.path.isfile(input_file), f"`{split}` split not found"
            dataset = self._read_examples_from_file(input_file)

            with tf.io.TFRecordWriter(output_file) as file_writer:
                count = 0
                for record in dataset:
                    if number_of_examples is None or (number_of_examples is not None and count < number_of_examples):
                        if verbose:
                            sys.stdout.write(f"\rProcessing record: {count}")
                            sys.stdout.flush()
                        words, labels = tf.split(record, 2)
                        # words, labels = tf.strings.split(words).flat_values, tf.strings.split(labels).flat_values
                        processed_record = self.model([words, labels])
                        file_writer.write(self._serialize_example(processed_record))
                        count += 1
                    else:
                        break
                self._info['splits'][split] = count
                if verbose:
                    print()

            # Save dataset_info
            with open(os.path.join(self._params.output_path, 'dataset_info.pkl'), 'wb') as f:
                pickle.dump(self._info, f)

    def _build_vocab(self):
        """Builds label to id vocab."""
        with open(self._params.vocab_path, 'rb') as f:
            dump = pickle.load(f)
        _labels = []
        _labels_id = []
        for k, v in dump['label2id'].items():
            _labels.append(k)
            _labels_id.append(v)

        self._info['labels'] = _labels

        self.id2label = {v: k for k, v in dump['label2id'].items()}
        self.label2id = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(_labels, dtype=tf.string),
                values=tf.constant(_labels_id, dtype=tf.int64),
                key_dtype=tf.string,
                value_dtype=tf.int64),
            num_oov_buckets=5)

    @staticmethod
    def _read_examples_from_file(input_file: str) -> tf.data.Dataset:
        """Read a BIO-Formatted data!

        Args: input_file

        Returns: list of data_instances ((words, labels))
        """
        with open(input_file, 'r') as rf:

            lines = rf.readlines()

            data, words, labels = [], [], []
            line_no = 0

            while line_no < len(lines):
                line = lines[line_no]
                _line = line.strip().split('\t')
                words.append(_line[0])
                labels.append(_line[-1])

                # check if document ends
                if len(line) == 1:
                    _label = ' '.join([label for label in labels if len(label) > 0])
                    _word = ' '.join([word for word in words if len(word) > 0])
                    data.append((_word, _label))
                    words = []
                    labels = []

                    # skip two
                    line_no += 2

                line_no += 1

        return tf.data.Dataset.from_tensor_slices(data)

    def _build_model(self) -> tf.keras.Model:
        """Returns Model mapping string features to BERT inputs.

        Args: self

        Returns: A Keras Model that can be called on a list or dict of string Tensors
        and returns a dict of tensors for input to BERT.
        """

        # create `words` and `labels` Input layers
        words = tf.keras.layers.Input(shape=(), dtype=tf.string, name='words')
        labels = tf.keras.layers.Input(shape=(), dtype=tf.string, name='labels')

        if self.preprocess_model is None:
            self._init_preprocess_model()

        # create tokenizer layer
        tokenizer = hub.KerasLayer(self.preprocess_model.tokenize, name='tokenizer')
        tokenized_words = tokenizer(words)

        # add a lambda layer to split labels and apply label 'X' label to word pieces
        def labeler(x):
            # x[0] = tf.expand_dims(tf.squeeze(x[0], axis=1), axis=0)
            x[1] = tf.cast(x[1], dtype=tf.string)
            x[1] = tf.strings.split(tf.strings.lower(x[1]))
            x[1] = tf.ragged.map_flat_values(lambda a: self.label2id[a], x[1])
            x[1] = tf.cast(x[1], dtype=tf.int32)
            x[1] = tf.expand_dims(x[1], axis=-1)

            # x[1] = tf.expand_dims(tf.squeeze(x[1], axis=1), axis=0)

            def map_batch(y):
                def map_sentence(z):
                    # `-1` is for word pieces other than head word piece
                    x_tokens = tf.fill(tf.math.subtract(tf.shape(z[0]), 1), value=-1)
                    label_expanded = tf.concat([z[1], x_tokens], axis=0)
                    return label_expanded

                y = tf.map_fn(fn=map_sentence, elems=y,
                              fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0, dtype=tf.int32, shape=(None,)))
                return y

            x = tf.map_fn(fn=map_batch, elems=x,
                          fn_output_signature=tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32))
            return x

        expanded_labels = tf.keras.layers.Lambda(labeler, name='labeler')([tokenized_words, labels])

        # TODO: implement sliding window

        # Pack inputs layer. The details (start/end token ids, dict of output tensors)
        # are model-dependent, so this gets loaded from the SavedModel.
        packer = hub.KerasLayer(self.preprocess_model.bert_pack_inputs,
                                arguments=dict(seq_length=self._params.seq_length),
                                name='packer')
        packed_inputs = packer([tokenized_words])

        # Pack label_ids layer
        def pack_label_id(x):
            # flattened and trimmed to sequence length
            flattened = x[1].flat_values[:self._params.seq_length - 2]
            # start_of_sequence_id = tf.expand_dims(self.special_tokens_dict['start_of_sequence_id'], axis=0)
            # end_of_segment_id = tf.expand_dims(self.special_tokens_dict['end_of_segment_id'], axis=0)
            start_of_sequence_id = tf.constant([-100])
            end_of_segment_id = tf.constant([-100])
            flattened = tf.concat([start_of_sequence_id, flattened, end_of_segment_id], axis=0)
            padded = tf.pad(flattened, [[0, self._params.seq_length - tf.shape(flattened)[-1]]], constant_values=-100)
            x[0]['label_ids'] = tf.expand_dims(padded, axis=0)
            return x[0]

        model_output = tf.keras.layers.Lambda(pack_label_id, name='pack_label_id')([packed_inputs, expanded_labels])

        return tf.keras.Model([words, labels], model_output)

    def _init_preprocess_model(self) -> None:
        """Returns BERT Preprocess Model"""
        if self._params.preprocess_model_path is None:
            raise ValueError('`preprocess_model_path` is not defined')
        self.preprocess_model = hub.load(self._params.preprocess_model_path)
        self.special_tokens_dict = self.preprocess_model.tokenize.get_special_tokens_dict()

    def _serialize_example(self, example):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        features = {f: _int64_feature(tf.io.serialize_tensor(example[f]).numpy()) for f in self._features}

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    @staticmethod
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def tokenization_diff(self):
        for split in self._splits:
            print(f"Processing {split} split.")
            input_file = os.path.join(self._params.input_path, split + '.txt')
            dataset = self._read_examples_from_file(input_file)

            count = 0
            for record in dataset:
                is_diff = self._tokenization_diff(record)
                if not is_diff[0]:
                    print(is_diff[1], is_diff[2])
                    print(record[0])
                count += 1

    def _tokenization_diff(self, record: List[tf.Tensor]):
        """Difference between bert tokenization and nltk tokenization"""
        n_nltk = tf.cast(tf.shape(tf.strings.split(record[1])), dtype=tf.int32)
        n_bert = tf.cast(self.preprocess_model.tokenize([record[0]]).bounding_shape()[1:2], dtype=tf.int32)

        return tf.math.equal(n_nltk, n_bert).numpy()[0], n_nltk.numpy()[0], n_bert.numpy()[0]


if __name__ == '__main__':
    preprocessor = Preprocessor(PreprocessorConfig())
    preprocessor.process()
