from typing import Mapping, Optional, Tuple, Dict

import official.core.config_definitions as cfg
from official.nlp.data import data_loader
from official.core import input_reader
import tensorflow as tf


class TaggingDataLoader(data_loader.DataLoader):
    """A class to load dataset for tagging (e.g., NER and POS) task."""

    def __init__(self, params: cfg.DataConfig):
        self._params = params
        self._seq_length = params.seq_length
        self._include_sentence_id = params.include_sentence_id

    def _decode(self, record: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Decodes a serialized tf.Example."""
        name_to_features = {
            'input_word_ids': tf.io.FixedLenFeature([], tf.string),
            'input_mask': tf.io.FixedLenFeature([], tf.string),
            'input_type_ids': tf.io.FixedLenFeature([], tf.string),
            'label_ids': tf.io.FixedLenFeature([], tf.string),
        }
        if self._include_sentence_id:
            name_to_features['sentence_id'] = tf.io.FixedLenFeature([], tf.int64)
            name_to_features['sub_sentence_id'] = tf.io.FixedLenFeature([], tf.int64)

        example = tf.io.parse_single_example(record, name_to_features)

        # parse binary_string back to tensors
        for name in example:
            example[name] = tf.io.parse_tensor(example[name], out_type=tf.int32)
            example[name] = tf.squeeze(example[name])

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in example:
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def _parse(self, record: Mapping[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Parses raw tensors into a dict of tensors to be consumed by the model."""
        x = {
            'input_word_ids': record['input_word_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['input_type_ids']
        }
        if self._include_sentence_id:
            x['sentence_id'] = record['sentence_id']
            x['sub_sentence_id'] = record['sub_sentence_id']

        y = record['label_ids']
        return x, y

    def load(self, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
        """Returns a tf.dataset.Dataset."""
        reader = input_reader.InputReader(
            params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
        return reader.read(input_context)
