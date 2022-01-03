import os
import pickle
import dataclasses

from typing import Optional, List, Tuple

import orbit
import tensorflow as tf
from official.core import exp_factory, task_factory  # for defining experiment
from official.core import base_task  # for defining TaggingTask
from official.nlp.modeling import models  # for BertTokenClassifier
from official.nlp.tasks import utils  # for predict and get_encoder_from_hub
from official.nlp.configs import encoders  # for EncoderConfig and build_encoder
from official.modeling.hyperparams import base_config, OneOfConfig  # for defining ModelConfig
from official.modeling.optimization.configs import optimization_config  # for OptimizationConfig
import official.core.config_definitions as cfg  # for DataConfig, TaskConfig, ExperimentConfig, TrainerConfig
from tagging_data_loader import TaggingDataLoader
from seqeval import metrics as seqeval_metrics

HOME = os.environ.get('MODEL_EXPERIMENT_HOME', None)
SEQ_LENGTH = 512


def _masked_labels_and_weights(y_true):
    """Masks negative values from token level labels.

    Args:
      y_true: Token labels, typically shape (batch_size, seq_len), where tokens
        with negative labels should be ignored during loss/accuracy calculation.

    Returns:
      (masked_y_true, masked_weights) where `masked_y_true` is the input
      with each negative label replaced with zero and `masked_weights` is 0.0
      where negative labels were replaced and 1.0 for original labels.
    """
    # Ignore the classes of tokens with negative values.
    mask = tf.greater_equal(y_true, 0)
    # Replace negative labels, which are out of bounds for some loss functions,
    # with zero.
    masked_y_true = tf.where(mask, y_true, 0)
    return masked_y_true, tf.cast(mask, tf.float32)


@dataclasses.dataclass
class EncoderConfig(OneOfConfig):
    """Encoder configuration."""
    type: Optional[str] = "bert"
    bert: encoders.BertEncoderConfig = encoders.BertEncoderConfig(max_position_embeddings=SEQ_LENGTH)


@dataclasses.dataclass
class ModelConfig(base_config.Config):
    """A base span labeler configuration."""
    encoder: encoders.EncoderConfig = EncoderConfig()
    head_dropout: float = 0.1
    head_initializer_range: float = 0.02


@dataclasses.dataclass
class TaggingConfig(cfg.TaskConfig):
    """The model config."""
    # At most one of `init_checkpoint` and `hub_module_url` can be specified.
    init_checkpoint: str = ''
    hub_module_url: str = ''
    model: ModelConfig = ModelConfig()

    # The real class names, the order of which should match real label id.
    # Note that a word may be tokenized into multiple word_pieces tokens, and
    # we assume the real label id (non-negative) is assigned to the first token
    # of the word, and a negative label id is assigned to the remaining tokens.
    # The negative label id will not contribute to loss and metrics.
    class_names: Optional[List[str]] = None
    train_data: cfg.DataConfig = cfg.DataConfig()
    validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(TaggingConfig)
class TaggingTask(base_task.Task):
    """Task object for tagging (e.g., NER or POS)."""

    def build_model(self) -> tf.keras.Model:
        if self.task_config.hub_module_url and self.task_config.init_checkpoint:
            raise ValueError('At most one of `hub_module_url` and '
                             '`init_checkpoint` can be specified.')
        if self.task_config.hub_module_url:
            encoder_network = utils.get_encoder_from_hub(
                self.task_config.hub_module_url)
        else:
            encoder_network = encoders.build_encoder(self.task_config.model.encoder)

        return models.BertTokenClassifier(
            network=encoder_network,
            num_classes=len(self.task_config.class_names),
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.task_config.model.head_initializer_range),
            dropout_rate=self.task_config.model.head_dropout,
            output='logits',
            output_encoder_outputs=False)

    # TODO: build crf loss function
    def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
        logits = tf.cast(model_outputs['logits'], tf.float32)
        masked_labels, masked_weights = _masked_labels_and_weights(labels)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            masked_labels, logits, from_logits=True)
        numerator_loss = tf.reduce_sum(loss * masked_weights)
        denominator_loss = tf.reduce_sum(masked_weights)
        loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
        return loss

    def build_inputs(self, params: cfg.DataConfig, input_context=None):
        """Returns tf.data.Dataset for sentence_prediction task."""
        return TaggingDataLoader(params).load(input_context)

    def inference_step(self, inputs, model: tf.keras.Model):
        """Performs the forward step."""
        logits = model(inputs, training=False)['logits']
        return {'logits': logits,
                'predict_ids': tf.argmax(logits, axis=-1, output_type=tf.int32)}

    def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
        """Validatation step.

        Args:
          inputs: a dictionary of input tensors.
          model: the keras.Model.
          metrics: a nested structure of metrics objects.

        Returns:
          A dictionary of logs.
        """
        features, labels = inputs
        outputs = self.inference_step(features, model)
        loss = self.build_losses(labels=labels, model_outputs=outputs)

        # Negative label ids are padding labels which should be ignored.
        real_label_index = tf.where(tf.greater_equal(labels, 0))
        predict_ids = tf.gather_nd(outputs['predict_ids'], real_label_index)
        label_ids = tf.gather_nd(labels, real_label_index)
        return {
            self.loss: loss,
            'predict_ids': predict_ids,
            'label_ids': label_ids,
        }

    def aggregate_logs(self, state=None, step_outputs=None):
        """Aggregates over logs returned from a validation step."""
        if state is None:
            state = {'predict_class': [], 'label_class': []}

        def id_to_class_name(batched_ids):
            class_names = []
            for per_example_ids in batched_ids:
                class_names.append([])
                for per_token_id in per_example_ids.numpy().tolist():
                    class_names[-1].append(self.task_config.class_names[per_token_id])

            return class_names

        # Convert id to class names, because `seqeval_metrics` relies on the class
        # name to decide IOB tags.
        state['predict_class'].extend(id_to_class_name(step_outputs['predict_ids']))
        state['label_class'].extend(id_to_class_name(step_outputs['label_ids']))
        return state

    # TODO: calculate custom metrics / fix this metrics
    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Reduces aggregated logs over validation steps."""
        label_class = aggregated_logs['label_class']
        predict_class = aggregated_logs['predict_class']
        return {
            'f1':
                seqeval_metrics.f1_score(label_class, predict_class),
            'precision':
                seqeval_metrics.precision_score(label_class, predict_class),
            'recall':
                seqeval_metrics.recall_score(label_class, predict_class),
            'accuracy':
                seqeval_metrics.accuracy_score(label_class, predict_class),
        }


@dataclasses.dataclass
class BertNerConfig(object):
    seq_length: int = SEQ_LENGTH
    global_batch_size: int = 1
    include_sentence_id: bool = False
    input_path: str = os.path.join(HOME, 'Data/preprocessed')
    pretrained_model_path: str = os.path.join(HOME, "Models/BERT-PubMed")


def get_task(mode: str = 'train') -> TaggingConfig:
    _params = BertNerConfig()

    with open(os.path.join(_params.input_path, 'dataset_info.pkl'), 'rb') as f:
        dataset_info = pickle.load(f)

    model_config = ModelConfig({
        # 'encoder': encoders.EncoderConfig({'type': 'bert'}),
    })

    if mode == 'train':
        train_data_config = cfg.DataConfig({
            'input_path': os.path.join(_params.input_path, 'train.tfrecord'),
            'is_training': True,
            'global_batch_size': _params.global_batch_size,
            'seq_length': _params.seq_length,
            'include_sentence_id': _params.include_sentence_id
        })

        valid_data_config = cfg.DataConfig({
            'input_path': os.path.join(_params.input_path, 'train.tfrecord'),
            'is_training': False,
            'global_batch_size': _params.global_batch_size,
            'seq_length': _params.seq_length,
            'include_sentence_id': _params.include_sentence_id
        })

        task_config = TaggingConfig({
            'init_checkpoint': _params.pretrained_model_path,
            'model': model_config,
            'class_names': dataset_info['labels'],
            'train_data': train_data_config,
            'validation_data': valid_data_config
        })

    else:
        task_config = TaggingConfig({
            'init_checkpoint': _params.pretrained_model_path,
            'model': model_config,
            'class_names': dataset_info['labels'],
        })

    return task_config


@exp_factory.register_config_factory('bert/tagging')
def bert_tagging() -> cfg.ExperimentConfig:
    """BERT tagging task."""
    config = cfg.ExperimentConfig(
        task=get_task(),
        trainer=cfg.TrainerConfig(
            optimizer_config=optimization_config.OptimizationConfig({
                'optimizer': {
                    'type': 'adamw',
                    'adamw': {
                        'weight_decay_rate':
                            0.01,
                        'exclude_from_weight_decay':
                            ['LayerNorm', 'layer_norm', 'bias'],
                    }
                },
                'learning_rate': {
                    'type': 'polynomial',
                    'polynomial': {
                        'initial_learning_rate': 8e-5,
                        'end_learning_rate': 0.0,
                    }
                },
                'warmup': {
                    'type': 'polynomial'
                }
            })),
        restrictions=[
            'task.train_data.is_training != None',
            'task.validation_data.is_training != None',
        ])
    return config


def predict(task: TaggingTask,
            params: cfg.DataConfig,
            model: tf.keras.Model) -> List[Tuple[List[int], List[int]]]:
    """Predicts on the input data.

          Args:
            task: A `TaggingTask` object.
            params: A `cfg.DataConfig` object.
            model: A keras.Model.

          Returns:
            A list of tuple. Each tuple contains list of `word_ids` and
              a list of `predict_ids`.
        """

    def predict_step(inputs):
        """Replicated prediction calculation."""
        x, _ = inputs
        _outputs = task.inference_step(x, model)
        predict_ids = _outputs['predict_ids']
        label_mask = tf.greater(x['input_mask'], 0)
        return dict(
            word_ids=x['input_word_ids'],
            predict_ids=predict_ids,
            label_mask=label_mask)

    # def aggregate_fn(state, outputs):
    #     """Concatenates model's outputs."""
    #     if state is None:
    #         state = []
    #
    #     for (batch_predict_ids, batch_label_mask, batch_sentence_ids,
    #          batch_sub_sentence_ids) in zip(outputs['predict_ids'],
    #                                         outputs['label_mask'],
    #                                         outputs['sentence_ids'],
    #                                         outputs['sub_sentence_ids']):
    #         for (tmp_predict_ids, tmp_label_mask, tmp_sentence_id,
    #              tmp_sub_sentence_id) in zip(batch_predict_ids.numpy(),
    #                                          batch_label_mask.numpy(),
    #                                          batch_sentence_ids.numpy(),
    #                                          batch_sub_sentence_ids.numpy()):
    #             real_predict_ids = []
    #             assert len(tmp_predict_ids) == len(tmp_label_mask)
    #             for i in range(len(tmp_predict_ids)):
    #                 # Skip the padding label.
    #                 if tmp_label_mask[i]:
    #                     real_predict_ids.append(tmp_predict_ids[i])
    #             state.append((tmp_sentence_id, tmp_sub_sentence_id, real_predict_ids))
    #
    #     return state

    def aggregate_fn(state, outputs):
        """Concatenates model's outputs."""
        if state is None:
            state = []

        for (batch_word_ids, batch_predict_ids, batch_label_mask) in zip(outputs['word_ids'], outputs['predict_ids'],
                                                                         outputs['label_mask']):
            for (tmp_word_ids, tmp_predict_ids, tmp_label_mask) in zip(batch_word_ids, batch_predict_ids.numpy(),
                                                                       batch_label_mask.numpy()):
                real_predict_ids = []
                real_word_ids = []
                assert len(tmp_predict_ids) == len(tmp_label_mask)
                for i in range(len(tmp_predict_ids)):
                    # Skip the padding label.
                    if tmp_label_mask[i]:
                        real_predict_ids.append(tmp_predict_ids[i])
                        real_word_ids.append(tmp_word_ids[i].numpy())
                assert len(real_word_ids) == len(real_predict_ids), "word_ids and predict_ids are of different length"
                state.append((real_word_ids, real_predict_ids))

        return state

    dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                   task.build_inputs, params)
    outputs = utils.predict(predict_step, aggregate_fn, dataset)

    return outputs

# class BERTTokenClassifier(object):
#
#     def __init__(self, params: BERTTokenClassifierConfig):
#         self._params = params
#         self.train_data_config = None
#         self.valid_data_config = None
#         self.dataset_info = None
#
#     def get_dataset(self):
#         self.train_data_config = cfg.DataConfig({
#             'input_path': os.path.join(self._params.input_path, 'train.tfrecord'),
#             'is_training': True,
#             'global_batch_size': self._params.global_batch_size,
#             'seq_length': self._params.seq_length,
#             'include_sentence_id': self._params.include_sentence_id
#         })
#
#         self.valid_data_config = cfg.DataConfig({
#             'input_path': os.path.join(self._params.input_path, 'train.tfrecord'),
#             'is_training': False,
#             'global_batch_size': self._params.global_batch_size,
#             'seq_length': self._params.seq_length,
#             'include_sentence_id': self._params.include_sentence_id
#         })
#
#         # self.train_dataset = TaggingDataLoader(self.train_data_config).load()
#         # self.valid_dataset = TaggingDataLoader(self.valid_data_config).load()
#
#         with open(os.path.join(self._params.input_path, 'dataset_info.pkl'), 'rb') as f:
#             self.dataset_info = pickle.load(f)
#
#     def get_task(self):
#         model_config = ModelConfig({
#             # 'encoder': encoders.EncoderConfig({'type': 'bert'}),
#         })
#
#         self.get_dataset()
#
#         task_config = TaggingConfig({
#             'init_checkpoint': self._params.pretrained_model_path,
#             'model': model_config,
#             'class_names': self.dataset_info['labels'],
#             'train_data': self.train_data_config,
#             'validation_data': self.valid_data_config
#         })
#
#         # return task_config
#
#         self.task = TaggingTask(task_config)
#
#     def _build_losses(self):
#         def build_losses(labels, model_outputs, aux_losses=None):
#             return self.task.build_losses(labels, {'logits': model_outputs}, aux_losses)
#
#         return build_losses
#
#     def _build_model(self, epochs: int, init_lr: float):
#         self.model = self.task.build_model()
#
#         self.train_steps = self.dataset_info['splits']['train'] // self._params.global_batch_size
#         self.valid_steps = self.dataset_info['splits']['dev'] // self._params.global_batch_size
#         num_train_steps = self.train_steps * epochs
#         num_warmup_steps = num_train_steps // 10
#
#         optimizer = optimization.create_optimizer(
#             init_lr=init_lr,
#             num_train_steps=num_train_steps,
#             num_warmup_steps=num_warmup_steps,
#             optimizer_type='adamw'
#         )
#
#         metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
#
#         self.model.compile(optimizer=optimizer, loss={'logits': self._build_losses()}, metrics=[metrics])
#
#     def _prepare(self):
#         self._init_dataset()
#         self._init_task()
#
#     def train(self, epochs: int, init_lr: float = 2e-5):
#         self._prepare()
#         self._build_model(epochs, init_lr)
#         self.model.fit(
#             x=self.train_dataset,
#             steps_per_epoch=self.train_steps,
#             epochs=epochs,
#             validation_data=self.valid_dataset,
#             validation_steps=self.valid_steps,
#             callbacks=[])
#
#     def plot(self, metric: str):
#         train_metric = self.history.history[metric]
#         valid_metric = self.history.history['val_' + metric]
#
#         epochs = range(len(train_metric))
#
#         plt.plot(epochs, train_metric)
#         plt.plot(epochs, valid_metric)
#         plt.title('Training and validation ' + metric)
#         plt.figure()
#
#     def save(self):
#         self.model.save(self._params.model_export_path, save_traces=True)
#
#     def load(self):
#         return tf.keras.models.load_model(self._params.model_export_path)
#
#     def predict(self):
#         pass
