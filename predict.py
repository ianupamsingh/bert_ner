import os
import pickle

import tensorflow as tf
import official.core.config_definitions as cfg
import tagging
from preprocessor import Preprocessor, PreprocessorConfig
# from tagging_data_loader import TaggingDataLoader

HOME = os.environ.get('MODEL_EXPERIMENT_HOME', None)


def main():
    task = tagging.TaggingTask(tagging.get_task('predict'))
    model: tf.keras.Model = task.build_model()
    latest = tf.train.latest_checkpoint(os.path.join(HOME, "Models/BERT-NER"))

    preprocessor = Preprocessor(PreprocessorConfig())

    # model = model.load_weights(latest).expect_partial() # is not working
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(latest).expect_partial()

    _params = tagging.BertNerConfig()
    test_data_config = cfg.DataConfig({
        'input_path': os.path.join(_params.input_path, 'test.tfrecord'),
        'is_training': False,
        'global_batch_size': _params.global_batch_size,
        'seq_length': _params.seq_length,
        'include_sentence_id': _params.include_sentence_id
    })

    # dataset = TaggingDataLoader(test_data_config).load()
    # for each in dataset:
    #     print(each)
    #     break

    # TODO: read dataset using data config and extract max_context information

    outputs = tagging.predict(task, test_data_config, checkpoint.root)

    # TODO: add logic for writer from BERTv1 (compressing strides)

    labels = [[preprocessor.id2label[label_id] for label_id in output[1]] for output in outputs]
    with open('predict.log', 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    main()

