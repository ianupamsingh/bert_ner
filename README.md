# BERT NER v2

Tensorflow 2.6+ implementation of BERT NER using [tf-models-official](https://github.com/tensorflow/models/tree/master/official)

This implementation requires following pre-trained models to be downloaded and saved at `$MODEL_HOME/Models/<model-name>` :
- BERT-Pretrained - [bert_en_uncased_preprocess](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)
- BERT-PubMed - [experts/bert/pubmed](https://tfhub.dev/google/experts/bert/pubmed/2)

## Pre-requisite

- Create a new environment with python 3.8+
- Activate python environment
- Install requirements `pip install requirement.txt`
- Change `$MODEL_HOME` environment variable to path of ModelExperiments folder

## Training

`python train.py --experiment=bert/tagging --config_file=experiments/parameters.yaml --mode=train --model_dir=tmp/`

## Predicting

`python predict.py`