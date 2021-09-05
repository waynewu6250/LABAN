# LABAN (EMNLP 2021)
A Label-Aware BERT Attention Network for Zero-Shot Multi-Intent Detection in Spoken Language Understanding

This repository contains the code and resources for the following paper.

## To Use
There are three experiments and use cases for LABAN.

### 1. Dependency
Install dependency via command:
>
    pip install -r requirements.txt

### 2. Configurations

First specify the mode in `config.py`:

1. `datatype`: data to use (semantic, mixatis, mixsnips, e2e, sgd) <br>
2. `is_zero_shot`: whether to use zero-shot (True/False) <br>
3. `real_num`: real number of seen intents <br>
4. `ratio`: parameter for splitting train/test labels <br>
5. `is_few_shot`: whether to use few-shot (True/False) <br>
6. `few_shot_ratio`: few shot ratio of data for training <br>
7. `retrain`: use trained model weights <br>
8. `test_mode`: test mode (validation, data, embedding, user)
    >
        validation: produces scores
        data:       produces scores & error analysis
        embedding:  produces sentence embeddings
        user:       predict tag given a sentence

### 3. Parse data
Locate in data/
1. mixatis/mixsnips/semantic:
    >
        (normal):    python train_data.py -d [data_type]
        (zero-shot): python train_data_zero_shot.py -d [data_type] -r [ratio]
2. e2e/sgd: (We do not provide sgd in data.zip since it exceeds upload limit)
    >
        python dialogue_data.py

### 2. Multi-intent detection

Set `is_zero_shot`: False.

To train:
>
    python bert_laban.py train

To test:
(Set `retrain`: True)
>
    python bert_laban.py test

### 3. Zero-shot detection

Set `is_zero_shot`: True. <br>
Specify `real_num` and `ratio`.

To train:
>
    python bert_zsl.py train
To test:
(Set `retrain`: True)
>
    python bert_zsl.py test

### 4. Few-shot detection

Set `is_zero_shot`: True. <br>
Set `is_few_shot`: True. <br>
Specify `few_shot_ratio`.

To train:
>
    python bert_zsl.py train
To test:
(`retrain`: True)
>
    python bert_zsl.py test


# Citation

Please cite if you use the above resources for your research


