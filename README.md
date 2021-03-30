# LABAN
A Label-Aware BERT Attention Network for Zero-Shot Multi-Intent Detection in Spoken Language Understanding

## To Use
There are three experiments and use cases for LABAN.
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

### 1. Parse data

1. mixatis/mixsnips/semantic:
    >
        (normal):    python train_data.py -d [data_type]
        (zero-shot): python train_data_zero_shot.py -d [data_type] -r [ratio]
2. e2e/sgd:
    >
        python dialogue_data.py

### 2. Multi-intent detection

To train:
>
    python bert_laban.py train

To test:
(`retrain`: True)
>
    python bert_laban.py test

### 3. Zero-shot detection

Set `is_zero_shot`: True.
Specify `real_num` and `ratio`.

To train:
>
    python bert_zsl.py train
To test:
(`retrain`: True)
>
    python bert_zsl.py test

### 4. Few-shot detection

Set `is_zero_shot`: True.
Set `is_few_shot`: True.
Specify `few_shot_ratio`.

To train:
>
    python bert_zsl.py train
To test:
(`retrain`: True)
>
    python bert_zsl.py test
