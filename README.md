# LABAN (EMNLP 2021)

This repository contains the code and resources for the following paper, published in [EMNLP 2021](https://2021.emnlp.org/).

```
Ting-Wei Wu, Ruolin Su and Biing-Hwang Juang, "A Label-Aware BERT Attention Network for Zero-Shot Multi-Intent Detection in Spoken Language Understanding". In EMNLP 2021 (Main Conference)
```

## To Use
There are three experiments and use cases for LABAN:

1. Normal multi-intent detection
2. Generalized zero-shot multi-intent detection
3. Few-shot multi-intent detection

### 1. Dependency
* Python 3.6
* Pytorch 1.4.0
* CUDA 10.0 supported GPU

    First create a conda environment with python 3.6 and run the following command to install pytorch:
    >
        conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    
    Install dependency via command:
    >
        pip install -r requirements.txt

### 2. Configurations

Specify the mode in `config.py`:

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
    
    1. normal
        >
            python train_data.py -d [data_type]
    2. zero-shot (Creat directory data/<dataset_name>/zeroshot/ first)
        >  
            python train_data_zero_shot.py -d [data_type] -r [ratio]
        
2. e2e/sgd:

    (We do not provide sgd in data.zip since it exceeds upload limit, please
    download sgd dataset [here](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).)
    >
        python dialogue_data.py

### 4. Multi-intent detection

Set `is_zero_shot`: False.

To train:
>
    python bert_laban.py train

To test:
(Set `retrain`: True)
>
    python bert_laban.py test

### 5. Zero-shot detection

Set `is_zero_shot`: True. <br>
Specify `real_num` and `ratio`.

To train:
>
    python bert_zsl.py train
To test:
(Set `retrain`: True)
>
    python bert_zsl.py test

### 6. Few-shot detection

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


### 7. Run baselines




# Citation

Please cite if you use the above resources for your research


