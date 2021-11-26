"""Data script for parsing zero-shot multi-intent dataset"""
import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd
import re
import pickle
import h5py
import json
import os
import csv
import spacy
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AlbertTokenizer
# from gensim.models.keyedvectors import KeyedVectors
# from nltk import word_tokenize
import time

class Data:
    """Main data abstract class"""

    def __init__(self, data_path, rawdata_path, intent2id_path):

        self.data_path = data_path
        self.rawdata_path = rawdata_path
        self.intent2id_path = intent2id_path
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Uncomment the following lines to use other tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
        # self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        # self.w2v = self.load_w2v('../vectors.kv')


    #==================================================#
    #                   Text Prepare                   #
    #==================================================#
    
    #pure virtual function
    def prepare_text(self):
        """transform the text into pickle files"""
        raise NotImplementedError("Please define virtual function!!")

    # prepare text
    def text_prepare(self, text, mode, snips_intent=False):
        """preprocess the text"""

        if snips_intent:
            text = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', text)
            text = ' '.join(text)
        text = text.lower() # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        if mode == "Bert":
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(text)
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            text = tokenized_ids
        elif mode == "fasttext":
            text = text.replace('_', ' ')
            tokenized_text = word_tokenize(text)
            text = []
            for w in tokenized_text:
                if w in self.w2v.key_to_index:
                    text.append(self.w2v.key_to_index[w])
                else:
                    text.append(len(self.w2v.key_to_index))
        return text
    
    def load_w2v(self, file_name):
        """ load w2v model
            input: model file name
            output: w2v model
        """
        # w2v = KeyedVectors.load_word2vec_format(file_name, binary=False, limit=500000)
        w2v = KeyedVectors.load(file_name)
        return w2v


############################################################################


class SemanticData(Data):
    """FSPS dataset"""

    def __init__(self, data_path, rawdata_path, intent2id_path, rawdata_path2=None, intent2id_path2=None, ratio=None, done=True):

        super(SemanticData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.ratio = int(ratio)
        self.rawdata_path2 = rawdata_path2
        self.intent2id_path2 = intent2id_path2
        self.raw_data, self.intent2id = self.prepare_text(done)
    
    def prepare_text(self, done):
        """transform the text into pickle files"""

        if done:
            with open(self.rawdata_path, "rb") as f:
                raw_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return raw_data, intent2id
        
        data = pd.read_csv(self.data_path, sep='\t', names = ["question", "question2", "info"])
        data["intent"] = data["info"].apply(lambda x: "@".join(set(sorted(re.findall(r'\[IN:(\w+)', x)))))

        ptime = time.time()
        
        raw_data = []

        ######################### zero-shot setting #########################
        intent_set = {}
        for i, (text, intents) in enumerate(zip(data["question"].values, data["intent"].values)):
            # single intent:
            # if intent not in intent2id:
            #     intent2id[intent] = counter
            #     counter += 1
            # raw_data.append((self.text_prepare(text, "Bert"), intent2id[intent]))

            # multi intents
            intents = [intent.lower().replace('_', ' ') for intent in intents.split('@')]
            for intent in intents:
                if intent not in intent_set:
                    intent_set[intent] = 0
            raw_data.append((self.text_prepare(text, "Bert"), intents))
            
            print("Finish: ", i)
        
        ############ split data into seen and unseen labels ############
        counter1 = 0
        counter2 = 0
        train_intent2id = {}
        test_intent2id = {}
        # intent_set = sorted(list(intent_set))
        train_set = []
        test_set = []

        # use the ratio to split seen and unseen intents
        for i, intent in enumerate(intent_set):
            if i < self.ratio:
                train_set.append(intent)
            elif i > self.ratio and i % 2 == 0:
                train_set.append(intent)
            else:
                test_set.append(intent)
        test_set = train_set + test_set

        for intent in train_set:
            train_intent2id[intent] = (counter1, self.text_prepare(intent, "Bert"))
            counter1 += 1
        for intent in test_set:
            test_intent2id[intent] = (counter2, self.text_prepare(intent, "Bert"))
            counter2 += 1

        train_data = []
        test_data = []
        for text, intents in raw_data:
            key = True
            for intent in intents:
                # If one of intent is not in train, push it to the test data
                if intent not in train_intent2id:
                    key = False
            if key:
                train_data.append((text, [train_intent2id[intent][0] for intent in intents]))
            else:
                test_data.append((text, [test_intent2id[intent][0] for intent in intents]))
        self.new_train_data = train_data#[:int(0.7*len(train_data))]
        self.new_test_data = test_data# train_data[int(0.7*len(train_data)):] + test_data
        print('Number of the train data: ', len(self.new_train_data))
        print('Number of the test data: ', len(self.new_test_data))
        print('Intent dictionary of the train data: \n', train_intent2id)
        print('Intent dictionary of the test data: \n', test_intent2id)
        print('Number of the train intents:', len(train_intent2id))
        print('Number of the test intents: ', len(test_intent2id))
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(self.new_train_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(train_intent2id, f)

        with open(self.rawdata_path2, "wb") as f:
            pickle.dump(self.new_test_data, f)
        with open(self.intent2id_path2, "wb") as f:
            pickle.dump(test_intent2id, f)
        ######################### zero-shot setting #########################
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data, train_intent2id


############################################################################


class MIXData(Data):
    """Main pipeline for MixATIS, MixSNIPS"""

    def __init__(self, data_path, rawdata_path, intent2id_path, rawdata_path2=None, intent2id_path2=None, ratio=None, done=True):

        super(MIXData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.rawdata_path2 = rawdata_path2
        self.intent2id_path2 = intent2id_path2
        self.ratio = int(ratio)
        self.raw_data, self.intent2id = self.prepare_text(done)
    
    def tokenize(self, tokens, text_labels):
        """Auxiliary function for parsing tokens.
        @param tokens: raw tokens
        @param text_labels: raw_labels
        """
        tokenized_sentence = []
        labels = []

        # Reparse the labels in parallel with the results after Bert tokenization
        for word, label in zip(tokens, text_labels):

            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)

            labels.extend([label] * n_subwords)
        
        tokenized_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+tokenized_sentence+['[SEP]'])

        return tokenized_sentence, tokenized_ids, labels
    
    def prepare_text(self, done):
        """transform the text into pickle files"""

        if done:
            with open(self.rawdata_path, "rb") as f:
                raw_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return raw_data, intent2id
        
        ptime = time.time()

        raw_data = []
        intent2id = {}
        icounter = 0

        with open(self.data_path, 'r') as f:
            text = []
            tag = []
            counter = 0
            for line in f:
                if len(line) > 1:
                    if ' ' not in line:
                        print('data ', counter)

                        intents = line.strip('\n').split('#')
                        for intent in intents:
                            if intent not in intent2id:
                                intent2id[intent] = icounter
                                icounter += 1
                        
                        sent, text, tag = self.tokenize(text, tag)
                        raw_data.append((text, intents))
                        text = []
                        tag = []
                        intents = []
                        counter += 1
                        continue
                    text.append(line.split(' ')[0])
                    tag.append(line.split(' ')[1].strip('\n'))
        
        ############ split data into seen and unseen labels ############
        counter1 = 0
        counter2 = 0
        train_intent2id = {}
        test_intent2id = {}
        # intent_set = sorted(list(intent_set))
        train_set = []
        test_set = []

        # use the ratio to split seen and unseen intents
        for i, (intent,_) in enumerate(intent2id.items()):
            if i < self.ratio:
                train_set.append(intent)
            elif i > self.ratio and i % 2 == 0:
                train_set.append(intent)
            else:
                test_set.append(intent)
        test_set = train_set + test_set

        for intent in train_set:
            train_intent2id[intent] = (counter1, self.text_prepare(intent, "Bert", snips_intent=False))
            counter1 += 1
        for intent in test_set:
            test_intent2id[intent] = (counter2, self.text_prepare(intent, "Bert", snips_intent=False))
            counter2 += 1

        train_data = []
        test_data = []
        for text, intents in raw_data:
            key = True
            for intent in intents:
                # If one of intent is not in train, push it to the test data
                if intent not in train_intent2id:
                    key = False
            if key:
                train_data.append((text, [train_intent2id[intent][0] for intent in intents]))
            else:
                test_data.append((text, [test_intent2id[intent][0] for intent in intents]))
        self.new_train_data = train_data#[:int(0.7*len(train_data))]
        self.new_test_data = test_data# train_data[int(0.7*len(train_data)):] + test_data
        print('Number of the train data: ', len(self.new_train_data))
        print('Number of the test data: ', len(self.new_test_data))
        print('Intent dictionary of the train data: \n', train_intent2id)
        print('Intent dictionary of the test data: \n', test_intent2id)
        print('Number of the train intents:', len(train_intent2id))
        print('Number of the test intents: ', len(test_intent2id))
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(self.new_train_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(train_intent2id, f)

        with open(self.rawdata_path2, "wb") as f:
            pickle.dump(self.new_test_data, f)
        with open(self.intent2id_path2, "wb") as f:
            pickle.dump(test_intent2id, f)
        ######################### zero-shot setting #########################
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data, intent2id


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Put arguments to parse data')

    # For data/mode
    parser.add_argument('-d', '--data', default='atis', dest='mode')
    parser.add_argument('-r', '--ratio', default='18', dest='ratio')
    args = parser.parse_args()

    dirs = ['MixATIS_clean/zeroshot/', 'MixSNIPS_clean/zeroshot/', 'semantic/zeroshot/']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    if args.mode == 'semantic':

        # semantic zero-shot
        data_path = "../raw_datasets/top-dataset-semantic-parsing/train.tsv"
        rawdata_path = "semantic/zeroshot/raw_data_multi_se_zst_train{}.pkl".format(args.ratio)
        rawdata_path2 = "semantic/zeroshot/raw_data_multi_se_zst_test{}.pkl".format(args.ratio)
        intent2id_path = "semantic/zeroshot/intent2id_multi_se_with_tokens_zst_train{}.pkl".format(args.ratio)
        intent2id_path2 = "semantic/zeroshot/intent2id_multi_se_with_tokens_zst_test{}.pkl".format(args.ratio)
        data = SemanticData(data_path, rawdata_path, intent2id_path, rawdata_path2, intent2id_path2, args.ratio, done=False)
    
    elif args.mode == 'mixatis':

        # mixatis
        data_path = "../raw_datasets/MixATIS_clean/train.txt"
        rawdata_path = "MixATIS_clean/zeroshot/raw_data_multi_ma_train{}.pkl".format(args.ratio)
        rawdata_path2 = "MixATIS_clean/zeroshot/raw_data_multi_ma_test{}.pkl".format(args.ratio)
        intent2id_path = "MixATIS_clean/zeroshot/intent2id_multi_ma_with_tokens_train{}.pkl".format(args.ratio)
        intent2id_path2 = "MixATIS_clean/zeroshot/intent2id_multi_ma_with_tokens_test{}.pkl".format(args.ratio)
        data = MIXData(data_path, rawdata_path, intent2id_path, rawdata_path2, intent2id_path2, args.ratio, done=False)
    
    elif args.mode == 'mixsnips':

        # mixsnips
        data_path = "../raw_datasets/MixSNIPS_clean/train.txt"
        rawdata_path = "MixSNIPS_clean/zeroshot/raw_data_multi_sn_train{}.pkl".format(args.ratio)
        rawdata_path2 = "MixSNIPS_clean/zeroshot/raw_data_multi_sn_test{}.pkl".format(args.ratio)
        intent2id_path = "MixSNIPS_clean/zeroshot/intent2id_multi_sn_with_tokens_train{}.pkl".format(args.ratio)
        intent2id_path2 = "MixSNIPS_clean/zeroshot/intent2id_multi_sn_with_tokens_test{}.pkl".format(args.ratio)
        data = MIXData(data_path, rawdata_path, intent2id_path, rawdata_path2, intent2id_path2, args.ratio, done=False)
    
    print('Dataset parsed: ', args.mode)
    print('============ Sample data ============')
    print(data.new_train_data[0])
    print('============ Sample dictionary ============')
    print(data.intent2id)





        




