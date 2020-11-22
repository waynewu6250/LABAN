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
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import time

class Data:

    def __init__(self, data_path, rawdata_path, intent2id_path):

        self.data_path = data_path
        self.rawdata_path = rawdata_path
        self.intent2id_path = intent2id_path
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    #==================================================#
    #                   Text Prepare                   #
    #==================================================#
    
    #pure virtual function
    def prepare_text(self):
        raise NotImplementedError("Please define virtual function!!")

    # prepare text
    def text_prepare(self, text, mode):
        """
            text: a string       
            return: modified string
        """
        
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
        return text

############################################################################


class SemanticData(Data):

    def __init__(self, data_path, rawdata_path):

        super(SemanticData, self).__init__(data_path, rawdata_path, None)
        self.raw_data = self.prepare_text()
    
    def prepare_text(self):
        
        data = pd.read_csv(self.data_path, sep='\t', names = ["question", "question2", "info"])
        data["intent"] = data["info"].apply(lambda x: "@".join(set(sorted(re.findall(r'\[IN:(\w+)', x)))))

        ptime = time.time()

        ######################### normal setting #########################
        texts = []
        intents = []
        slots = []
        for i, (text, intent, info) in enumerate(zip(data["question2"].values, data["intent"].values, data["info"])):
            # single intent:
            texts.append(text.split(' '))
            intents.append(intent)
            print("Finish: ", i)
        
        raw_data = {'texts': texts, 'intents': intents}

        with open(self.rawdata_path, "wb") as f:
            pickle.dump(raw_data, f)
        ######################### normal setting #########################
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data


############################################################################


if __name__ == "__main__":
    
    # semantic
    data_path = "../raw_datasets/top-dataset-semantic-parsing/eval.tsv"
    rawdata_path = "semantic/raw_data_se_eval.pkl"
    data = SemanticData(data_path, rawdata_path)

    print(data.raw_data['texts'][100])
    print(data.raw_data['intents'][100])





        




