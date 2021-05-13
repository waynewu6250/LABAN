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
from train_data import Data
import time 

class E2EData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, done=True):

        super(E2EData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.train_data, self.intent2id = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)
    
    def prepare(self, data_path, intent2id, counter):

        print('Parsing file: ', data_path)

        all_data = []
        data = []
        prev_id = '1'

        with open(self.data_path+data_path, 'r') as f:
        
            for i, line in enumerate(f):
                if i == 0:
                    continue
                
                infos = line.split('\t')
                dialogue_id = infos[0]
                message_id = infos[1]
                speaker = infos[3]
                text = infos[4]
                intents = []
                slots = []
                for act in infos[5:]:
                    if act[:act.find('(')] != '':
                        intents.append(act[:act.find('(')])
                    s = re.findall('\((.*)\)', act)
                    if s:
                        slots.append(s[0].split(';'))
                
                # single intent
                # intents = "@".join(sorted(intents))
                # if intents not in intent2id:
                #     intent2id[intents] = counter
                #     counter += 1
                # intents = intent2id[intents]
                
                # multi intents
                for intent in intents:
                    if intent not in intent2id:
                        intent2id[intent] = (counter, self.text_prepare(intent, 'Bert')) # counter
                        counter += 1
                intents = [intent2id[intent][0] for intent in intents]
                
                if data and prev_id != dialogue_id:
                    all_data.append(data)
                    data = []
                    prev_id = dialogue_id
                
                data.append((self.text_prepare(text, 'Bert'), intents, slots))
                # data.append((text, intents, slots))
        
        return all_data, counter
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues
        for each dialogue:
            [(sent1, [label1, label2], [slot1, slot2]), 
             (sent2, [label2], [slot2]),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return train_data, intent2id
        
        ptime = time.time()

        # if os.path.exists(self.intent2id_path):
        #     with open(self.intent2id_path, "rb") as f:
        #         intent2id = pickle.load(f)
        #     counter = len(intent2id)
        # else:
        intent2id = {}
        counter = 0
        
        all_data = []
        for data_path in os.listdir(self.data_path):
            data, counter = self.prepare(data_path, intent2id, counter)
            all_data += data
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data, intent2id


############################################################################


class SGDData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, turn_path, done=True):

        super(SGDData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.turn_path = turn_path
        self.train_data, self.intent2id, self.turn_data_all = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)
    
    def build_ids(self, items, item2id, counter):
        for item in items:
            if item not in item2id:
                item2id[item] = (counter, self.text_prepare(item, 'Bert')) # counter
                counter += 1
        items = [item2id[item][0] for item in items]
        return items, item2id, counter
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues (utterance-level)
        for each dialogue:
            [(sent1, [label1, label2], [slot1, slot2]), 
             (sent2, [label2], [slot2]),...]
        
        a list of dialogues (turn-level)
        for each dialogue:
            [(turn1, intents1, requested_slots1, slots1, values1),...
             (turn2, intents2, requested_slots2, slots2, values2),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            with open(self.turn_path, "rb") as f:
                turn_data_all = pickle.load(f)
            return train_data, intent2id, turn_data_all
        
        ptime = time.time()

        if os.path.exists(self.intent2id_path):
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            counter = len(intent2id)
        else:
            intent2id = {}
            counter = 0
        
        aintent2id = {}
        acounter = 0
        request2id = {}
        rcounter = 0

        all_data = []
        all_data_turn = []
        services = []

        for file in sorted(os.listdir(self.data_path))[:-1]:
            
            with open(os.path.join(self.data_path, file), 'r') as f:
                print('Parsing file: ', file)
                raw_data = json.load(f)
                for dialogue in raw_data:

                    # utterance data
                    data = []

                    # turn data
                    prev_text = 'this is a dummy sentence'
                    prev_data = ('', '', '')
                    data_turn = []

                    for turns in dialogue['turns']:

                        ###################### utterance ##########################
                        intents = []
                        slots = []
                        for action in turns['frames'][0]['actions']:
                            intents.append(action['act'])
                            slots.append((action['slot'], action['values']))
                        
                        intents = list(set(intents))

                        # single intent
                        # intents = "@".join(intents)
                        # if intents not in intent2id:
                        #     intent2id[intents] = counter
                        #     counter += 1
                        # intents = intent2id[intents]
                        
                        # multi intents
                        for intent in intents:
                            if intent not in intent2id:
                                intent2id[intent] = (counter, self.text_prepare(intent, 'Bert')) # counter
                                counter += 1
                        intents = [intent2id[intent][0] for intent in intents]

                        data.append((self.text_prepare(turns['utterance'], 'Bert'), intents, slots))
                        # data.append((turns['utterance'], intents, slots))

                        ###################### turn ##########################
                        if 'state' in turns['frames'][0]:
                            slot_values = turns['frames'][0]['state']['slot_values']
                            if not slot_values:
                                s_turn = []
                                v_turn = []
                            else:
                                s_turn, v_turn = zip(*[(k,v[0]) for k, v in slot_values.items()])
                            
                            encoded = self.tokenizer.encode_plus(prev_text, text_pair=turns['utterance'], return_tensors='pt')
                            aintents, aintent2id, acounter = self.build_ids([turns['frames'][0]['state']['active_intent']], aintent2id, acounter)
                            requests, request2id, rcounter = self.build_ids(turns['frames'][0]['state']['requested_slots'], request2id, rcounter)

                            data_turn.append((encoded['input_ids'], aintents, requests, s_turn, v_turn, (prev_data, data[-1])))
                            prev_text = turns['utterance']
                        else:
                            prev_text = turns['utterance']
                            prev_data = data[-1]

                    
                    all_data.append(data)
                    all_data_turn.append(data_turn)
                    services.append(dialogue['services'])
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        with open("sgd_dialogue/services.pkl", "wb") as f:
            pickle.dump(services, f)
        turn_data_all = {'turns': all_data_turn,
                         'aintent2id': aintent2id,
                         'request2id': request2id}
        with open(self.turn_path, "wb") as f:
            pickle.dump(turn_data_all, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data, intent2id, turn_data_all
    
    
if __name__ == "__main__":
    # e2e dataset
    data_path = "../raw_datasets/e2e_dialogue/"
    rawdata_path = "e2e_dialogue/dialogue_data_multi.pkl"
    intent2id_path = "e2e_dialogue/intent2id_multi_with_tokens.pkl"
    data = E2EData(data_path, rawdata_path, intent2id_path, done=False)

    # sgd dataset
    # data_path = "../raw_datasets/dstc8-schema-guided-dialogue/test"
    # rawdata_path = "sgd_dialogue/dialogue_data_multi_test.pkl"
    # intent2id_path = "sgd_dialogue/intent2id_multi_with_tokens.pkl"
    # turn_path = "sgd_dialogue/turns.pkl"
    # data = SGDData(data_path, rawdata_path, intent2id_path, turn_path, done=False)
    # print(data.turn_data_all['turns'][0])
    
    print(data.train_data[100])
    print(data.intent2id)