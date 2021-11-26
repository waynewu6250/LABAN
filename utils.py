"""Utility functions"""
from keras.preprocessing.sequence import pad_sequences
import torch
from config import opt

def load_data(X, maxlen):

    input_ids = pad_sequences(X, maxlen=maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return (input_ids, attention_masks)

def calc_score(outputs, labels):
    corrects = 0
    totals = 0
    preds = 0
    acc = 0
    if opt.data_mode == 'single':
        corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
    else:
        for i, logits in enumerate(outputs):
            log = torch.sigmoid(logits)
            correct = (labels[i][torch.where(log>0.5)[0]]).sum()
            total = len(torch.where(labels[i]==1)[0])
            pred = len(torch.where(log>0.5)[0])
            corrects += correct
            totals += total
            preds += pred
            
            p = (torch.where(log>0.5)[0])
            r = (torch.where(labels[i]==1)[0])
            if len(p) == len(r) and (p == r).all():
                acc += 1
    return corrects, totals, preds, acc

def f1_score_intents(outputs, labels):
    
    P, R, F1, acc = 0, 0, 0, 0
    outputs = torch.sigmoid(outputs)

    for i in range(outputs.shape[0]):
        TP, FP, FN = 0, 0, 0
        for j in range(outputs.shape[1]):
            if outputs[i][j] > 0.5 and labels[i][j] == 1:
                TP += 1
            elif outputs[i][j] <= 0.5 and labels[i][j] == 1:
                FN += 1
            elif outputs[i][j] > 0.5 and labels[i][j] == 0:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        P += precision
        R += recall

        p = (torch.where(outputs[i]>0.5)[0])
        r = (torch.where(labels[i]==1)[0])
        if len(p) == len(r) and (p == r).all():
            acc += 1
        
    P /= outputs.shape[0]
    R /= outputs.shape[0]
    F1 /= outputs.shape[0]
    return P, R, F1, acc