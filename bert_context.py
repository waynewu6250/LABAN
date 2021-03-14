"""For model training and inference (multi-intent detection)
Data input should be a single sentence.
"""
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer, BertModel, BertConfig, AdamW

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import copy
import numpy as np
import collections
from tqdm import tqdm

from model import BertContextNLU
from all_data_context import get_dataloader_context
from config import opt

def load_data(X, maxlen):

    input_ids = pad_sequences(X, maxlen=maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return (input_ids, attention_masks)

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

#####################################################################

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    np.random.seed(0)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    print('Dataset to use: ', opt.train_path)
    print('Dictionary to use: ', opt.dic_path_with_tokens)
    print('Data Mode: ', opt.data_mode)
    print('Sentence Mode: ', opt.sentence_mode)

    # dataset
    with open(opt.dic_path_with_tokens, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    if opt.test_path:
        with open(opt.test_path, 'rb') as f:
            test_data = pickle.load(f)

    if opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X_train, y_train = zip(*train_data)
        # X_test, y_test = zip(*test_data)
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, mask_train = load_data(X_train, opt.maxlen)
        X_test, mask_test = load_data(X_test, opt.maxlen)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        indices = np.random.permutation(len(train_data))
        train = np.array(train_data)[indices[:int(len(train_data)*0.7)]]
        test = np.array(train_data)[indices[int(len(train_data)*0.7):]]
    elif 'mix' in opt.datatype:
        # Mix dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
        X_train, mask_train = load_data(X_train, opt.maxlen)
        X_test, mask_test = load_data(X_test, opt.maxlen)
    
    train_loader = get_dataloader_context(train, dic, opt)
    val_loader = get_dataloader_context(test, dic, opt)

    # label tokens
    intent_tokens = [intent for name, (tag, intent) in dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)
    intent_tokens = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens[i] = torch.tensor(mask_tok[i])
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertContextNLU(config, len(dic))
    
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    else:
        print("Train from scratch...")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=opt.learning_rate_bert)
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    best_loss = 100
    best_accuracy = 0

    #################################### Start training ####################################
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch+1, opt.epochs))

        # Training Phase
        total_train_loss = 0
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.train()
        ccounter = 0
        for (result_ids, result_token_masks, result_masks, lengths, result_labels) in tqdm(train_loader):

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_labels = result_labels.to(device)

            optimizer.zero_grad()

            outputs, labels = model(result_ids, result_token_masks, result_masks, lengths, result_labels)
            train_loss = criterion(outputs, labels)
            
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

        print('Average train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/train_loader.dataset.num_data)
        

        # Validation Phase
        total_val_loss = 0
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.eval()
        ccounter = 0
        for (result_ids, result_token_masks, result_masks, lengths, result_labels) in val_loader:

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_labels = result_labels.to(device)
            
            with torch.no_grad():
                outputs, labels = model(result_ids, result_token_masks, result_masks, lengths, result_labels)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

        print('Average val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))

        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/val_loader.dataset.num_data)
        val_acc = total_acc/val_loader.dataset.num_data
        
        if val_acc > best_accuracy:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_accuracy = val_acc

            torch.save(model.state_dict(), 'checkpoints/best_{}_{}.pth'.format(opt.datatype, opt.data_mode))
        
        print()
    print('Best total val loss: {:.4f}'.format(total_val_loss))
    print('Best Test Accuracy: {:.4f}'.format(best_accuracy))


#####################################################################


def test(**kwargs):

    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    np.random.seed(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    print('Dataset to use: ', opt.train_path)
    print('Dictionary to use: ', opt.dic_path_with_tokens)

    # dataset
    with open(opt.dic_path_with_tokens, 'rb') as f:
        dic = pickle.load(f)
    print(dic)
    reverse_dic = {v[0]: k for k,v in dic.items()}
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(opt.test_path, 'rb') as f:
        test_data = pickle.load(f)

    if opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X_train, y_train = zip(*train_data)
        # X_test, y_test = zip(*test_data)
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, mask_train = load_data(X_train, opt.maxlen)
        X_test, mask_test = load_data(X_test, opt.maxlen)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        indices = np.random.permutation(len(train_data))
        train = np.array(train_data)[indices[:int(len(train_data)*0.7)]]
        test = np.array(train_data)[indices[int(len(train_data)*0.7):]][:1000]
    elif 'mix' in opt.datatype:
        # Mix dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
        X_train, mask_train = load_data(X_train, opt.maxlen)
        X_test, mask_test = load_data(X_test, opt.maxlen)

    train_loader = get_dataloader_context(train, dic, opt)
    test_loader = get_dataloader_context(test, dic, opt)

    # label tokens
    intent_tokens = [intent for name, (tag, intent) in dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)
    intent_tokens = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens[i] = torch.tensor(mask_tok[i])
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertContextNLU(config, len(dic))

    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model {} has been loaded.".format(opt.model_path))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Store embeddings
    if opt.test_mode == "embedding":
        
        test_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt)

        results = []
        model.eval()
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            
            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                hidden_states, pooled_output, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)
                print("Saving Data: %d" % i)

                for ii in range(len(labels)):
                    
                    embedding = pooled_output[ii].data.cpu().numpy().reshape(-1) # (h,)
                    word_embeddings = hidden_states[ii].data.cpu().numpy() # (t,h)
                    
                    tokens = tokenizer.convert_ids_to_tokens(captions_t[ii].data.cpu().numpy())
                    tokens = [token for token in tokens if token != "[CLS]" and token != "[SEP]" and token != "[PAD]"]
                    original_sentence = " ".join(tokens)

                    results.append((original_sentence, embedding, word_embeddings, labels[ii]))

        torch.save(results, opt.embedding_path)
    
    # Run multi-intent validation
    elif opt.test_mode == "validation":
        
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.eval()
        ccounter = 0
        for (result_ids, result_token_masks, result_masks, lengths, result_labels) in test_loader:

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_labels = result_labels.to(device)
            
            with torch.no_grad():
                outputs, labels = model(result_ids, result_token_masks, result_masks, lengths, result_labels)

            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)
    
    # Run test classification
    elif opt.test_mode == "data":
        
        # Single instance
        # index = np.random.randint(0, len(X_test), 1)[0]
        # input_ids = X_test[index]
        # attention_masks = mask_test[index]
        # print(" ".join(tokenizer.convert_ids_to_tokens(input_ids)))

        # captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        # mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     pooled_output, outputs = model(captions_t, mask)
        # print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
        # print("Real label: ", reverse_dic[y_test[index]])

        # Validation Phase
        pred_labels = []
        real_labels = []
        error_ids = []
        total_P, total_R, total_F1, total_acc = 0, 0, 0, 0
        ccounter = 0
        model.eval()
        print(len(test_loader.dataset))
        for num, (result_ids, result_token_masks, result_masks, lengths, result_labels) in enumerate(test_loader):
            print('predict batches: ', num)

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_labels = result_labels.to(device)

            # Remove padding
            texts_no_pad = []
            for i in range(len(result_ids)):
                texts_no_pad.append(result_ids[i,:lengths[i],:])
            texts_no_pad = torch.vstack(texts_no_pad)
            
            with torch.no_grad():
                outputs, labels = model(result_ids, result_token_masks, result_masks, lengths, result_labels)

                # total
                P, R, F1, acc = f1_score_intents(outputs, labels)
                total_P += P
                total_R += R
                total_F1 += F1
                total_acc += acc
                
                ccounter += 1

                for i, logits in enumerate(outputs):
                    log = torch.sigmoid(logits)
                    correct = (labels[i][torch.where(log>0.5)[0]]).sum()
                    total = len(torch.where(labels[i]==1)[0])
                    wrong_caption = tokenizer.convert_ids_to_tokens(texts_no_pad[i], skip_special_tokens=True)
                    error_ids.append(wrong_caption)
                    pred_ls = [p for p in torch.where(log>0.5)[0].detach().cpu().numpy()]
                    real_ls = [i for i, r in enumerate(labels[i].detach().cpu().numpy()) if r == 1]
                    pred_labels.append(pred_ls)
                    real_labels.append(real_ls)

        with open('error_analysis/{}_{}_context.txt'.format(opt.datatype, opt.data_mode), 'w') as f:
            f.write('----------- Examples ------------\n')
            for i, (caption, pred, real) in enumerate(zip(error_ids, pred_labels, real_labels)):
                f.write(str(i)+'\n')
                f.write(' '.join(caption)+'\n')
                p_r = [reverse_dic[p] for p in pred]
                r_r = [reverse_dic[r] for r in real]
                f.write('Predicted label: {}\n'.format(p_r))
                f.write('Real label: {}\n'.format(r_r))
                f.write('------\n')
        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)



if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    