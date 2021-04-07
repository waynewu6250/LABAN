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

from model import BertZSL
from all_data import get_dataloader
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

#####################################################################

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
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

    X_lengths_train = None
    X_lengths_test = None
    if opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X_train, y_train = zip(*train_data)
        # X_test, y_test = zip(*test_data)
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        all_data = []
        if not opt.dialog_data_mode:
            dialogue_id = {}
            dialogue_counter = 0
            counter = 0
            for data in train_data:
                for instance in data:
                    all_data.append(instance)
                    dialogue_id[counter] = dialogue_counter
                    counter += 1
                dialogue_counter += 1
            X, y, _ = zip(*all_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_lengths = []
            for dialog in train_data:
                X_lengths.extend([len(dialog)]*25)
                if len(dialog) < opt.max_dialog_size:
                    pad_num = opt.max_dialog_size-len(dialog)
                    # Pad dummy sentences
                    pad = ([101,0,0,102], [0], [0])
                    dialog.extend([pad]*pad_num)
                all_data.append(dialog)
            all_data = [sent for dialog in all_data for sent in dialog]
            
            X, y, _ = zip(*all_data)
            train_num = int(len(train_data)*0.7)*25
            X_train = X[:train_num]
            y_train = y[:train_num]
            X_test = X[train_num:]
            y_test = y[train_num:]
            X_lengths_train = X_lengths[:train_num]
            X_lengths_test = X_lengths[train_num:]
    elif 'mix' in opt.datatype:
        # Mix dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
    
    X_train, mask_train = load_data(X_train, opt.maxlen)
    X_test, mask_test = load_data(X_test, opt.maxlen)
    
    # length = int(len(X_train)*0.1)
    # X_train = X_train[:length]
    # y_train = y_train[:length]
    # mask_train = mask_train[:length]
    
    train_loader = get_dataloader(X_train, y_train, mask_train, len(dic), opt, X_lengths=X_lengths_train)
    val_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt, X_lengths=X_lengths_test)

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
    
    if not opt.dialog_data_mode:
        model = BertZSL(config, len(dic))
    else:
        model = BertDST(config, opt, len(dic))
    
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    else:
        print("Train from scratch...")
    model = model.to(device)

    # optimizer, criterion
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    # ]
    # optimizer = BertAdam(optimizer_grouped_parameters,lr=opt.learning_rate_bert, warmup=.1)

    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=opt.learning_rate_bert)
    if opt.data_mode == 'single':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
        # criterion = nn.MSELoss().to(device)
    best_loss = 100
    best_accuracy = 0

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch+1, opt.epochs))

        # Training Phase
        total_train_loss = 0
        train_corrects = 0
        totals = 0
        preds = 0
        total_acc = 0
        model.train()
        for (captions_t, labels, masks) in tqdm(train_loader):

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            _, _, outputs, _ = model(captions_t, masks, intent_tokens, mask_tokens, labels)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            co, to, pr, acc = calc_score(outputs, labels)
            train_corrects += co
            totals += to
            preds += pr
            total_acc += acc

        print('Average train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        if opt.data_mode == 'single':
            train_acc = train_corrects.double() / train_loader.dataset.num_data
            print('Train accuracy: {:.4f}'.format(train_acc))
        elif opt.data_mode == 'multi':
            recall = train_corrects.double() / totals
            precision = train_corrects.double() / preds
            f1 = 2 * (precision*recall) / (precision + recall)
            print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
            print('Accuracy: ', total_acc/train_loader.dataset.num_data)
        

        # Validation Phase
        total_val_loss = 0
        val_corrects = 0
        totals = 0
        preds = 0
        total_acc = 0
        model.eval()
        for (captions_t, labels, masks) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs, _ = model(captions_t, masks, intent_tokens, mask_tokens, labels)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            co, to, pr, acc = calc_score(outputs, labels)
            val_corrects += co
            totals += to
            preds += pr
            total_acc += acc

        print('Average val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))
        if opt.data_mode == 'single':
            val_acc = val_corrects.double() / val_loader.dataset.num_data
            print('Val accuracy: {:.4f}'.format(val_acc))
        elif opt.data_mode == 'multi':
            recall = val_corrects.double() / totals
            precision = val_corrects.double() / preds
            f1 = 2 * (precision*recall) / (precision + recall)
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    print('Dataset to use: ', opt.test_path)
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

    if opt.datatype == "atis":
        # ATIS Dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
    elif opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X, y = zip(*test_data)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, y_test = zip(*test_data)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        all_data = []
        dialogue_id = {}
        dialogue_counter = 0
        counter = 0
        for data in train_data:
            for instance in data:
                all_data.append(instance)
                dialogue_id[counter] = dialogue_counter
                counter += 1
            dialogue_counter += 1
        X, y, _ = zip(*all_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif 'mix' in opt.datatype:
        # Mix dataset
        X_test, y_test, _ = zip(*test_data)

    #X_train, mask_train = load_data(X_train, opt.maxlen)
    X_test, mask_test = load_data(X_test, opt.maxlen)

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
    
    if not opt.dialog_data_mode:
        # with open('data/MixATIS_clean/intent2id_multi_ma_with_tokens.pkl', 'rb') as f:
        #     dicc = pickle.load(f)
        model = BertZSL(config, len(dic))
    else:
        model = BertDST(config, opt, len(dic))

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

        test_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt)
        
        val_corrects = 0
        totals = 0
        preds = 0
        total_acc = 0
        model.eval()

        f_output = []
        all_labels = []
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            print('Run prediction: ', i)

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs, clusters = model(captions_t, masks, intent_tokens, mask_tokens, labels)

            co, to, pr, acc = calc_score(outputs, labels)
            val_corrects += co
            totals += to
            preds += pr
            total_acc += acc

            f_output.append(outputs)
            all_labels.append(labels)

        f_output = torch.cat(f_output,dim=0)
        all_labels = torch.cat(all_labels,dim=0)
        dic = {'outputs': f_output, 'clusters': clusters, 'labels': all_labels}

        recall = val_corrects.double() / totals
        precision = val_corrects.double() / preds
        f1 = 2 * (precision*recall) / (precision + recall)
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)

        torch.save(dic, 'result_snips.pth')
    
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
        test_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt)
        
        error_ids = []
        pred_labels = []
        real_labels = []
        test_corrects = 0
        totals = 0
        preds = 0
        total_acc = 0
        model.eval()
        print(len(test_loader.dataset))
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            print('predict batches: ', i)

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)
                co, to, pr, acc = calc_score(outputs, labels)
                test_corrects += co
                totals += to
                preds += pr
                total_acc += acc

                if opt.data_mode == 'single':
                    idx = torch.max(outputs, 1)[1] != labels
                    wrong_ids = [tokenizer.convert_ids_to_tokens(caption, skip_special_tokens=True) for caption in captions_t[idx]]
                    error_ids += wrong_ids
                    pred_labels += [reverse_dic[label.item()] for label in torch.max(outputs, 1)[1][idx]]
                    real_labels += [reverse_dic[label.item()] for label in labels[idx]]
                else:
                    for i, logits in enumerate(outputs):
                        log = torch.sigmoid(logits)
                        correct = (labels[i][torch.where(log>0.5)[0]]).sum()
                        total = len(torch.where(labels[i]==1)[0])
                        if correct != total:
                            wrong_caption = tokenizer.convert_ids_to_tokens(captions_t[i], skip_special_tokens=True)
                            error_ids.append(wrong_caption)
                            pred_ls = [p for p in torch.where(log>0.5)[0].detach().cpu().numpy()]
                            real_ls = [i for i, r in enumerate(labels[i].detach().cpu().numpy()) if r == 1]
                            pred_labels.append(pred_ls)
                            real_labels.append(real_ls)
            

        with open('error_analysis/{}_{}.txt'.format(opt.datatype, opt.data_mode), 'w') as f:
            f.write('----------- Wrong Examples ------------\n')
            for i, (caption, pred, real) in enumerate(zip(error_ids, pred_labels, real_labels)):
                f.write(str(i)+'\n')
                f.write(' '.join(caption)+'\n')
                p_r = [reverse_dic[p] for p in pred]
                r_r = [reverse_dic[r] for r in real]
                f.write('Predicted label: {}\n'.format(p_r))
                f.write('Real label: {}\n'.format(r_r))
                f.write('------\n')
        recall = test_corrects.double() / totals
        precision = test_corrects.double() / preds
        f1 = 2 * (precision*recall) / (precision + recall)
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)

        results = {'pred': pred_labels, 'real': real_labels}
        with open('error_analysis/predictions/{}_{}.pkl'.format(opt.datatype, opt.data_mode), 'wb') as f:
            pickle.dump(results, f)

    
    # User defined
    elif opt.test_mode == "user":
        while True:
            print("Please input the sentence: ")
            text = input()
            print("\n======== Predicted Results ========")
            print(text)
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(text)
            tokenized_ids = np.array(tokenizer.convert_tokens_to_ids(tokenized_text))[np.newaxis,:]
            
            input_ids = pad_sequences(tokenized_ids, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post").squeeze(0)
            attention_masks = [float(i>0) for i in input_ids]

            captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
            with torch.no_grad():
                pooled_output, outputs = model(captions_t, mask)
            print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
            print("=================================")    
    
    





if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    