"""For model training and inference (zero-shot learning)
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
from utils import *

def train(**kwargs):
    """Main zero-shot training pipeline"""
    
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
    ## get dictionary
    dic_path = opt.dic_path_with_tokens_test if opt.is_few_shot else opt.dic_path_with_tokens
    with open(dic_path, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    if opt.test_path:
        with open(opt.test_path, 'rb') as f:
            test_data = pickle.load(f)
    print('Number of labels: ', len(dic))
    
    if opt.is_few_shot:
        train_data = train_data+test_data[:int(len(test_data)*opt.few_shot_ratio)]
        print(int(len(test_data)*opt.few_shot_ratio))
    
    ## get data
    X_lengths_train = None
    X_lengths_test = None
    if opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X_train, y_train = zip(*train_data)
        # X_test, y_test = zip(*test_data)
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif 'mix' in opt.datatype:
        # Mix dataset
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train, mask_train = load_data(X_train, opt.maxlen)
    X_test, mask_test = load_data(X_test, opt.maxlen)
    
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
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.train()
        ccounter = 0
        for (captions_t, labels, masks) in tqdm(train_loader):

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            _, _, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)
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
        for (captions_t, labels, masks) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)
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

            torch.save(model.state_dict(), 'checkpoints/best_{}_{}_{}.pth'.format(opt.datatype, opt.data_mode, opt.ratio))
        
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
        train_dic = pickle.load(f)
    with open(opt.dic_path_with_tokens_test, 'rb') as f:
        dic = pickle.load(f)
    print('Train dictionary: \n', train_dic)
    print('Test dictionary: \n', dic)
    print('Number of training labels: ', len(train_dic))
    print('Number of testing labels: ', len(dic))
    reverse_dic = {v[0]: k for k,v in dic.items()}
    with open(opt.test_path, 'rb') as f:
        test_data = pickle.load(f)
        if opt.is_few_shot:
            test_data = test_data[int(len(test_data)*opt.few_shot_ratio):]

    if opt.datatype == "semantic":
        # Semantic parsing Dataset
        # X, y = zip(*test_data)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, y_test = zip(*test_data)
    elif 'mix' in opt.datatype:
        # Mix dataset
        X_test, y_test = zip(*test_data)

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

    use_dic = dic if opt.is_few_shot else train_dic
    if not opt.dialog_data_mode:
        model = BertZSL(config, len(use_dic))
    else:
        model = BertDST(config, opt, len(use_dic))

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
    
    # Run zero-shot validation
    elif opt.test_mode == "validation":

        test_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt)
        
        val_corrects = 0
        total_P, total_R, total_F1, total_acc = 0, 0, 0, 0
        total_P_seen, total_R_seen, total_F1_seen, total_acc_seen = 0, 0, 0, 0
        total_P_unseen, total_R_unseen, total_F1_unseen, total_acc_unseen = 0, 0, 0, 0
        ccounter = 0
        model.eval()
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            print('Run prediction: ', i)

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)

            # total
            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            # seen
            P, R, F1, acc = f1_score_intents(outputs[:,:opt.real_num], labels[:,:opt.real_num])
            total_P_seen += P
            total_R_seen += R
            total_F1_seen += F1
            total_acc_seen += acc
            # unseen
            P, R, F1, acc = f1_score_intents(outputs[:,opt.real_num:], labels[:,opt.real_num:])
            total_P_unseen += P
            total_R_unseen += R
            total_F1_unseen += F1
            total_acc_unseen += acc

            ccounter += 1

        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)

        precision = total_P_seen / ccounter
        recall = total_R_seen / ccounter
        f1 = total_F1_seen / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc_seen/test_loader.dataset.num_data)

        precision = total_P_unseen / ccounter
        recall = total_R_unseen / ccounter
        f1 = total_F1_unseen / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc_unseen/test_loader.dataset.num_data)
    
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
        total_P, total_R, total_F1, total_acc = 0, 0, 0, 0
        total_P_seen, total_R_seen, total_F1_seen, total_acc_seen = 0, 0, 0, 0
        total_P_unseen, total_R_unseen, total_F1_unseen, total_acc_unseen = 0, 0, 0, 0
        ccounter = 0
        model.eval()
        print(len(test_loader.dataset))
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            print('predict batches: ', i)

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks, intent_tokens, mask_tokens, labels)

                # total
                P, R, F1, acc = f1_score_intents(outputs, labels)
                total_P += P
                total_R += R
                total_F1 += F1
                total_acc += acc
                # seen
                P, R, F1, acc = f1_score_intents(outputs[:,:opt.real_num], labels[:,:opt.real_num])
                total_P_seen += P
                total_R_seen += R
                total_F1_seen += F1
                total_acc_seen += acc
                # unseen
                P, R, F1, acc = f1_score_intents(outputs[:,opt.real_num:], labels[:,opt.real_num:])
                total_P_unseen += P
                total_R_unseen += R
                total_F1_unseen += F1
                total_acc_unseen += acc
                
                ccounter += 1

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
                        # if correct != total:
                        wrong_caption = tokenizer.convert_ids_to_tokens(captions_t[i], skip_special_tokens=True)
                        error_ids.append(wrong_caption)
                        pred_ls = [p for p in torch.where(log>0.5)[0].detach().cpu().numpy()]
                        real_ls = [i for i, r in enumerate(labels[i].detach().cpu().numpy()) if r == 1]
                        pred_labels.append(pred_ls)
                        real_labels.append(real_ls)

        with open('error_analysis/{}_{}_zsl.txt'.format(opt.datatype, opt.data_mode), 'w') as f:
            f.write('----------- Wrong Examples ------------\n')
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

        precision = total_P_seen / ccounter
        recall = total_R_seen / ccounter
        f1 = total_F1_seen / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc_seen/test_loader.dataset.num_data)

        precision = total_P_unseen / ccounter
        recall = total_R_unseen / ccounter
        f1 = total_F1_unseen / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc_unseen/test_loader.dataset.num_data)

        results = {'pred': pred_labels, 'real': real_labels}
        with open('error_analysis/predictions/{}_{}_{}.pkl'.format(opt.datatype, opt.data_mode, opt.ratio), 'wb') as f:
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
    


            








        








    


    