class Config:

    #################### For BERT fine-tuning ####################
    # control
    datatype = "semantic"
    is_zero_shot = True                # For zero-shot training/testing
    real_num = 17
    ratio = '13'
    is_few_shot = True                # For few-shot training/testing
    few_shot_ratio = 0.1               
    retrain = False                     # Reuse trained model weights
    test_mode = "validation" #"user", "data", "embedding", "validation"

    data_mode = "multi" #"single"      # single or multi intent in data
    sentence_mode = "one" #"two"       # one or two sentence in data
    dialog_data_mode = False           # for dialogue-wise data (A+B)
    #################################

    if datatype == "atis":
        # atis dataset
        train_path = "data/atis/raw_data.pkl"
        test_path = "data/atis/raw_data_test.pkl"
        dic_path = "data/atis/intent2id.pkl"
        embedding_path = "finetune_results/atis_embeddings_with_hidden.pth"
    
    elif datatype == "snips":
        # snips dataset
        train_path = "data/snips/raw_data_train.pkl"
        test_path = "data/snips/raw_data_test.pkl"
        dic_path = "data/snips/intent2id.pkl"

    elif datatype == "semantic":
        # semantic parsing dataset
        if not is_zero_shot:
            # normal
            train_path = "data/semantic/raw_data_se.pkl" if data_mode == "single" else "data/semantic/raw_data_multi_se.pkl"
            test_path = "data/semantic/raw_data_multi_se_test.pkl"
            dic_path = "data/semantic/intent2id_se.pkl" if data_mode == "single" else "data/semantic/intent2id_multi_se.pkl"
            dic_path_with_tokens = "data/semantic/intent2id_multi_se_with_tokens.pkl"
            embedding_path = "embeddings/se_embeddings_with_hidden.pth"
        else:
            # zero-shot/few-shot
            train_path = "data/semantic/raw_data_se.pkl" if data_mode == "single" else "data/semantic/zeroshot/raw_data_multi_se_zst_train{}.pkl".format(ratio)
            test_path = "data/semantic/zeroshot/raw_data_multi_se_zst_test{}.pkl".format(ratio)
            dic_path = "data/semantic/intent2id_se.pkl" if data_mode == "single" else "data/semantic/intent2id_multi_se.pkl"
            dic_path_with_tokens = "data/semantic/zeroshot/intent2id_multi_se_with_tokens_zst_train{}.pkl".format(ratio)
            dic_path_with_tokens_test = "data/semantic/zeroshot/intent2id_multi_se_with_tokens_zst_test{}.pkl".format(ratio)
    
    elif datatype == "mixatis":
        # mix atis dataset
        if not is_zero_shot:
            train_path = "data/MixATIS_clean/raw_data_multi_ma_train.pkl"
            dev_path =  "data/MixATIS_clean/raw_data_multi_ma_dev.pkl"
            test_path = "data/MixATIS_clean/raw_data_multi_ma_test.pkl"
            dic_path_with_tokens = "data/MixATIS_clean/intent2id_multi_ma_with_tokens.pkl" 
        else:
            train_path = "data/MixATIS_clean/zeroshot/raw_data_multi_ma_train{}.pkl".format(ratio)
            test_path = "data/MixATIS_clean/zeroshot/raw_data_multi_ma_test{}.pkl".format(ratio)
            dic_path_with_tokens = "data/MixATIS_clean/zeroshot/intent2id_multi_ma_with_tokens_train{}.pkl".format(ratio)
            dic_path_with_tokens_test = "data/MixATIS_clean/zeroshot/intent2id_multi_ma_with_tokens_test{}.pkl".format(ratio)
        embedding_path = "embeddings/mixatis_embeddings_with_hidden.pth"
    
    elif datatype == "mixsnips":
        # mix snips dataset
        if not is_zero_shot:
            train_path = "data/MixSNIPS_clean/raw_data_multi_sn_train.pkl"
            dev_path =  "data/MixSNIPS_clean/raw_data_multi_sn_dev.pkl"
            test_path = "data/MixSNIPS_clean/raw_data_multi_sn_test.pkl"
            dic_path_with_tokens = "data/MixSNIPS_clean/intent2id_multi_sn_with_tokens.pkl"
        else:
            train_path = "data/MixSNIPS_clean/zeroshot/raw_data_multi_sn_train{}.pkl".format(ratio)
            test_path = "data/MixSNIPS_clean/zeroshot/raw_data_multi_sn_test{}.pkl".format(ratio)
            dic_path_with_tokens = "data/MixSNIPS_clean/zeroshot/intent2id_multi_sn_with_tokens_train{}.pkl".format(ratio)
            dic_path_with_tokens_test = "data/MixSNIPS_clean/zeroshot/intent2id_multi_sn_with_tokens_test{}.pkl".format(ratio)
        embedding_path = "embeddings/mixsnips_embeddings_with_hidden.pth"
    
    elif datatype == "e2e":
        # Microsoft e2e dialogue dataset
        train_path = "data/e2e_dialogue/dialogue_data.pkl" if data_mode == "single" else "data/e2e_dialogue/dialogue_data_multi.pkl"
        test_path = "data/e2e_dialogue/dialogue_data_multi.pkl"
        dic_path = "data/e2e_dialogue/intent2id.pkl" if data_mode == "single" else "data/e2e_dialogue/intent2id_multi.pkl"
        dic_path_with_tokens = "data/e2e_dialogue/intent2id_multi_with_tokens.pkl"
        embedding_path = "embeddings/e2e_embeddings_with_hidden.pth"
        pretrain_path = "data/e2e_dialogue/dialogue_data_pretrain.pkl"
    
    elif datatype == "sgd":
        # dstc8-sgd dialogue dataset
        train_path = "data/sgd_dialogue/dialogue_data.pkl" if data_mode == "single" else "data/sgd_dialogue/dialogue_data_multi.pkl"
        test_path = "data/sgd_dialogue/dialogue_data_multi.pkl"
        dic_path = "data/sgd_dialogue/intent2id.pkl" if data_mode == "single" else "data/sgd_dialogue/intent2id_multi.pkl"
        dic_path_with_tokens = "data/sgd_dialogue/intent2id_multi_with_tokens.pkl"
        embedding_path = "embeddings/sgd_embeddings_with_hidden.pth"
        pretrain_path = "data/sgd_dialogue/dialogue_data_pretrain.pkl"
    
    elif datatype == "woz":
        # multiWOZ dataset
        train_path = "data/MULTIWOZ2.1/dialogue_data.pkl"
        test_path = None
        dic_path = "data/MULTIWOZ2.1/intent2id.pkl"
        dialogue_id_path = "data/MULTIWOZ2.1/dialogue_id.pkl"
        embedding_path ="finetune_results/woz_embeddings_sub.pth"
    
    if not is_zero_shot:
        model_path = None if not retrain else "checkpoints/best_{}_{}.pth".format(datatype, data_mode)
    else:
        model_path = None if not retrain else "checkpoints/best_{}_{}_{}.pth".format(datatype, data_mode, ratio)

    # model_path = "checkpoints/best_mixatis_multi.pth"

    maxlen = 50 #20
    batch_size = 128 #128 e2e 16/8 sgd4 32 baseline
    epochs = 50 #30, 5
    learning_rate_bert = 2e-5 # for bert: 2e-5, for baseline: 1e-3
    learning_rate_classifier = 1e-3
    max_dialog_size = 25 if datatype == "e2e" else 50
    dialog_batch_size = 100

    rnn_hidden = 256

opt = Config()