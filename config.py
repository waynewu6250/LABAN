class Config:

    #################### For BERT fine-tuning ####################
    # control
    datatype = "e2e"
    is_zero_shot = False                # For zero-shot training/testing
    real_num = 5
    ratio = '4'
    is_few_shot = False                # For few-shot training/testing
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
    batch_size = 128 #16
    epochs = 30 #30, 5
    learning_rate_bert = 2e-5 #1e-3
    learning_rate_classifier = 1e-3
    max_dialog_size = 25 if datatype == "e2e" else 50
    dialog_batch_size = 100

    #################### For Clustering & DCEC ####################
    
    dic_path_for_cluster = "./data/semantic/intent2id_se.pkl"
    embedding_path_for_cluster = "./finetune_results/se_embeddings_raw_with_hidden.pth"
    woz_dic_path_for_cluster = "./data/MULTIWOZ2.1/intent2id.pkl"
    woz_embedding_path_for_cluster = "./finetune_results/woz_embeddings_sub.pth"
    
    # Model
    input_shape = (20, 768)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    b_size = 1024
    n_clusters = 8 #180
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = None #'checkpoints-dcec/dcec_model_att_99.h5'

    # clustering
    cluster_data_path = "clustering_results/data_att_woz_pair.pkl"
    cluster_label_path =  "clustering_results/labels_att_woz_pair.pkl"
    cluster_weight_path =  "clustering_results/weight_att_woz_pair.pkl"
    cluster_id = 0

    #################### For scBERT ####################

    se_path_for_sc = "data/semantic/raw_data_se_not_tokenize.pkl"
    se_dic_path_for_sc = "data/semantic/intent2id_se_not_tokenize.pkl"

    atis_path_for_sc = "data/atis/raw_data_not_tokenize.pkl"
    atis_dic_path_for_sc = "data/atis/intent2id_not_tokenize.pkl"

    neg_size = 100
    hidden_dim = 768





opt = Config()