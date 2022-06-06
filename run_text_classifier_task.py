#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/28 14:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 14:51   wangfc      1.0         None
"""
import os
import sys


def main():
    try:
        from conf.config_parser import OUTPUT_DIR, OUTPUT_MODEL_SUBDIR, LOG_FILENAME, TASK_NAME, GPU_MEMORY_CONFIG, \
            CORPUS, DATASET_NAME, DATA_SUBDIR, DATA_DIR, STOPWORDS_PATH, \
            model_config, LOG_LEVEL
        # MAX_SEQ_LENGTH, MODEL_NAME, PRETRAINED_MODEL_DIR, VOCAB_FILENAME, EPOCH, OPTIMIZER_NAME, LEARNING_RATE, TRAIN_BATCH_SIZE
        from utils.common import init_logger
        log_dir = os.path.join(OUTPUT_DIR, 'log')
        logger = init_logger(output_dir=log_dir, log_filename=LOG_FILENAME, log_level=LOG_LEVEL)
        logger.info(f'开始任务 {TASK_NAME}')
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)

        from tasks.text_classifier_task import TransformerClassifierTask
        classifier_task = TransformerClassifierTask(task=TASK_NAME, corpus=CORPUS, dataset_name=DATASET_NAME,
                                                    data_subdir=DATA_SUBDIR, data_dir=DATA_DIR,
                                                    label_column=LABEL_COLUMN,
                                                    output_dir=OUTPUT_DIR,
                                                    output_model_subdir=OUTPUT_MODEL_SUBDIR,
                                                    stopwords_path=STOPWORDS_PATH,
                                                    debug=DEBUG, run_eagerly=RUN_EAGERLY,
                                                    **model_config,
                                                    )
        classifier_task._prepare_data()
        classifier_task._prepare_model()
        classifier_task.train()

        logger.info(f"训练完成")
    except Exception as e:
        logger.error(e, exc_info=True)


def run_chinese_text_classifier_task():
    import time
    import torch
    import numpy as np
    from importlib import import_module
    from cli.arguments.task_argument_parser import create_text_classifier_task_argument_parser
    from models.models_torch.train_eval import train, init_network

    paser = create_text_classifier_task_argument_parser()
    args = paser.parse_args()

    dataset = 'corpus/THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils.utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from models.models_torch.train_utils_torch import build_dataset, build_iterator, get_time_dif

    x = import_module('models.models_torch.classifiers.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)


def run_text_classifier_task_torch():

    try:
        from cli.arguments.task_argument_parser import create_text_classifier_task_argument_parser
        from tasks.torch_task import TextClassifierTaskTorch
        from models.models_torch.classifiers.TextCNN import Config as TextCNNConfig
        from models.models_torch.classifiers.TextCNN import Model as TextCNNModel
        from tokenizations.tokenizatoions_utils import basic_tokenizer

        paser = create_text_classifier_task_argument_parser(mode='train',adversarial_train_mode='free',cuda_no=0)
        args = paser.parse_args()
        model_name = args.model
        adversarial_train_mode = args.adversarial_train_mode
        optimizer_name = args.optimizer_name
        output_dir = args.output_dir
        random_sampling = args.random_sampling
        test_no = "02"

        if adversarial_train_mode is None:
            output_dir = os.path.join(output_dir, model_name)
        else:
            output_dir = os.path.join(output_dir, f"{model_name}_{adversarial_train_mode}_{test_no}")

        from utils.common import init_logger
        log_dir = os.path.join(output_dir, 'log')
        logger = init_logger(output_dir=log_dir, log_filename=args.mode, log_level='info')

        dataset = 'corpus/THUCNews'  # 数据集
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
        embedding = 'embedding_SougouNews.npz'

        if args.model in ["TextCNN", "TextCNNAT"]:
            model_config = TextCNNConfig(dataset=dataset, embedding=embedding,
                                         model_name=model_name,
                                         adversarial_train_mode=adversarial_train_mode)

        data_dir = os.path.join(dataset, 'data')
        num_train_epochs = model_config.num_epochs
        batch_size = model_config.batch_size
        learning_rate = model_config.learning_rate
        tokenizer = basic_tokenizer(ues_word=False)

        text_classifier_task_torch = TextClassifierTaskTorch(mode=args.mode,
                                                             cuda_no=args.cuda_no,
                                                             data_dir=data_dir, output_dir=output_dir,
                                                             model_config=model_config,
                                                             tokenizer=tokenizer,
                                                             vocab_path=model_config.vocab_path,
                                                             optimizer_name=optimizer_name, learning_rate=learning_rate,
                                                             train_filename='train.txt', dev_filename='dev.txt',
                                                             test_filename='test.txt',
                                                             num_train_epochs=num_train_epochs,
                                                             train_batch_size=batch_size,
                                                             adversarial_train_mode=adversarial_train_mode,
                                                             random_sampling=random_sampling,
                                                             attack_iters = args.attack_iters,
                                                             minibatch_replays = args.minibatch_replays
                                                             )
        if args.mode == 'train':
            text_classifier_task_torch.train()
        else:
            text_classifier_task_torch.test()
    except Exception as e:
        logger.error(msg=e,exc_info=True)
        sys.exit(0)


if __name__ == '__main__':
    # main()
    # run_chinese_text_classifier_task()
    run_text_classifier_task_torch()
