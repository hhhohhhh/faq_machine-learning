#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/4 16:10 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/4 16:10   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
from __future__ import print_function

import json
import random
import time
import os
import torch
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer


# from model.WordAveragingModel import WordAveragingModel
from model.RNN import RNN, LSTM
from model.FastText import FastText
from model.TextCNN import TextCNNV1, TextCNN1d
from model.BertGRU import BertGRU
from utils.evaluation import binary_accuracy
from utils.train_utils import *
from utils.utils import count_parameters
from conf.configs import *
from utils.common import init_logger
from utils.data_processing import generate_n_gram
from tokenization.tokenization_albert import FullTokenizer
from tokenization.tokenization_albert import get_tokenize_and_cut
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

init_logger(log_file=LOG_FILE)

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device(f'cuda:{GPU_NO}' if torch.cuda.is_available() else 'cpu')

if MODEL_NAME in ['bert','albert']:
    # tokenizer= FullTokenizer(vocab_file=VOCAB_FILE)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_DIR)
    logger.info(f"len(tokenizer.vocab) = {len(tokenizer.vocab)}")
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    logger.info("init_token={}, eos_token={}, pad_token={}, unk_token={}".format(init_token, eos_token, pad_token, unk_token))
    init_token_idx = tokenizer.convert_tokens_to_ids([init_token])[0]
    eos_token_idx = tokenizer.convert_tokens_to_ids([eos_token])[0]
    pad_token_idx = tokenizer.convert_tokens_to_ids([pad_token])[0]
    unk_token_idx = tokenizer.convert_tokens_to_ids([unk_token])[0]
    logger.info("init_token_idx={}, eos_token_idx={}, pad_token_idx={}, unk_token_idx={}".format(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx))
    max_input_length = 512 # tokenizer.max_model_input_sizes[BERT_MODEL]
    logger.info(f"max_input_length={max_input_length}")
else:
    # tokenize='spacy'
    tokenize = lambda x: x.split()
"""
Field对象指定要如何处理某个字段.
"""
preprocessing = None
if MODEL_NAME == "fasttext":
    # 当使用 FastText模型的时候，需要生成 n_gram: 可以在 field 的时候输入 preprocessing
    preprocessing = generate_n_gram

include_lengths = False
if MODEL_NAME == 'lstm':
    # 使用RNN模型的时候，需要做pack，需要知道每句话的长度：当 include_lengths为true的时候，得到 text 为 （tokens,length）
    include_lengths = True

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


train_data_path = os.path.join(MODEL_OUTPUT_DIR,'train.json')
test_data_path = os.path.join(MODEL_OUTPUT_DIR,'test.json')

if os.path.exists(train_data_path) and os.path.exists(test_data_path):
    if MODEL_NAME in ['bert']:
        # 定义各个 field
        TEXT = data.Field(batch_first=True)
    else:
        TEXT = data.Field()
    LABEL = data.LabelField(batch_first=True)
    # 创建 fields
    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    # 创建 datasets
    train_data, test_data = data.TabularDataset.splits(
        path=os.path.dirname(train_data_path),
        train='train.json',
        test='test.json',
        format='json',
        fields=fields
    )
    logger.info(f"读取数据 Number of training examples: {len(train_data)} "
                f"Number of testing examples: {len(test_data)} from {train_data_path}")
    example_index = 0
    example =vars(train_data.examples[example_index])
    logger.info(f"example[{example_index}]={example}")
    logger.info(f"train_data[{example_index}]={tokenizer.convert_ids_to_tokens(example['text'])}")

else:
    if MODEL_NAME in ['bert']:
        #
        TEXT = data.Field(batch_first=True, use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx
                          )
    else:
        TEXT = data.Field(tokenize=tokenize, include_lengths=include_lengths,
                          preprocessing=preprocessing)  # ,tokenizer_language="en-core-web-sm")

    LABEL = data.LabelField(dtype=torch.long)  # 改为 long 类型

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    logger.info(f'Number of training examples: {len(train_data)}')
    logger.info(f'Number of testing examples: {len(test_data)}')

    train_examples = [vars(t) for t in train_data]
    test_examples = [vars(t) for t in test_data]

    with open(train_data_path, 'w+') as f:
        for example in train_examples:
            json.dump(example, f)
            f.write('\n')
    logger.info(f"保存 training examples 数据到{train_data_path}")

    with open(test_data_path, 'w+') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')
    logger.info(f"保存 testing examples 数据到{test_data_path}")



# logger.info("example[0]:")
# example = vars(train_data.examples[0])
# for k, value in example.items():
#     logger.info("key={},value={}".format(k, value))

# 创建 vocabulary 。vocabulary 就是把每个单词一一映射到一个数字。
if USE_PRETRAINED_EMBEDDINGS:
    # 在分割之前创建 vocab
    TEXT.build_vocab(train_data, max_size=MAX_VACAB_SIZE, vectors=PRETRAINED_EMBEDDINGS, unk_init=torch.Tensor.normal_)
else:
    TEXT.build_vocab(train_data, max_size=MAX_VACAB_SIZE)

LABEL.build_vocab(train_data)
logger.info(f"LABEL.vocab.stoi={LABEL.vocab.stoi}")

vocab_size = len(TEXT.vocab)
output_size = len(LABEL.vocab)

# 获取  UNK_INDEX 和 PAD_INDEX
UNK_INDEX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_INDEX = TEXT.vocab.stoi[TEXT.pad_token]

train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO)
logger.info(f'Number of training examples: {len(train_data)}')
logger.info(f'Number of validation examples: {len(valid_data)}')
logger.info(f'Number of testing examples: {len(test_data)}')

logger.info(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
logger.info(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# model = WordAveragingModel(output_size=output_size,vocab_size=vocab_size,embedding_dim=EMBEDDING_DIM,padding_index=PAD_INDEX)
if MODEL_NAME.lower() == "raw-rnn":
    model = RNN(output_size=output_size, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, padding_index=PAD_INDEX,
                hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL)
elif MODEL_NAME.lower() == 'lstm':
    model = LSTM(output_size=output_size, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, padding_index=PAD_INDEX,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
elif MODEL_NAME.lower() == 'fasttext':
    model = FastText(output_size=output_size, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM,
                     padding_index=PAD_INDEX,dropout=DROPOUT)
elif MODEL_NAME.lower() == "textcnn":
    model = TextCNNV1(output_size=output_size, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM,
                      padding_idx=PAD_INDEX,n_filters=N_FILTERS, filter_sizes=FILTER_SIZES,dropout=DROPOUT)
elif MODEL_NAME.lower() == 'textcnn1d':
    model = TextCNN1d(output_size=output_size, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM,
                      padding_idx=PAD_INDEX,n_filters=N_FILTERS, filter_sizes=FILTER_SIZES,dropout=DROPOUT)
elif MODEL_NAME.lower() == 'bert':
    model = BertGRU(pretrained_bert_dir=PRETRAINED_BERT_DIR,output_size=output_size,
                   hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYERS,bidirectional=BIDIRECTIONAL,dropout=DROPOUT)
    logger.info(f'The {MODEL_NAME} model has {count_parameters(model):,} trainable parameters')
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
    logger.info(f'The {MODEL_NAME} model after frozen Bert has {count_parameters(model):,} trainable parameters')

logger.info(f'The {MODEL_NAME} model has {count_parameters(model):,} trainable parameters')

model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

if MODE == 'train':
    # 创建 iterators
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        datasets= (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        # sort_within_batch=include_lengths,
        sort =False, # By default, the train data is shuffled each epoch, but the validation/test data is sorted.
        device=device)

    # embedding 参数初始化
    if USE_PRETRAINED_EMBEDDINGS:
        # 预训练的参数 vectors = torch.Size([25002, 100])
        pretrained_embeddings = TEXT.vocab.vectors
        # 使用 预训练的词向量对 embedding 进行初始化
        model.embedding.weight.data.copy_(pretrained_embeddings)
        # 显示的设置 UNK_INDEX 和 PAD_INDEX 对应的权重 为 0
        #  PAD_INDEX 在训练的过程中会保持为0，但是 UNK_INDEX 对应的词向量会训练
        model.embedding.weight.data[UNK_INDEX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_INDEX] = torch.zeros(EMBEDDING_DIM)

    if OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_loss = float('inf')
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(epoch=epoch,model=model, iterator=train_iterator, optimizer=optimizer, loss_fn=loss_fn)
        valid_loss, valid_acc = evaluate(epoch=epoch,model=model, itetator=valid_iterator, loss_fn=loss_fn)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # 在测试集上进行测试工作
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_acc = evaluate(model, itetator=test_iterator, loss_fn=loss_fn)
    logger.info(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

elif MODE == 'predict':
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    sentences = ["This film is terrible", "This film is great"]
    for sentence in sentences:
        y_pred, probability = predict(model_name=MODEL_NAME,model=model, sentence=sentence, tokenize=tokenize, TEXT=TEXT, device=device)
        logger.info(f"sentence={sentence},y_pred={y_pred},probability={probability}")
