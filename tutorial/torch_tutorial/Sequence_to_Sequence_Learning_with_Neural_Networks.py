#!/usr/bin/env python
# coding: utf-8

# # 1 - Sequence to Sequence Learning with Neural Networks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu

import spacy
import nltk
import jieba
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import random
import math
import time


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

CUDA_NO = 'cuda:0'
device = torch.device(CUDA_NO if torch.cuda.is_available() else 'cpu')

USE_JIEBA_TOKENIZER = False
SAVE_MODEL_NAME = 'tut6-model-jieba-tokenizer.pt' if USE_JIEBA_TOKENIZER else 'tut6-model.pt'

batch_size = 64
BASE_MODEL = 'transformer'
BATCH_FIRST = False if BASE_MODEL == 'rnn' else True


def load_data(in_file, use_jieba_tokenizer=False):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("\t")
            #             print(line[0])

            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            # split chinese sentence into characters

            if use_jieba_tokenizer:
                cn.append(["BOS"] + jieba.lcut(line[1]) + ["EOS"])
            else:
                cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])

    return en, cn


train_file = "nmt/en-cn/train.txt"
dev_file = "nmt/en-cn/dev.txt"

train_en_tokenized, train_cn_tokenized = load_data(train_file, USE_JIEBA_TOKENIZER)
dev_en_tokenized, dev_cn_tokenized = load_data(dev_file, USE_JIEBA_TOKENIZER)

## 用以标注未知的单词和PAD的字符
UNK_IDX = 0
PAD_IDX = 1


def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 2
    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    return word_dict, total_words


en_dict, en_total_words = build_dict(train_en_tokenized)
cn_dict, cn_total_words = build_dict(train_cn_tokenized)
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}

print(f'en_total_words:{en_total_words},cn_total_words:{cn_total_words}')


def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    # sort sentences by english lengths
    # 该函数写得很好啊
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        # 按照 en 的句子长度排序
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


train_en, train_cn = encode(train_en_tokenized, train_cn_tokenized, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en_tokenized, dev_cn_tokenized, en_dict, cn_dict)


def get_minibatches(n, minibatch_size, shuffle=True):
    # 返回每个batch的index
    idx_list = np.arange(0, n, minibatch_size)  # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs, batch_first=False, reverse=False):
    # 返回 seqs的 np.array = [n_samples, max_len] 和 x_lengths = [n_samples,]
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    ## 初始化全部为1是否更加合适？？？
    # x = np.zeros((n_samples, max_len)).astype('int32') #
    x = np.ones((n_samples, max_len)).astype('int32')  #

    x_lengths = np.array(lengths).astype("int32")
    if reverse:
        for idx, seq in enumerate(seqs):
            x[idx, -lengths[idx]:] = seq[::-1]
    else:
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
    if batch_first == False:
        x_reshape = x.transpose([1, 0])
        return x_reshape, x_lengths  # x_mask
    else:
        return x, x_lengths  # x_mask


def gen_examples(en_sentences, cn_sentences, batch_size, batch_first=False, reverse=True):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        # 获取每个batch的 token_id
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        # 转换为np
        mb_x, mb_x_len = prepare_data(mb_en_sentences, batch_first=batch_first, reverse=reverse)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences, batch_first=batch_first)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex


train_data = gen_examples(train_en, train_cn, batch_size, batch_first=BATCH_FIRST, reverse=True)
random.shuffle(train_data)
dev_data = gen_examples(dev_en, dev_cn, batch_size, batch_first=BATCH_FIRST, reverse=True)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))  # embedded = [src len, batch size, emb dim]

        # 因为需要处理可变长度序列，先要进行 pack ---> RNN --> unpack
        embed_packed = pack_padded_sequence(embedded, lengths=src_lengths)

        # 此时得到的 hidden,cell 是最后 非<pad> 字符的输出
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        return hidden, cell



# ### Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        # 每次输入一个单词长度的batch,每次输出 prediction, hidden, cell
        input = input.unsqueeze(0)  # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


# ### Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]  # trg_len 是该batch中最大的长度？？？
        trg_vocab_size = self.decoder.output_dim

        # 初始化输出：tensor to store decoder outputs = [trg_len, batch_size, trg_vocab_size]
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_lengths)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        # 对于每个batch作 1 到 trg_len 的循环
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# # Training the Seq2Seq Model
# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)

INPUT_DIM = en_total_words
OUTPUT_DIM = cn_total_words

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)



def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





optimizer = optim.Adam(model.parameters())


criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # print(f'batch:{i}')
        src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
        src_lengths = torch.tensor(batch[1]).to(device)
        trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
        trg_lengths = torch.tensor(batch[3]).to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(src, src_lengths, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)




def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # src = batch.src
            # trg = batch.trg
            src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
            src_lengths = torch.tensor(batch[1]).to(device)
            trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
            trg_lengths = torch.tensor(batch[3]).to(device)

            output = model(src, src_lengths, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Next, we'll create a function that we'll use to tell us how long an epoch takes.

# In[34]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_dev(model, i, base_model='rnn', batch_first=False):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(f'en_sent = {en_sent}')
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print(f'cn_sent = {cn_sent}')

    if batch_first == False:
        mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(-1, 1)).long().to(device)
        mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
        bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)
        trg = torch.zeros(20, 1, dtype=torch.long)
        trg[0, :] = bos
        trg_device = trg.to(device)
    else:
        mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
        bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)
        trg = torch.zeros(1, 20, dtype=torch.long)
        trg[:, 0] = bos
        trg_device = trg.to(device)

    if base_model == 'rnn':
        outputs = model.forward(mb_x, mb_x_len, trg_device)
    elif base_model == 'transformer':
        outputs, _ = model.forward(mb_x, trg_device[:, :-1])
    translation = outputs.squeeze(1)[1:, :].argmax(1)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print('模型结果：' + " ".join(trans))


def translate_sentence(sentence, src_init_token, src_eos_token, src_word_dict,
                       trg_init_token, trg_eos_token, trg_word_dict, inv_trg_word_dict,
                       model, device, max_len=50, base_model='rnn',batch_first=False):
    # 设置 eval 模式
    model.eval()
    # 对句子进行 tokenizer
    if isinstance(sentence, str):
        tokenizer = nltk.word_tokenize()
        tokens = [token.text.lower() for token in tokenizer(sentence)]
        tokens = [src_init_token] + tokens + [src_eos_token]
    else:
        # tokens = [token.lower() for token in sentence]
        tokens = sentence

    # 对 token 进行 numerizer
    src_indexes = [src_word_dict.get(token, 0) for token in tokens]

    if batch_first == False:
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq,1]
    else:
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)  # [1,]

    if base_model == 'rnn':
        "首先输入 <SOS>,然后每次做计算后得到一个token，循环迭代直到输出 <EOS>"
        with torch.no_grad():
            # 对输入语句进行 encoder: 因为 RNN需要做pack和 unpack来得到最后一个非 pad 的 hidden_state
            encoder_outputs, hidden = model.encoder(src_tensor, src_len)
        # 创建 mask ： 使得 attention 的时候只关注那些 non pad 的位置
        mask = model.create_mask(src_tensor)
        # 创建 tag的初始token为 <SOS>
        trg_indexes = [trg_word_dict[trg_init_token]]
        # 初始化 每个输出对于 src中每个单词的attention score
        attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

        # 循环做预测
        for i in range(max_len):
            # 每次循环预测的时候只看 trg的前一个字符 和 encoder部分的attention输出
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

            with torch.no_grad():
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

            # 记录 attention
            attentions[i] = attention

            pred_token = output.argmax(1).item()

            trg_indexes.append(pred_token)
            # 直到 pred_token 输出为 <EOS> 的时候才停止
            if pred_token == trg_word_dict[trg_eos_token]:
                break
        trg_tokens = [inv_trg_word_dict[i] for i in trg_indexes]
        return trg_tokens[1:], attentions[:len(trg_tokens) - 1]

    elif base_model == 'cnn':
        with torch.no_grad():
            # encoder部分得到 src 的 encoder_conved, encoder_combined 作为 Key, Value
            encoder_conved, encoder_combined = model.encoder(src_tensor)
        # 创建 tag的初始token为 <SOS>
        trg_indexes = [trg_word_dict[trg_init_token]]
        for i in range(max_len):
            # 每次循环预测的时候只看 trg的前面所有字符 和 encoder部分的attention输出
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

            pred_token = output.argmax(2)[:, -1].item()

            trg_indexes.append(pred_token)

            if pred_token == trg_word_dict[trg_eos_token]:
                break

        trg_tokens = [inv_trg_word_dict[i] for i in trg_indexes]
        # 最后返回的 attention = [batch_size,trg_len, src_len]
        return trg_tokens[1:], attention
    elif base_model == 'transformer':
        ""
        # 计算 src_mask = [1, src_len, encorder_dim]
        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            # 计算 encorder 的输出
            enc_src = model.encoder(src_tensor, src_mask)

        # 创建 tag的初始token为 <SOS>, 在循环迭代的过程中会逐渐增加 trg_index 输入到decoder中
        trg_indexes = [trg_word_dict[trg_init_token]]

        for i in range(max_len):
            # trg_tensor = [1, trg_len]
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            # 计算 trg_mask = [1, trg_len, ???]
            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                # 计算 decorder 的输出： output = [1,trg_len, dec_dim] , attention = [1,trg_len, src_len]
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # 使用 greed search, 计算最新的预测的 pred_token
            pred_token = output.argmax(2)[:, -1].item()
            print(f'trg_index:{i},trg_indexes:{trg_indexes},output:{output.argmax(2)[:, :].data},pred_token:{pred_token}')

            #  增加 预测的token
            trg_indexes.append(pred_token)

            if pred_token == trg_word_dict[trg_eos_token]:
                break

        trg_tokens = [inv_trg_word_dict[i] for i in trg_indexes]

        return trg_tokens[1:], attention


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + sentence,
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def calculate_bleu(data, src_init_token, src_eos_token, src_word_dict,
                   trg_init_token, trg_eos_token, trg_word_dict, inv_trg_word_dict,
                   model, device, max_len=50, base_model='rnn',batch_first=False):
    trgs = []
    pred_trgs = []

    for index, datum in enumerate(data):
        if index % 100 == 0:
            print(f'index:{index}')

        src = datum[0]
        trg = datum[1]

        pred_trg, _ = translate_sentence(src, src_init_token, src_eos_token, src_word_dict,
                                         trg_init_token, trg_eos_token, trg_word_dict, inv_trg_word_dict,
                                         model, device,max_len,base_model,batch_first )

        # cut off <eos> token
        pred_trg = pred_trg[:-1]
        pred_trgs.append(pred_trg)

        trgs.append([trg[1:-1]])

    nltk_bleu_score = corpus_bleu(trgs, pred_trgs)

    return nltk_bleu_score


if __name__ == '__main__':

    N_EPOCHS = 20
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_data, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, dev_data, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(model.state_dict(), SAVE_MODEL_NAME)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load(SAVE_MODEL_NAME))
    print(f'load(SAVE_MODEL_NAME):{SAVE_MODEL_NAME}')

    # test_loss = evaluate(model, test_iterator, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    for i in range(100, 120):
        translate_dev(model, i)
        print()
