#!/usr/bin/env python
# coding: utf-8

from Sequence_to_Sequence_Learning_with_Neural_Networks import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

# Set the random seeds for reproducability.
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Create the iterators.
BATCH_SIZE = 128
SAVE_MODEL_NAME = 'tut4-model-jieba-tokenizer.pt' if USE_JIEBA_TOKENIZER else 'tut4-model.pt'


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lenths):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))  # embedded = [src len, batch size, emb dim]

        embedded_packed = pack_padded_sequence(embedded, src_lenths)

        outputs, hidden = self.rnn(embedded_packed)
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        outputs_unpacked, _ = pad_packed_sequence(outputs)
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs_unpacked, hidden


# ### Attention
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        "decoder 中的 hidden state 对 encoder的 output 层进行attention"
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times : 沿着第2个维度 重复 src_len 次
        hidden = hidden.unsqueeze(1).repeat(1, src_len,1)  # [batch size,1, dec hid dim] ---> [batch size, src len, dec hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch size, src len, enc hid dim * 2]

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        # 这儿的 attension 方法也叫做 多层感知机的方式
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # energy = [batch size, src len, dec hid dim]

        # 最后进行 softmax 得到 attension scores
        attention = self.v(energy).squeeze(2)  # attention= [batch size, src len, dec hid dim] --->[batch size, src len]

        # 对 attention 做 mask = [batch_size，seq_length]
        # 因为decoder的时候，这儿模型是逐个token进行decoder的，因此 attention= [batch size, src len]
        attention_masked = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention_masked, dim=1)


# ### Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        # 在decoder 的时候插入 attention机制
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)  # input = [1, batch size], 单个词对应的 index

        embedded = self.dropout(self.embedding(input))  # embedded = [1, batch size, emb dim]

        # 对 previous hidden state 与 encoder_outputs 做 attention
        # 增加 mask 方式 mask = [seq_length ,batch_size]
        a = self.attention(hidden, encoder_outputs, mask)  # a = [batch size, src len]

        a = a.unsqueeze(1)  # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)  # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)  # weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))  # prediction = [batch size, output dim]
        # 增加 attention的输出
        return prediction, hidden.squeeze(0), a.squeeze(1)


# ### Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src):
        # src = [src len, batch size]
        mask = (src != self.src_pad_idx).permute(1, 0)
        # mask = [batch_size, seq_length ]
        return mask

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        # mask 用于 attention的时候只关注 non pad 的部分。 mask = [seq_length ,batch_size]
        mask = self.create_mask(src)

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # 增加 attention的输出, 在增加mask方式
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)

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


# ## Training the Seq2Seq Model

# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)


# We use a simplified version of the weight initialization scheme used in the paper. Here, we will initialize all biases to zero and all weights from $\mathcal{N}(0, 0.01)$.

# In[15]:


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)


# Calculate the number of parameters. We get an increase of almost 50% in the amount of parameters from the last model.

# In[16]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# We create an optimizer.

optimizer = optim.Adam(model.parameters())

# We initialize the loss function.
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# We then create the training loop...
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
        src_lengths = torch.tensor(batch[1]).to(device)
        trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
        trg_lengths = torch.tensor(batch[3]).to(device)
        optimizer.zero_grad()

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


# ...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing.
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
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


# Finally, define a timing function.

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Then, we train our model, saving the parameters that give us the best validation loss.

# N_EPOCHS = 10
# CLIP = 1
#
# best_valid_loss = float('inf')
#
# for epoch in range(N_EPOCHS):
#
#     start_time = time.time()
#
#     train_loss = train(model, train_data, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, dev_data, criterion)
#
#     end_time = time.time()
#
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), SAVE_MODEL_NAME)
#
#     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Finally, we test the model on the test set using these "best" parameters.
model.load_state_dict(torch.load(SAVE_MODEL_NAME))

# test_loss = evaluate(model, test_iterator, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# for i in range(100, 120):
#     translate_dev(model, i)
#     print()

# example_idx = 100
#
# src_encoded = dev_en[example_idx]
# trg_encoded = dev_cn[example_idx]
#
# src = [inv_en_dict[encoded] for encoded in src_encoded]
# trg = [inv_cn_dict[encoded] for encoded in trg_encoded]
# print(f'src = {src}')
# print(f'trg = {trg}')
#
# translation, attention = translate_sentence(sentence=src,
#                                             src_init_token="BOS", src_eos_token="EOS", src_word_dict=en_dict,
#
#                                             trg_init_token="BOS", trg_eos_token="EOS", trg_word_dict=cn_dict,
#                                             inv_trg_word_dict=inv_cn_dict,
#                                             model=model, device=device, max_len=50)
#
# print(f'predicted trg = {translation}')
#
# display_attention(src, translation, attention)

bleu = calculate_bleu(zip(dev_en_tokenized, dev_cn_tokenized), src_init_token="BOS", src_eos_token="EOS", src_word_dict=en_dict,
                      trg_init_token="BOS", trg_eos_token="EOS", trg_word_dict=cn_dict,
                      inv_trg_word_dict=inv_cn_dict,
                      model=model, device=device, max_len=50)
print(f'bleu:{bleu}')
