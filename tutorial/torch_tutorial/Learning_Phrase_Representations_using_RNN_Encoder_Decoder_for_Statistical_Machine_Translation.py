#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
from Sequence_to_Sequence_Learning_with_Neural_Networks import *


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


SAVE_MODEL_NAME =  'tut2-model-jieba-tokenizer.pt' if USE_JIEBA_TOKENIZER else 'tut2-model.pt'
print(f'SAVE_MODEL_NAME"{SAVE_MODEL_NAME}')
print(f'en_total_words:{en_total_words},cn_total_words:{cn_total_words}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size = BATCH_SIZE,
#     device = device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src,src_lengths):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))  # embedded = [src len, batch size, emb dim]

        embedded_packed = pack_padded_sequence(embedded, src_lengths)
        outputs, hidden = self.rnn(embedded_packed)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # decoder的RNN 的输入和输出： emb_dim + hid_dim
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        # decoder 的 LINEAR 的输入： emb_dim + hid_dim * 2
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0) # input = [1, batch size]

        embedded = self.dropout(self.embedding(input)) # embedded = [1, batch size, emb dim]

        #emb_con 是 embedded 和 context的 concate 拼接
        emb_con = torch.cat((embedded, context), dim=2) # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # decoder 的 LINEAR 的输入： embedded + hidden + context
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


# ## Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, src_lengths , trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src,src_lengths)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

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
# 
# The rest of this tutorial is very similar to the previous one. 
# 
# We initialise our encoder, decoder and seq2seq model (placing it on the GPU if we have one). As before, the embedding dimensions and the amount of dropout used can be different between the encoder and the decoder, but the hidden dimensions must remain the same.

# In[14]:


# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)



def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model):,} trainable parameters')

# We initiaize our optimizer.
optimizer = optim.Adam(model.parameters(),lr=1e-3)

# We also initialize the loss function, making sure to ignore the loss on `<pad>` tokens.

# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index= PAD_IDX)


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

        output = model(src,src_lengths, trg)

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



def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
            src_lengths = torch.tensor(batch[1]).to(device)
            trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
            trg_lengths = torch.tensor(batch[3]).to(device)

            output = model(src, src_lengths, trg, 0)  # turn off teacher forcing = [trg_len,batch_size, output_dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim) # [(trg_len-1)*batch_size , output_dim]
            trg = trg[1:].view(-1)  # [(trg_len-1)*batch_size]

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# We'll also define the function that calculates how long an epoch takes.

# In[21]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



N_EPOCHS = 20
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_data, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, dev_data, criterion)

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
    translate_dev(model,i)
    print()