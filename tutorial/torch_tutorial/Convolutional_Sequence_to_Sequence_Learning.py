#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
from Sequence_to_Sequence_Learning_with_Neural_Networks import *

# In[2]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


BATCH_SIZE = 128


# ## Building the Model
# Next up is building the model. As before, the model is made of an encoder and decoder. The encoder *encodes* the input sentence, in the source language, into a *context vector*. The decoder *decodes* the context vector to produce the output sentence in the target language.
# 
# ### Encoder
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.device = device

        # scale 起作用的原理不是特别清楚
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        # 使用 nn.ModuleList() 函数可以建立多个conv list:
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim, # 因为使用 glu 函数
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2) # 使用padding，保持输出的长度seq_len保持不变
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size = src.shape[0]  # src = [batch size, src len]
        src_len = src.shape[1]

        # create position tensor：pos = [0, 1, 2, 3, ..., src len - 1]  ---> pos = [1, src len] ---> pos = [batch size, src len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)  # tok_embedded = pos_embedded = [batch size, src len, emb dim]
        pos_embedded = self.pos_embedding(pos)

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)  # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)  # conv_input = [batch size, src len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)  # conv_input = [batch size, hid dim, src len]

        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer : conv_input = [batch size, hid dim, src len]
            conved = conv(self.dropout(conv_input))  # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function: 通过 glu之后维度缩减了一半
            conved = F.glu(conved, dim=1)  # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # ...end convolutional blocks

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))  # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale  # combined = [batch size, src len, emb dim]
        return conved, combined


# ### Decoder
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 device,
                 max_length=100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx  # 对于 trg 做padding用
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder 部分：
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim 使向量都具有相同的维度
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))  # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale  # combined = [batch size, trg len, emb dim]

        # decorder 中实现 Attention 机制：
        # decorder 中 combined 作为Query， encoder 中 encoder_conved 作为 Key， encoder_combined 作为 Value
        # 计算 energy : combined  向量 （作为Query） 和每个 encoder_conved （作为为 Key）的相互影响： Q * K^T
        # energy = [batch size, trg len, src len]
        # print(f'combined.shape:{combined.shape},encoder_conved.permute(0, 2, 1)={encoder_conved.permute(0, 2, 1).shape}')
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # 对enegy 做 softmax 得到 attention
        attention = F.softmax(energy, dim=2)  # attention = [batch size, trg len, src len]

        # 得到 weighted sum of values: attention * V
        attended_encoding = torch.matmul(attention,         # encoder_combined = [batch_size, src_len, emb_dim]
                                         encoder_combined)  # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)  # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2,
                                                                1)) * self.scale  # attended_combined = [batch size, hid dim, trg len]
        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)  # tok_embedded = [batch size, trg len, emb dim]
        pos_embedded = self.pos_embedding(pos)  # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)  # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)  # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)  # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        # decorder 中进行 conv
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # 创建 padding :need to pad so decoder can't "cheat" 对前 kernel_size - 1 进行 padding
            # padding = [batch_size, hid_dim, kernel_size -1]
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            # 对每一句话进行padding：conv_input = [batch_size, hid_dim, trg_len]
            padded_conv_input = torch.cat((padding, conv_input),
                                          dim=2)  # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)  # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)  # conved = [batch size, hid dim, trg len]

            # calculate attention 获取 attetion 并且 weighted sum of 。。。
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)
            # attention = [batch size, trg len, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale  # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))  # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))  # output = [batch size, trg len, output dim]

        return output, attention


# ### Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention


# ## Training the Seq2Seq Model

EMB_DIM = 256
HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10  # number of conv. blocks in encoder
DEC_LAYERS = 10  # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3  # must be odd!
DEC_KERNEL_SIZE = 3  # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = PAD_IDX

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
        src_lengths = torch.tensor(batch[1]).to(device)
        trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
        trg_lengths = torch.tensor(batch[3]).to(device)


        optimizer.zero_grad()

        # 输入 trg 除去 最后一个字符
        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# The evaluation loop is the same as the training loop, just without the gradient calculations and parameter updates.

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = torch.tensor(batch[0], dtype=torch.long).type(torch.LongTensor).to(device)
            src_lengths = torch.tensor(batch[1]).to(device)
            trg = torch.tensor(batch[2], dtype=torch.long).type(torch.LongTensor).to(device)
            trg_lengths = torch.tensor(batch[3]).to(device)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Again, we have a function that tells us how long each epoch takes.

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we train our model. Notice we have reduced the `CLIP` value from 1 to 0.1 in order to train this model more reliably. With higher `CLIP` values, the gradient occasionally explodes.
# 
# Although we have almost twice as many parameters as the attention based RNN model, it actually takes around half the time as the standard version and about the same time as the packed padded sequences version. This is due to all calculations being done in parallel using the convolutional filters instead of sequentially using RNNs. 
# 
# **Note**: this model always has a teacher forcing ratio of 1, i.e. it will always use the ground truth next token from the target sequence. This means we cannot compare perplexity values against the previous models when they are using a teacher forcing ratio that is not 1. See [here](https://github.com/bentrevett/pytorch-seq2seq/issues/39#issuecomment-529408483) for the results of the attention based RNN using a teacher forcing ratio of 1. 


N_EPOCHS = 10
CLIP = 0.1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_data, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, dev_data, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut5-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# We then load the parameters which obtained the lowest validation loss and calculate the loss over the test set.


model.load_state_dict(torch.load('tut5-model.pt'))



# ## Inference

# Now we can can translations from our model with the `translate_sentence` function below.
# 
# The steps taken are:
# - tokenize the source sentence if it has not been tokenized (is a string)
# - append the `<sos>` and `<eos>` tokens
# - numericalize the source sentence
# - convert it to a tensor and add a batch dimension
# - feed the source sentence into the encoder
# - create a list to hold the output sentence, initialized with an `<sos>` token
# - while we have not hit a maximum length
#   - convert the current output sentence prediction into a tensor with a batch dimension
#   - place the current output and the two encoder outputs into the decoder
#   - get next output token prediction from decoder
#   - add prediction to current output sentence prediction
#   - break if the prediction was an `<eos>` token
# - convert the output sentence from indexes to tokens
# - return the output sentence (with the `<sos>` token removed) and the attention from the last layer


# Next, we have a function what will display how much the model pays attention to each input token during each step of the decoding.

example_idx = 100

src_encoded = dev_en[example_idx]
trg_encoded = dev_cn[example_idx]

src = [inv_en_dict[encoded] for encoded in src_encoded]
trg = [inv_cn_dict[encoded] for encoded in trg_encoded]
print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(sentence=src,
                                            src_init_token="BOS", src_eos_token="EOS", src_word_dict=en_dict,

                                            trg_init_token="BOS", trg_eos_token="EOS", trg_word_dict=cn_dict,
                                            inv_trg_word_dict=inv_cn_dict,
                                            model=model, device=device, max_len=50,base_model=BASE_MODEL)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

bleu = calculate_bleu(zip(dev_en_tokenized, dev_cn_tokenized), src_init_token="BOS", src_eos_token="EOS", src_word_dict=en_dict,
                      trg_init_token="BOS", trg_eos_token="EOS", trg_word_dict=cn_dict,
                      inv_trg_word_dict=inv_cn_dict,
                      model=model, device=device, max_len=50)
print(f'bleu:{bleu}')

