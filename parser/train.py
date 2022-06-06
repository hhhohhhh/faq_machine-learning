#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/26 14:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/26 14:47   wangfc      1.0         None
"""

import math
from torch import nn, optim
from tqdm import tqdm
import torch
from parser.parser_utils import minibatches, AverageMeter,Parser
import logging
logger = logging.getLogger(__name__)

# -----------------
# Primary Functions
# -----------------
def train(parser:Parser, train_data, dev_data, output_path,
          batch_size=1024, n_epochs=10, lr=0.0005,device= 'cpu'):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """



    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func`
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    best_dev_UAS = 0
    optimizer = optim.Adam(params=parser.model.parameters(),lr= lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        logger.info("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser=parser,train_data=train_data,dev_data=dev_data,optimizer=optimizer,
                        loss_func=loss,batch_size=batch_size,device= device)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            logger.info(f"New best dev UAS= {best_dev_UAS} ! Saving model.")
            torch.save(parser.model.state_dict(), output_path)


def train_for_epoch(parser:Parser, train_data, dev_data,
                    optimizer:optim.Adam, loss_func:nn.CrossEntropyLoss, batch_size,
                    device):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    # Places model in "train" mode, i.e. apply dropout layer
    parser.model.train()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long().to(device)
            train_y = torch.from_numpy(train_y.nonzero()[1]).long().to(device)

            ### YOUR CODE HERE (~5-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            logits = parser.model(train_x)

            loss = loss_func(logits,train_y)

            loss.backward()

            optimizer.step()

            prog.update(1)

            loss_meter.update(loss.item())

    logger.info ("Average Train Loss: {}".format(loss_meter.avg))

    logger.info("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    logger.info("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS
