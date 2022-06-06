#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time
import torch
from parser.parser_model import ParserModel
from parser.parser_utils import  load_and_preprocess_data
from parser.train import train

import supar.cmds.biaffine_dep as biaffine_dep_cmd
from utils.torch.environment import _setup_torch_environment
device = _setup_torch_environment()

from torchtext.datasets import PennTreebank

def train_fast_neural_dependency_parser(mode='train',debug = False):
    """
    CS224N 2018-19: Homework 3
    run.py: Run the dependency parser.
    Sahil Chopra <schopra8@stanford.edu>
    """
    # Note: Set debug to False, when training on entire corpus
    # debug = True

    from utils.common import init_logger
    model_name = 'fast_neural_dependency_parser'
    output_dir = os.path.join('output',model_name)
    log_output_dir = os.path.join(output_dir,'log')
    log_filename = mode
    logger = init_logger(output_dir=log_output_dir, log_filename=log_filename)

    # output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "/model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # assert(torch.__version__ == "1.2.0"),  "Please install torch version 1.0.0"

    logger.info(80 * "=")
    logger.info("INITIALIZING")
    logger.info(80 * "=")
    # 初始化
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()

    # 使用 embeddings 初始化 ParserModel
    model = ParserModel(embeddings,n_features=parser.n_features,n_classes=parser.n_trans,
                        hidden_size=200,dropout_prob=0.5,device=device)

    parser.model = model.to(device)

    logger.info("took {:.2f} seconds\n".format(time.time() - start))


    if mode =='train':
        logger.info(80 * "=")
        logger.info("TRAINING")
        logger.info(80 * "=")

        train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005,
              device=device)

    if mode == 'test':
        logger.info(80 * "=")
        logger.info("TESTING")
        logger.info(80 * "=")
        logger.info("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        logger.info("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        logger.info("- test UAS: {:.2f}".format(UAS * 100.0))
        logger.info("Done!")



if __name__ == "__main__":
    mode='test'
    debug = False
    # train_fast_neural_dependency_parser(mode=mode,debug=debug)

    biaffine_dep_cmd