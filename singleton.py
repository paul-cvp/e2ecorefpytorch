#!/usr/bin/env python

import os
import sys

sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import torch

if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch

from torch import optim
from torch import nn
from torch.autograd import Variable
import coref_model_pytorch as cm
import coref_model_dataset as cmdata
import util

if __name__ == "__main__":
    config = util.get_config("experiments.conf")['best']
    coref_data = cmdata.TrainCorefDataset(config)
    # train_loader = torch.utils.data.DataLoader(coref_data, batch_size=1, shuffle=True, num_workers=1)

    model = cm.CorefModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["decay_rate"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["decay_frequency"])
    # criterion = nn.NLLLoss().cuda()  # loss function

    model.train()
    for epoch in range(150):
        initial_time = time.time()
        running_loss = 0.0
        for i in range(coref_data.length):
            model.zero_grad()
            word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids = coref_data.__getitem__(i)
            # data [0 -> 4] inputs 5 is training flag [6 -> 8] predictions to be trained against

            # gold = (gold_starts, gold_ends, cluster_ids)

            predicted, loss = model(word_emb, char_index, text_len, speaker_ids, genre, 1, gold_starts, gold_ends, cluster_ids)
            # loss = criterion(predicted, gold)
            scheduler.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"])
            scheduler.step()

            # print statistics
            print("[x] Loss: {}".format(loss.item()))
            running_loss += loss.item()
            global_step = i+1
            report_frequency = 2
            if global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = global_step / total_time
                average_loss = running_loss / report_frequency
                print("[x] Epoch: {} [{}] loss={:.2f}, steps/s={:.2f}".format(epoch, global_step, average_loss, steps_per_second))
                running_loss = 0.0

    print('Finished Training')
