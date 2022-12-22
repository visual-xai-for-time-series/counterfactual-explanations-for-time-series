import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import base_data as bd
import base_model as bm

import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

import cf_native_guide.native_guide as ng
import cf_wachter.wachter as w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloader_train, dataset_train = bd.get_UCR_dataloader(split='train')
dataloader_test, dataset_test = bd.get_UCR_dataloader(split='test')
model = bm.SimpleCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

def trainer(model, dataloader_train, criterion):
    running_loss = 0

    model.train()

    for idx, (inputs, labels) in enumerate(dataloader_train):
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels.argmax(dim=-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader_train)
    
    return train_loss


def validator(model, dataloader_test, criterion):
    running_loss = 0

    model.eval()

    for idx, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        preds = model(inputs)
        loss = criterion(preds, labels.argmax(dim=-1))
        
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader_train)
    
    return train_loss

epochs = 10

for epoch in range(epochs):
    train_loss = trainer(model, dataloader_train, loss)
    if epoch % 10 == 0:
        print('Val', validator(model, dataloader_test, loss))
    print('Train', train_loss)


sample, label = dataset_test[0]
cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, dataset_test, model)
cf_w, prediction_w = w.wachter_genetic_uni_cf(sample, model)

print(label, prediction_ng, prediction_w)

fig, axs = plt.subplots(5)
fig.suptitle('Counterfactual Explanations')
axs[0].plot(sample)
axs[1].plot(cf_ng)
axs[2].plot(cf_w)
axs[3].plot(sample)
axs[3].plot(cf_ng)
axs[4].plot(sample)
axs[4].plot(cf_w)
plt.savefig('exp.png')
