import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import base.data as bd
import base.model as bm

import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

import cf_native_guide.native_guide as ng
import cf_wachter.wachter as w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading dataset')
dataloader_train, dataset_train = bd.get_UCR_dataloader(split='train')
dataloader_test, dataset_test = bd.get_UCR_dataloader(split='test')

model = bm.SimpleCNN().to(device)

print('Training model')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

def trainer(model, dataloader_train, criterion):
    running_loss = 0

    model.train()

    for _, (inputs, labels) in enumerate(dataloader_train):
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

    for _, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        preds = model(inputs)
        loss = criterion(preds, labels.argmax(dim=-1))
        
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader_train)
    
    return train_loss

epochs = 100

for epoch in range(epochs):
    train_loss = trainer(model, dataloader_train, loss)
    if epoch % 10 == 0:
        val_loss = validator(model, dataloader_test, loss)
        print(f'Val loss: {val_loss:.3f}')
    print(f'Train e{epoch:=4} loss: {train_loss:.3f}')

print('Generating counterfactual')
sample, label = dataset_test[0]
print('Start with native guide')
cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, dataset_test, model)
print('Start with Wachter et al.')
cf_wg, prediction_wg = w.wachter_gradient_uni_cf(sample, dataset_test, model)
cf_w, prediction_w = w.wachter_genetic_uni_cf(sample, model, step_size=np.mean(dataset_test.std))

print('Results')
print(label, prediction_ng, prediction_w, prediction_wg)

fig, axs = plt.subplots(7)
fig.suptitle('Counterfactual Explanations')
i = 0
axs[i].set_title('Original sample')
axs[i].plot(sample)
i += 1
axs[i].set_title('Native Guide Counterfactual')
axs[i].plot(cf_ng)
i += 1
axs[i].set_title('Wachter et al Gradient Counterfactual')
axs[i].plot(cf_wg)
i += 1
axs[i].set_title('Wachter et al Genetic Counterfactual')
axs[i].plot(cf_w)
i += 1

axs[i].plot(cf_ng)
axs[i].plot(sample)
i += 1
axs[i].plot(cf_wg)
axs[i].plot(sample)
i += 1
axs[i].plot(cf_w)
axs[i].plot(sample)
i += 1
plt.savefig('counterfactuals.png')
