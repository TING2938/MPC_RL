# 设置随机数种子保证论文可复现
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class Args():
    def __init__(self) -> None:
        self.batch_size = 32
        self.lr = 0.0001
        self.in_dim = 1
        self.out_dim = 1
        self.n_hidden_1 = 64
        self.n_hidden_2 = 64
        self.epochs = 2
        self.patience = 20
        self.device = "cpu"
        N = 1000000
        self.data_x_train = np.random.rand(N).reshape(-1, 1) * 40 - 20
        self.data_y_train = self.func(
            self.data_x_train) + np.random.randn(N, 1)

        self.data_x_val = np.random.rand(50).reshape(-1, 1) * 30 - 15
        self.data_y_val = self.func(self.data_x_val)

        self.data_x_mean = self.data_x_train.mean()
        self.data_x_std = self.data_x_train.std()
        self.data_y_mean = self.data_y_train.mean()
        self.data_y_std = self.data_y_train.std()

        self.norm_data()

    def func(self, val):
        return 3 * val * val - val - 1

    def norm_data(self):
        self.data_x_train = (self.data_x_train -
                             self.data_x_mean) / (self.data_x_std + 1e-7)
        self.data_y_train = (self.data_y_train -
                             self.data_y_mean) / (self.data_y_std + 1e-7)
        self.data_x_val = (self.data_x_val - self.data_x_mean) / \
            (self.data_x_std + 1e-7)
        self.data_y_val = (self.data_y_val - self.data_y_mean) / \
            (self.data_y_std + 1e-7)


class MPC_model(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score+self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss


def train(args: Args):
    train_dataset = TensorDataset(torch.tensor(
        args.data_x_train), torch.tensor(args.data_y_train))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(torch.tensor(
        args.data_x_val), torch.tensor(args.data_y_val))
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)

    model = MPC_model(args.in_dim, args.n_hidden_1,
                      args.n_hidden_2, args.out_dim).to(args.device)  # 分类问题
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_epochs_loss = []
    valid_epochs_loss = []

    # early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        # =========================train=======================
        for idx, (data_x, data_y) in enumerate(train_dataloader):
            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(data_y, outputs)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.item())
            if idx % (len(train_dataloader)//2) == 0:
                print("epoch={}/{}, {}/{} of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(valid_dataloader):
            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            valid_epoch_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        # ==================early stopping======================
        # early_stopping(
        #     valid_epochs_loss[-1], model=model, path=r'./')
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        # ====================adjust lr========================
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        # if epoch in lr_adjust.keys():
        #     lr = lr_adjust[epoch]
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print('Updating learning rate to {}'.format(lr))

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_epoch_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")


def pred(val, args: Args):
    model = MPC_model(args.in_dim, args.n_hidden_1,
                      args.n_hidden_2, args.out_dim).to(args.device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    val_norm = (val - args.data_x_mean) / (args.data_x_std + 1e-7)
    x = torch.tensor(val_norm).reshape(1, -1).float()
    # 需要转换成相应的输入shape，而且得带上batch_size，因此转换成shape=(1,1)这样的形状
    res = model(x)
    res = res.item() * args.data_y_std + args.data_y_mean

    # real: tensor([[-5.2095, -0.9326]], grad_fn=<AddmmBackward0>) 需要找到最大值所在的列数，就是标签
    print(f"model({val}), pre: {res}, expect: {args.func(val)}")


# %%
if __name__ == '__main__':
    args = Args()
    train(args)
    pred(24, args)
    pred(3.14, args)
    pred(7.8, args)
