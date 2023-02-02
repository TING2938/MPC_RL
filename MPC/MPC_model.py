import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

# 设置随机数种子保证论文可复现
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class Args():
    def __init__(self) -> None:
        self.batch_size = 1
        self.lr = 0.001
        self.epochs = 10
        self.patience = 7
        self.device = "cpu"
        self.data_train = np.random.randint(-20, 20, 100)
        self.label_train = (self.data_train > 8).astype(int)
        self.data_val = np.random.randint(-15, 15, 20)
        self.label_val = (self.data_val > 8).astype(int)


args = Args()


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


def train():
    train_dataset = TensorDataset(torch.Tensor(
        args.data_train), torch.Tensor(args.label_train))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(torch.Tensor(
        args.data_val), torch.Tensor(args.label_val))
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)

    model = MPC_model(1, 32, 16, 2).to(args.device)  # 分类问题
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # =========================train=======================
        for idx, (label, inputs) in enumerate(train_dataloader):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs: torch.Tensor = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(
            100 * acc / nums, np.average(train_epoch_loss)))

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        acc, nums = 0, 0
        for idx, (label, inputs) in enumerate(valid_dataloader):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, label)
            valid_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        val_acc.append(100 * acc / nums)

        print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(
            epoch, 100 * acc / nums, np.average(valid_epoch_loss)))
        # ==================early stopping======================
        early_stopping(
            valid_epochs_loss[-1], model=model, path=r'./')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # ====================adjust lr========================
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

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


def pred(val):
    model = MPC_model(1, 32, 16, 2)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    val = torch.tensor(val).reshape(1, -1).float()
    # 需要转换成相应的输入shape，而且得带上batch_size，因此转换成shape=(1,1)这样的形状
    res = model(val)
    # real: tensor([[-5.2095, -0.9326]], grad_fn=<AddmmBackward0>) 需要找到最大值所在的列数，就是标签
    res = res.max(axis=1)[1].item()
    print("predicted label is {}, {} {} 8".format(
        res, val.item(), ('>' if res == 1 else '<')))


# %%
if __name__ == '__main__':
    train()
    pred(24)
    pred(3.14)
    pred(7.8)  # 这个会预测错误，所以数据量对于深度学习很重要
