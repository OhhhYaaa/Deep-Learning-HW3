import profile
import numpy as np
import pandas as pd
import time, os, pickle, json, random

from imageio import imread, imwrite
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import Pool
from skimage import feature, data, color
import skimage
import cv2
from functools import partial
from sklearn import preprocessing
from sklearn.linear_model import Perceptron

from Lenet5_Scratch import *
from memory_profiler import profile
cpus = multiprocessing.cpu_count()
print(cpus)

# onehot
def self_onehot(x, c = 50) : 
    x_onehot = np.zeros([x.shape[0], c])
    for i in range(x.shape[0]) :
        x_onehot[i, x[i].astype(int)] = 1
    return x_onehot

# 讀取圖片function
def read_img_32(path) :
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


os.chdir('/home/rita/111/111-2DL/HW1')
train_idx = pd.read_table('train.txt', header = None, sep = ' ')
val_idx = pd.read_table('val.txt', header = None, sep = ' ')
test_idx = pd.read_table('test.txt', header = None, delimiter = ' ')
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)
train_y = train_idx[::, 1].astype(float)
val_y = val_idx[::, 1].astype(float)
test_y = test_idx[::, 1].astype(float)

train_y_onehot = self_onehot(train_y)
val_y_onehot = self_onehot(val_y)
test_y_onehot = self_onehot(test_y)

if __name__ == '__main__' : 
    with Pool(processes = 80) as p:
        train_pic_32 = list(tqdm(p.imap(read_img_32, train_idx[::, 0], chunksize=100), total = train_idx.shape[0]))
        val_pic_32 = list(tqdm(p.imap(read_img_32, val_idx[::, 0], chunksize=100), total = val_idx.shape[0]))
        test_pic_32 = list(tqdm(p.imap(read_img_32, test_idx[::, 0], chunksize=100), total = test_idx.shape[0]))
os.chdir('/home/rita/111/111-2DL/HW3')

train_pic_32 = np.array(train_pic_32)
val_pic_32 = np.array(val_pic_32)
test_pic_32 = np.array(test_pic_32)

X_train, Y_train, X_test, Y_test = train_pic_32, train_y_onehot, test_pic_32, test_y_onehot
X_val, Y_val = val_pic_32, val_y_onehot
X_train, X_val, X_test = X_train/float(255), X_val/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_val -= np.mean(X_val)
X_test -= np.mean(X_test)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val = np.transpose(X_val, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

train_dataloader = Self_DataLoader(X_train, train_y_onehot, batch_size = 32, shuffle=True)
val_dataloader = Self_DataLoader(X_val, val_y_onehot, batch_size = 32, shuffle=True)
test_dataloader = Self_DataLoader(X_test, test_y_onehot, batch_size = 32)

# train_dataloader = Self_DataLoader(X_train, train_y, batch_size = 128, shuffle=True)

model = LeNet5()
optim = SGD(model.get_params(), lr=1e-2, reg=0)
criterion = CrossEntropyLoss()
print('Start Training !')

# TRAIN
# 加 dataloader 版本

@profile(precision=4, stream=open('./memory/memory_profiler_Scratch.log','w+'))

def train_lenet5(model, train_dataloader, n_epochs = 30):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for i in range(n_epochs):
        # train
        temp_train_loss, temp_train_acc = 0, 0
        with tqdm(total=train_dataloader.num_batches) as pbar:
            for X_batch, Y_batch in train_dataloader :
                Y_batch = self_onehot(Y_batch)
                Y_pred = model.forward(X_batch)
                loss, _ = criterion.get(Y_pred, Y_batch)
                dout = Y_pred - Y_batch  # pred - label
                model.backward(dout)
                optim.step()
                
                pred_y = np.argmax(Y_pred, axis=1)
                acc = np.mean(pred_y == np.argmax(Y_batch, axis=1))
                
                temp_train_loss += loss
                temp_train_acc += acc
                
                pbar.update(1)
        temp_train_loss /= train_dataloader.num_batches
        temp_train_acc /= train_dataloader.num_batches
        train_acc.append(temp_train_acc)
        train_loss.append(temp_train_loss)
        
        print("%s%% Epoch: %s, loss: %.5f" % (100*i/n_epochs, i + 1, loss))
        
        # validation
        Y_pred = model.forward(X_val)
        pred_y = np.argmax(Y_pred, axis=1)
        acc = np.mean(pred_y == val_y)
        # Y_batch = self_onehot(Y_val)
        loss, _ = criterion.get(Y_pred, Y_val)
        val_loss.append(loss)
        val_acc.append(acc)
    
    return model, train_loss, train_acc, val_loss, val_acc

model, train_loss, train_acc, val_loss, val_acc = train_lenet5(
    model = model, train_dataloader = train_dataloader, n_epochs = 1
)

# weights = model.get_params()
# with open("./model/lenet5_grad.pkl","wb") as f:
#     pickle.dump(weights, f)

# with open("./loss/lenet5_train_loss_grad.txt", "w") as fp:
#     json.dump(train_loss, fp)
# with open("./acc/lenet5_train_acc_grad.txt", "w") as fp:
#     json.dump(train_acc, fp)    
# with open("./loss/lenet5_val_loss_grad.txt", "w") as fp:
#     json.dump(val_loss, fp)   
# with open("./acc/lenet5_val_acc_grad.txt", "w") as fp:
#     json.dump(val_acc, fp)    
    
# # draw
# plt.title('Loss_grad')
# plt.plot(range(n_epochs), train_loss, label="train")
# plt.plot(range(n_epochs), val_loss, label="val", c = 'red')
# plt.legend()
# plt.savefig('./figure/lenet5_loss_grad.png')
# plt.show()

# plt.title('Accuracy_grad')
# plt.plot(range(n_epochs), train_acc, label="train")
# plt.plot(range(n_epochs), val_acc, label="val", c = 'red')
# plt.legend()
# plt.savefig('./figure/lenet5_acc_grad.png')
# plt.show()











































