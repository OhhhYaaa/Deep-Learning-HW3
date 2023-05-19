import numpy as np
import pandas as pd
import os, cv2, time
from imageio import imread, imwrite
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
# from .autonotebook import tqdm as notebook_tqdm
import multiprocessing
from multiprocessing import Pool

# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torchvision
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# import torch.nn.functional as F

import tensorflow as tf
import numpy as np
from memory_profiler import profile

# 讀取圖片function
def read_img(path) :
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    return img

def read_img_32(path) :
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# onehot
def self_onehot(x, c = 50) : 
    x_onehot = np.zeros([x.shape[0], c])
    for i in range(x.shape[0]) :
        x_onehot[i, int(x[i])] = 1
    return x_onehot


# 讀取index
os.chdir('/home/rita/111/111-2DL/HW1')
train_idx = np.array(pd.read_table('train.txt', header = None, sep = ' '))
val_idx = np.array(pd.read_table('val.txt', header = None, sep = ' '))
test_idx = np.array(pd.read_table('test.txt', header = None, delimiter = ' '))
train_y = train_idx[::, 1].astype(float)
val_y = val_idx[::, 1].astype(float)
test_y = test_idx[::, 1].astype(float)
train_onehot_y = self_onehot(train_y)
val_onehot_y = self_onehot(val_y)
test_onehot_y = self_onehot(test_y)
# os.chdir('/home/rita/111/111-2DL/HW3')


# 讀取圖片
# 2 mins
# os.chdir('/home/rita/111/111-2DL/HW1')
with Pool(processes = 80) as p:
    train_pic = list(tqdm(p.imap(read_img_32, train_idx[::, 0], chunksize=100), total = train_idx.shape[0]))
    val_pic = list(tqdm(p.imap(read_img_32, val_idx[::, 0], chunksize=100), total = val_idx.shape[0]))
    test_pic = list(tqdm(p.imap(read_img_32, test_idx[::, 0], chunksize=100), total = test_idx.shape[0]))
os.chdir('/home/rita/111/111-2DL/HW3')

train_pic = np.array(train_pic)
val_pic = np.array(val_pic)
test_pic = np.array(test_pic)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
# import tensorflow.compat.v1 as tf
print("TensorFlow version:", tf.__version__)

# test on 50 figure
# https://zhuanlan.zhihu.com/p/134149111

# tf.enable_eager_execution()

def preprocess(x, y):
    # tf.cast : Casts a tensor to a new type.
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 32, 32, 3])
    y = tf.one_hot(y, depth=50)  # one_hot 编码
    return x, y

batch_size = 32
# 加载数据集
x_train, y_train, x_test, y_test = train_pic.astype(np.uint8), train_y.astype(np.uint8),  test_pic.astype(np.uint8), test_y.astype(np.uint8)
x_val, y_val = val_pic.astype(np.uint8), val_y.astype(np.uint8)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(10000)  # 打乱训练集样本
train_db = train_db.batch(batch_size)
train_db = train_db.map(preprocess)

val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_db = val_db.shuffle(10000)  # 打乱训练集样本
val_db = val_db.batch(batch_size)
val_db = val_db.map(preprocess)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000)  # 打乱测试集样本
test_db = test_db.batch(batch_size)
test_db = test_db.map(preprocess)


# 创建模型
with tf.device('cpu'):
    model = keras.Sequential([
        # 卷积层1
        keras.layers.Conv2D(6, 5),  # 使用6个5*5的卷积核对单通道32*32的图片进行卷积，结果得到6个28*28的特征图
        keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 对28*28的特征图进行2*2最大池化，得到14*14的特征图
        keras.layers.ReLU(),  # ReLU激活函数
        # 卷积层2
        keras.layers.Conv2D(16, 5),  # 使用16个5*5的卷积核对6通道14*14的图片进行卷积，结果得到16个10*10的特征图
        keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 对10*10的特征图进行2*2最大池化，得到5*5的特征图
        keras.layers.ReLU(),  # ReLU激活函数
        # 卷积层3
        keras.layers.Conv2D(120, 5),  # 使用120个5*5的卷积核对16通道5*5的图片进行卷积，结果得到120个1*1的特征图
        keras.layers.ReLU(),  # ReLU激活函数
        # 将 (None, 1, 1, 120) 的下采样图片拉伸成 (None, 120) 的形状
        keras.layers.Flatten(),
        # 全连接层1
        keras.layers.Dense(84, activation='relu'),  # 120*84
        # 全连接层2
        keras.layers.Dense(50, activation='softmax')  # 84*10
    ])

    @profile(precision=4, stream=open('./memory/memory_profiler_keras.log','w+'))

    def train_lenet(model):
        model.build(input_shape=(batch_size, 32, 32, 3))
        model.summary()

        # model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        # 训练
        history = model.fit(train_db, epochs=30, validation_data = val_db, use_multiprocessing = True)
        return model, history

    model, history = train_lenet(model)

model.save('./model/keras_Lenet5')
print('Finish Training')

# plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="val", c = 'red')
plt.legend()
plt.savefig('./figure/keras_cnn_loss.png')
plt.show()
# plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label="train")
plt.plot(history.history['val_accuracy'], label="val", c = 'red')
plt.legend()
plt.savefig('./figure/keras_cnn_acc.png')
plt.show()









