
# coding: utf-8

# In[1]:

import random
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from math import sqrt
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# In[2]:

negative_dataset = np.genfromtxt("./negative_data/deepfishing_negative1.csv", delimiter=',', dtype=["S32", int, "S32",float, float, float, float, float, float, float, float, float, int])
positive_dataset = np.genfromtxt("./active_data/deepfishing_active1.csv", delimiter=',', dtype=["S32", int, "S32",float, float, float, float, float, float, float, float, float, int])


# In[3]:

NUM_CLASSES = 2 #  2クラス分類
NUM_STEPS = 2000 #  学習回数
LEN_SEQ = 10 # 系列長
SIZE_INPUT = 1 # 入力データ数
NUM_DATA = 186  # データ数
NUM_TEST = 20 # テスト用のデータ数
SIZE_BATCH = 10 # バッチサイズ
NUM_NODE = 1024  # ノード数
LEARNING_RATE = 0.01  # 学習率


# In[4]:

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[3:6] for item in dataset]
    raw_data = np.array(raw_data)
    return raw_data


# In[5]:

print get_data(active_dataset)[310][2]


# In[5]:

def create_sqrt_data(dataset):
  tmp = []
  sqrt_data = []
  for i in dataset:
      tmp.append(([sqrt(i[0]*i[0]+i[1]*i[1]+i[2]*i[2])]))
  sqrt_data = np.array(tmp)
  return sqrt_data


# In[6]:

positive_data = get_data(positive_dataset)
positive_data = create_sqrt_data(positive_data)
print (positive_data)


# In[7]:

def set_matrix(dataset,length):
  tmp = []
  ret = []
  print length
  for i in range(0,length):
    tmp = []
    for j in range(0,10):
      tmp.append(dataset[i*10+j])
    ret.append(np.array(tmp))
  return np.array(ret)


# In[8]:

len = (positive_data).size
positive_data = set_matrix(positive_data,len/10)


# In[9]:

def create_label(num,length):
  label = []
  for i in range(0,length):
    label.append([num])

  return np.array(label)


# In[24]:




# In[10]:

negative_data = get_data(negative_dataset)
negative_data = create_sqrt_data(negative_data)
neg_len = negative_data.size
print neg_len
negative_data = set_matrix(negative_data,neg_len/10)


# In[11]:

print positive_data.shape
print negative_data.shape


# In[12]:

positive_label = create_label(1,((positive_data).size)/10)
print positive_label.shape
negative_label = create_label(0,((negative_data).size)/10)
print negative_label.shape


# In[13]:

x_data = np.r_[positive_data, negative_data]
print x_data.shape
#print x_data
y_data = np.r_[positive_label, negative_label]
#print y_data
x_train_data = x_data[NUM_TEST:]
y_train_label = y_data[NUM_TEST:]
print y_train_label.shape
x_test_data = x_data[:NUM_TEST]
y_test_label = y_data[:NUM_TEST]


# In[14]:

x = tf.placeholder(tf.float32, [None, LEN_SEQ, SIZE_INPUT])
t = tf.placeholder(tf.int32, [None, 1])
print x


# In[15]:

t_on_hot = tf.one_hot(t, depth=NUM_CLASSES, dtype=tf.float32)


# In[16]:

x_transpose = tf.transpose(x, [1, 0, 2])
x_reshape = tf.reshape(x_transpose, [-1, 1])
x_split = tf.split(x_reshape, LEN_SEQ, 0)


# In[17]:

lstm_cell = rnn.BasicLSTMCell(NUM_NODE, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x_split, dtype=tf.float32)


# In[18]:

w = tf.Variable(tf.random_normal([NUM_NODE, NUM_CLASSES]))
b = tf.Variable(tf.random_normal([NUM_CLASSES]))
logits = tf.matmul(outputs[-1], w) + b
pred = tf.nn.softmax(logits)


# In[19]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)


# In[20]:

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)


# In[21]:

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[54]:

sess_t = tf.Session()
sess_t.run(tf.global_variables_initializer())


# In[55]:

batch_x, batch_t = x_data[0:10], y_data[0:10]


# In[56]:

print batch_t


# In[22]:

loss_train = []
acc_train = []
loss_test = []
acc_test = []


# In[24]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())
start = time()
i = 0
for _ in range(NUM_STEPS):
    cycle = int((NUM_DATA-NUM_TEST)/SIZE_BATCH)
    begin = int(SIZE_BATCH * (i % cycle))
    end = int(begin + SIZE_BATCH)
    batch_x, batch_t = x_train_data[begin:end], y_train_label[begin:end]
    i += 1
    sess.run(train_step, feed_dict={x: batch_x, t: batch_t})
    if i % 100 == 0:
        loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: batch_x, t: batch_t})
        loss_train.append(loss_)
        acc_train.append(acc_)
        loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x: x_test_data, t: y_test_label})
        loss_test.append(loss_test_)
        acc_test.append(acc_test_)
        print("[%i STEPS] %f sec" % (i, (time() - start)))
        print("[TRAIN] loss : %f, accuracy : %f" %(loss_, acc_))
        print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))


# In[ ]:
