import random
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from math import sqrt
import FaBo9Axis_MPU9250
import sys
import RPi.GPIO as GPIO
import requests
import json
import smbus
import time


tf.reset_default_graph() # 再実行時にグラフをクリア

CKPTDIR = "/home/pi/ckptdir" # ディレクトリはあらかじめ作っておく

sess = tf.InteractiveSession()

#-----AI code----------------
def get_datas(datas):
        a = mpu9250.readAccel()
        g = mpu9250.readGyro()
        m = mpu9250.readMagnet()
        value = (a['x'],  a['y'], a['z'], g['x'], g['y'], g['z'],m['x'],m['y'],m['z'])
        data = (value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8])
        sys.stdout.write("\r" + str(data))
        sys.stdout.flush()
        datas.append(data)

def isHit(datas):
    Hit = 0;
#         print datas[k][0]
    if(do_test(datas)):
        print "hit!?"
        Hit += 1

    if(Hit):
        return True
    else:
        return False
fishing_dic = {0: 'Failed', 1: 'Hit'}

saver = tf.train.Saver()

frommodel = False

ckpt = tf.train.get_checkpoint_state(CKPTDIR)
if ckpt:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = ckpt.model_checkpoint_path
    print(("load {0}".format(last_model)))
    # 学習済みモデルを読み込む
    saver.restore(sess, last_model)
    frommodel = True
else:
    print("initialization")
    # 初期化処理
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

def do_test(param):
    seq = param
    start = time.time()
#     print("start : "+str(start))
#     print(conv_shape(seq))
    ans = sess.run(pred, feed_dict={x: conv_shape(seq)})[0]
    print ans
#     print("end : "+str(start - time.time()))
    ans = np.array(ans)
    #print ans
    #print np.argmax(ans)
    #print("sequence: ", seq)
    #print("answer: ", fishing_dic[np.argmax(ans)])
    if(np.argmax(ans)):
        return True
    else:
        return False
#-----Finalizer--------------
def fin():

    bus.close() #Analog close
    sys.exit() #Exit

#============================

# LSTM_learning(dataset)

# while True: pass
#Analog IO init
bus = smbus.SMBus(1)

mpu9250 = FaBo9Axis_MPU9250.MPU9250()
i=0
datas = []
flag = True

#Start fishing!
try:
    while True:
        get_datas(datas)
        time.sleep(0.1)

        if(len(datas) == 10):
            params = convert(datas)
            #print params
            ans = isHit(params)
#             ans = False

            if(ans):
                print "called"
                flag = False
#                 servo.ChangeDutyCycle(7)
                #forward(0x3A)
                #count_rotation(ROTATE_COUNT)
#                 stop()
#                 brake()
#                 while True:
#                     # ボタン押下判定
#                     if( GPIO.input(BUTTONPIN)):

                datas = ['hoge']
#                         servo.ChangeDutyCycle(10)
#                         time.sleep(0.2)
#                         #down()
#                         break;

                ans = False
            else:
                print "Failed"

            datas.pop()
        i += 1
#         print i
        flag = True
except KeyboardInterrupt:
    sess.close()
    fin()
