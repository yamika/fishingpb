#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TensorFlow r1.0.0
# Python 2.7.6
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import FaBo9Axis_MPU9250
import time
import sys
import RPi.GPIO as GPIO
import requests
import json

def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[12] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == 1:
            labels.append([0.0,1.0])
        elif l == 0:
            labels.append([1.0,0.0])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[3:] for item in dataset]
    raw_data = [list(item)[:9] for item in raw_data]
    return np.array(raw_data)


def get_data2(dataset):
  """データセットをnparrayに変換する"""
  raw_data = [list(item)[:4] for item in dataset]
  return np.array(raw_data)


with tf.gfile.FastGFile("/home/pi/fishingpb/fishing_model7.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

mpu9250 = FaBo9Axis_MPU9250.MPU9250()

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(4,GPIO.IN)
GPIO.setup(5,GPIO.IN)
no = 21
i=0
datas = []
try:
    while True:
        #value = (a['x'],  a['y'], a['z'], g['x'], g['y'], g['z'],m['x'],m['y'],m['z'])
        #sys.stdout.write("\rax=%f, ay=%f, az=%f, gx=%f,gy=%f,gz=%f,mx=%f,my=%f,mz=%f" % value)
        #sys.stdout.flush()

        a = mpu9250.readAccel()
        g = mpu9250.readGyro()
        m = mpu9250.readMagnet()
        value = (a['x'],  a['y'], a['z'], g['x'], g['y'], g['z'],m['x'],m['y'],m['z'])
        data = (value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8])
        datas.append(data)
        time.sleep(0.1)
        if(len(datas) == 10):
            if(GPIO.input(4)):
                sess = tf.Session()
                    # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_operation_by_name('output')
                predictions = sess.run('output:0',{'input:0':np.array(datas)})
                print predictions
                noHit = 0;
                Hit = 0;
                for k in range(0,10):
                    maxIndex = -1
                    maxScore = 0
                    score = 0;
                    for l in range(0,2):
                      score = predictions[k][l]
                      if(score > maxScore):
                        maxIndex = l
                        maxScore = score
                    if(maxIndex == 0):
                        Hit+=1;
                    elif(maxIndex == 1):
                        noHit+=1;

                if(Hit > noHit):
                    print Hit
                    print ("Hit")
                elif(noHit > Hit):
                    print noHit
                    print("Nothing")
                elif(Hit == noHit):
                    print("Nothing")
            datas.pop()
        i += 1
        if(i == 10):
            i = 0
except KeyboardInterrupt:
    sys.exit()
