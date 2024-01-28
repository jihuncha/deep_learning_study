# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# normalize -> 입력 이미지의 픽셀 값을 0.0 ~ 1.0 으로 정규화, false면 0~255 사이의 값 유지
# 데이터를 특정 범위로 변환하는 처리 = 정규화 (normalization)
# 신경망의 입력 데이터에 특정 변환을 가하는 것 = 전처리 (pre-processing)
# flatten 은 입력 이미지 값을 평탄하게 (1차원 배열로) 만들지 결정, false면 1 x 28 x 28 로 3차원 배열, true면 784개의 원소로 이루어진 1차원 배열
# one_hot_label 은 원-핫 인코딩 (one-hot encoding) 형태로 저장할지.
# 원-핫 인코딩은 정답을 뜻하는 원소만 1이고, 나머지는 0인 배열 ([0,0,0,0,0,1,0,0,0])
# false 면 숫자 7,2, 등 숫자형태로 그냥 저장하고 true면 원-핫 인코딩을 하여 저장한다.

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i]) # mnist dataset의 사진을 한장씩 for문을 돌리면서 predict함수로 분류한다.
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得 배열에서 확률이 가장 높은 인덱스를 추출 = 예측 결과
    if p == t[i]:
        accuracy_cnt += 1 # 결과 반영

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))