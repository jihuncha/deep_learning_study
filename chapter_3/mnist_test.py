import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록!!설정

from dataset.mnist import load_mnist

# 훈련이미지, 훈련 레이블 / 시험이미지, 시험레이블
(x_train, t_train), (x_test, t_test ) = load_mnist(flatten=True, normalize=False)

# normalize -> 입력 이미지의 픽셀 값을 0.0 ~ 1.0 으로 정규화, false면 0~255 사이의 값 유지
# flatten 은 입력 이미지 값을 평탄하게 (1차원 배열로) 만들지 결정, false면 1 x 28 x 28 로 3차원 배열, true면 784개의 원소로 이루어진 1차원 배열
# one_hot_label 은 원-핫 인코딩 (one-hot encoding) 형태로 저장할지.
# 원-핫 인코딩은 정답을 뜻하는 원소만 1이고, 나머지는 0인 배열 ([0,0,0,0,0,1,0,0,0])
# false 면 숫자 7,2, 등 숫자형태로 그냥 저장하고 true면 원-핫 인코딩을 하여 저장한다.

# 각데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# (60000, 784)
# (60000,)
# (10000, 784)
# (10000,)
