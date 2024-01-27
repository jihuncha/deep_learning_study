import numpy as np

A = np.array([1,2,3,4])

print(A)

print(np.ndim(A)) # 배열의 차원수 확인

print(A.shape) # 배열의 형상 확인 (원소수 4개) -> 튜플 반환

print(A.shape[0])

B = np.array([[1,2], [3,4], [5,6]])

print(B)

print(np.ndim(B))

print(B.shape)

print('=================================================================')
# 행렬의 곱
A = np.array([[1,2], [3,4]])
print(A.shape)
B = np.array([[5,6], [7,8]])
print(B.shape)

print(np.dot(A,B))
