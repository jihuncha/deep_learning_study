import numpy as np


def step_function(x):
    # 단순한 계단함수
    # 다만 실수만 받아드린다.
    # if x > 0:
    #     return 1
    # else:
    #     return 0

    # 배열을 받으려면
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)
print(y.astype(np.int64))

# [-1.  1.  2.]
# [False  True  True]
# [0 1 1]
# 부등호 연산을 수행한 결과를 y에 대입하였음 -> 그 다음에 Int형으로 면환함