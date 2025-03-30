# 57. How to randomly place p elements in a 2D array?

import numpy as np

# 10x10 배열에 5개의 1 무작위 배치
n = 10
p = 5

# 먼저 0으로 채운 배열 생성
Z = np.zeros((n, n), dtype=int)
print("초기 배열:")
print(Z)

# 무작위로 p개의 위치 선택
np.put(Z, np.random.choice(range(n*n), p, replace=False), 1)
print("\np개의 요소 무작위 배치 후:")
print(Z)
print(f"1의 개수: {np.sum(Z)}")  # p와 일치해야 함

# 또 다른 방법
Z2 = np.zeros(n*n, dtype=int)
Z2[:p] = 1
np.random.shuffle(Z2)
Z2 = Z2.reshape(n, n)
print("\n또 다른 방법:")
print(Z2)
print(f"1의 개수: {np.sum(Z2)}")