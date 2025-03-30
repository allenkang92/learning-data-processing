# 67. Considering a four dimensions array, how to get sum over the last two axis at once?

import numpy as np

# 4차원 배열 생성 (2x3x4x5 크기)
Z = np.random.randint(0, 10, (2, 3, 4, 5))
print("4차원 배열 형태:", Z.shape)

# 방법 1: 마지막 두 축을 따로 합산
result1 = Z.sum(axis=3).sum(axis=2)
print("방법 1 (축을 따로 합산):", result1.shape)
print(result1)

# 방법 2: 마지막 두 축을 한 번에 합산
result2 = Z.sum(axis=(-2, -1))
print("방법 2 (튜플로 여러 축 한 번에 합산):", result2.shape)
print(result2)

# 결과 확인 (두 방법의 결과가 동일한지)
print("두 결과가 동일함:", np.array_equal(result1, result2))
