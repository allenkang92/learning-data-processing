# 55. What is the equivalent of enumerate for numpy arrays?

import numpy as np

# 테스트용 1D 배열
Z = np.arange(10)
print("원본 배열:", Z)

# 방법 1: 일반 파이썬 enumerate 사용
print("\n일반 파이썬 enumerate 사용:")
for i, value in enumerate(Z):
    print(f"인덱스: {i}, 값: {value}")

# 방법 2: NumPy 특화 방식 - ndenumerate 사용 (다차원 배열에 유용)
print("\nnp.ndenumerate 사용 (다차원 배열 가능):")
for index, value in np.ndenumerate(Z):
    print(f"인덱스: {index}, 값: {value}")

# 다차원 배열에서의 예제
Z_2d = np.arange(9).reshape(3, 3)
print("\n2D 배열:", Z_2d)
print("\nnp.ndenumerate로 2D 배열 순회:")
for index, value in np.ndenumerate(Z_2d):
    print(f"인덱스: {index}, 값: {value}")