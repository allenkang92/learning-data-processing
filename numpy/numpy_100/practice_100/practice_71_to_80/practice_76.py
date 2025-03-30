# 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])

import numpy as np

# 1차원 배열 생성
Z = np.arange(10)
print("원본 배열 Z:", Z)

# 방법 1: 반복문 사용
n = 3  # 각 행의 원소 개수
result1 = np.zeros((len(Z) - n + 1, n))
for i in range(len(Z) - n + 1):
    result1[i] = Z[i:i + n]
print("\n방법 1 (반복문):")
print(result1)

# 방법 2: 배열 슬라이싱과 스택킹 사용
rows = []
for i in range(len(Z) - n + 1):
    rows.append(Z[i:i + n])
result2 = np.vstack(rows)
print("\n방법 2 (슬라이싱과 스택킹):")
print(result2)

# 방법 3: 스트라이드 트릭 사용 (가장 효율적)
from numpy.lib.stride_tricks import as_strided

# 스트라이드 트릭을 사용하여 전체 배열에 대한 이동 윈도우 뷰 생성
result3 = as_strided(Z, shape=(len(Z)-n+1, n), 
                     strides=(Z.itemsize, Z.itemsize))
print("\n방법 3 (스트라이드 트릭):")
print(result3)

# 방법 4: NumPy 1.20.0 이상의 sliding_window_view 사용
try:
    from numpy.lib.stride_tricks import sliding_window_view
    result4 = sliding_window_view(Z, n)
    print("\n방법 4 (sliding_window_view):")
    print(result4)
except:
    print("\n방법 4는 NumPy 1.20.0 이상에서만 사용 가능합니다.")
