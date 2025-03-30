# 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?

import numpy as np

Z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
print(Z)

# 이 문제는 슬라이딩 윈도우 형태의 배열을 생성하는 문제입니다.
# 각 행이 Z 배열의 연속된 4개 요소로 구성되며, 각 행은 1칸씩 이동합니다.

# 방법 1: 반복문 사용
n = 4  # 슬라이딩 윈도우 크기
rows = Z.size - n + 1  # 결과 배열의 행 수
R1 = np.zeros((rows, n), dtype=Z.dtype)

for i in range(rows):
    R1[i] = Z[i:i+n]
    
print("\n방법 1 (반복문):")
print(R1)

# 방법 2: 스트라이드 트릭 사용 (가장 효율적)
from numpy.lib.stride_tricks import as_strided

R2 = as_strided(Z, shape=(Z.size - n + 1, n), 
                strides=(Z.itemsize, Z.itemsize))
print("\n방법 2 (스트라이드 트릭):")
print(R2)

# 방법 3: NumPy 1.20.0 이상에서 슬라이딩 윈도우 뷰 사용
try:
    from numpy.lib.stride_tricks import sliding_window_view
    R3 = sliding_window_view(Z, window_shape=n)
    print("\n방법 3 (sliding_window_view):")
    print(R3)
except:
    print("sliding_window_view는 NumPy 1.20.0 이상에서만 사용 가능합니다.")