# 60. How to tell if a given 2D array has null columns?

import numpy as np

# null이 없는 배열
arr1 = np.array([[1, 2, 3],
                [4, 5, 6]])
print("배열 1:")
print(arr1)

# null 열을 포함하는 배열 (2번째 열이 모두 NaN)
arr2 = np.array([[1, np.nan, 3],
                [4, np.nan, 6]])
print("\n배열 2:")
print(arr2)

# 잘못된 방법 (NaN은 비교 연산으로 확인 불가)
# print(arr[:, :] == np.nan)  # 항상 False 반환

# 올바른 방법: np.isnan 사용
print("\n각 값이 NaN인지 확인 (배열 2):")
null_mask = np.isnan(arr2)
print(null_mask)

# 열 전체가 NaN인지 확인
null_columns = np.all(null_mask, axis=0)
print("\n각 열이 전체 NaN인지 여부:")
print(null_columns)

# 최소한 하나의 NaN이 있는 열 확인
has_null = np.any(null_mask, axis=0)
print("\n각 열에 NaN이 하나라도 있는지 여부:")
print(has_null)