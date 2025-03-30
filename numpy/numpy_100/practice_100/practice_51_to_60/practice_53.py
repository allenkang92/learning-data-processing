# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

import numpy as np

# float32 배열 생성
Z = np.arange(10, dtype=np.float32)
print("원본 float32 배열:", Z)
print("데이터 타입:", Z.dtype)

# 방법 1: 새 배열 생성 (메모리 추가 사용)
Z_new = Z.astype(np.int32)
print("\n새 배열 생성 방식:")
print("변환된 int32 배열:", Z_new)
print("데이터 타입:", Z_new.dtype)

# 방법 2: 제자리 변환 (view 사용)
Z2 = np.arange(10, dtype=np.float32)
Z2.dtype = np.int32  # 실제로는 값이 손상될 수 있음
print("\n제자리 변환 방식 (view 사용):")
print("변환된 int32 배열:", Z2)
print("데이터 타입:", Z2.dtype)