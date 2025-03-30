# 43. Make an array immutable (read-only)

import numpy as np

# # 배열 생성
# arr = np.zeros(10)
# print("원본 배열:", arr)

# # 배열을 수정 불가능하게 만들기
# arr.flags.writeable = False
# print("쓰기 가능 여부:", arr.flags.writeable)

# try:
#     arr[0] = 1
# except ValueError as e:
#     print("오류 발생:", e)