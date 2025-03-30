# 59. How to sort an array by the nth column?

import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr[:, 2].sort()
print(arr)



n = 1  # 1번째 열로 정렬 (0부터 시작)
sorted_arr = arr[arr[:, n].argsort()]
print("\n1번째 열로 정렬된 배열:")
print(sorted_arr)

# 내림차순 정렬
sorted_desc = arr[arr[:, n].argsort()[::-1]]
print("\n1번째 열로 내림차순 정렬된 배열:")
print(sorted_desc)