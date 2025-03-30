# 10. Find indices of non-zero elements from [1,2,0,0,4,0]

import numpy as np

arr = np.array([1, 2, 0, 0, 4, 0])

# 첫 번째 시도,
# 이 방법은 틀렸다. 인덱스를 찾아오지 못했다.
# print(arr[arr != 0])
# [1 2 4]

# 인덱스를 출력.
print(np.nonzero(arr)[0])
# [0 1 4]