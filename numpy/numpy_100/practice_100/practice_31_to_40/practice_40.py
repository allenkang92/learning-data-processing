# 40. Create a random vector of size 10 and sort it


import numpy as np

arr = np.random.randn(10)
print(arr)
# [ 1.65794787  0.4926889   0.50445631 -0.19382926 -0.38997541 -0.54775886
#   0.41831234 -0.59842404  2.23623318  1.27726749]

arr.sort()
print(arr)
# [-0.59842404 -0.54775886 -0.38997541 -0.19382926  0.41831234  0.4926889
#   0.50445631  1.27726749  1.65794787  2.23623318]

# 또는 새로운 배열 반환
# sorted_arr = np.sort(arr)
# print(sorted_arr)