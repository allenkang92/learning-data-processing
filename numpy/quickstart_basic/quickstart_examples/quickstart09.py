

import numpy as np

print(np.arange(10000))
# [   0    1    2 ... 9997 9998 9999]

print(np.arange(10000).reshape(100, 100))
# [[   0    1    2 ...   97   98   99]
#  [ 100  101  102 ...  197  198  199]
#  [ 200  201  202 ...  297  298  299]
#  ...
#  [9700 9701 9702 ... 9797 9798 9799]
#  [9800 9801 9802 ... 9897 9898 9899]
#  [9900 9901 9902 ... 9997 9998 9999]]


# import sys
# np.set_printoptions(threshold = sys.maxsize)
# sys module should be imported