# 28. What are the result of the following expressions?

import numpy as np



# np.array(0) / np.array(0)
# -> 0
print(np.array(0) / np.array(0))
# py:7: RuntimeWarning: invalid value encountered in divide
#   print(np.array(0) / np.array(0))
# nan (0으로 나누기는 불능)



# np.array(0) // np.array(0)
# -> 0
print(np.array(0) // np.array(0))

# py:11: RuntimeWarning: divide by zero encountered in floor_divide
#   print(np.array(0) // np.array(0))
# 0


# np.array([np.nan]).astype(int).astype(float)
# -> nan
print(np.array([np.nan]).astype(int).astype(float))

# py:15: RuntimeWarning: invalid value encountered in cast
#   print(np.array([np.nan]).astype(int).astype(float))
# [-9.22337204e+18]
# np.array([np.nan]).astype(int).astype(float)의 결과는 매우 큰 음수값
# (nan을 정수로 변환할 때 발생하는 특성)