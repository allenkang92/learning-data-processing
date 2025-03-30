# 29. How to round away from zero a float array ?

import numpy as np

arr = np.array([1.1 , 2.5 , 3.7])
print(arr)
# [1.1 2.5 3.7]

print(np.round(arr))
# [1. 2. 4.]

Z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)), Z))
# [-9. -5.  1. -3.  3. -9.  2.  8.  1. -2.]

# More readable but less efficient
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
#[-9. -5.  1. -3.  3. -9.  2.  8.  1. -2.]
