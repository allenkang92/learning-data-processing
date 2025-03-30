# 35. How to compute ((A+B)*(-A/2)) in place (without copy)?

import numpy as np

A = np.ones(3)
print(A)
# [1. 1. 1.]

B = np.ones(3)
print(B)
# [1. 1. 1.]

# 현재 코드 - 제자리 연산이 아니었음
# print(np.add(A, B) * np.divide((-1 * A), 2))

# 제자리 연산을 위한 코드
np.add(A, B, out=A)    # A = A + B
np.divide(A, -2, out=A)  # A = -A/2
np.multiply(A, B, out=A)  # A = A * B

print(A)  
# [-1. -1. -1.]