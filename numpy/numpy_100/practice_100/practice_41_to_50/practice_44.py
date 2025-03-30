# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates

import numpy as np

# 10x2 랜덤 행렬 생성 (x, y 좌표 쌍)
Z = np.random.random((10, 2))
print("직교 좌표 (x, y):\n", Z)

# 극좌표로 변환: (r, theta)
# r = sqrt(x^2 + y^2)
# theta = arctan2(y, x)
x, y = Z[:, 0], Z[:, 1]
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# 결과를 합쳐서 극좌표 (r, theta) 쌍 생성
polar = np.column_stack((r, theta))
print("극좌표 (r, theta):\n", polar)