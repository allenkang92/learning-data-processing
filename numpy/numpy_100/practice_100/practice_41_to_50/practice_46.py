# 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area

import numpy as np

# 구조화된 데이터 타입 정의하기
dtype = [('x', float), ('y', float)]
Z = np.zeros((10, 10), dtype=dtype)

# [0,1]x[0,1] 영역 채우기
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)

Z['x'] = X
Z['y'] = Y

print(Z)
