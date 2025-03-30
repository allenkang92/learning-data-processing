# 51. Create a structured array representing a position (x,y) and a color (r,g,b)

import numpy as np

# 위치와 색상을 나타내는 구조화된 데이터 타입 정의
dtype = [('position', [('x', float), ('y', float)]), 
         ('color', [('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])]

# 구조화된 배열 생성
Z = np.array([
    ((0, 0), (255, 0, 0)),      # 빨간색 점 (0,0)
    ((1, 1), (0, 255, 0)),      # 녹색 점 (1,1)
    ((2, 2), (0, 0, 255))       # 파란색 점 (2,2)
], dtype=dtype)

print(Z['position'])
print(Z['color'])