


import numpy as np

palette = np.array([[0, 0, 0], # black
                    [255, 0, 0], # red
                    [0, 255, 0], # green
                    [0, 0, 255], # blue
                    [255, 255, 255]]) # white

image = np.array([[0, 1, 2, 0],
                  [0, 3, 4, 0]]) # each value corresponds to a color in the palette

print(palette[image]) # the (2, 4, 3) color image
# [[[  0   0   0]    # 검정
#   [255   0   0]    # 빨강
#   [  0 255   0]    # 초록
#   [  0   0   0]]   # 검정

#  [[  0   0   0]    # 검정
#   [  0   0 255]    # 파랑
#   [255 255 255]    # 흰색
#   [  0   0   0]]]  # 검정


