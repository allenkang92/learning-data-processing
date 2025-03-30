# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

import numpy as np

# 랜덤 RGB 이미지 생성 (32x32 크기, 0-255 사이의 uint8 값)
w, h = 16, 16
image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
print(f"이미지 크기: {image.shape}")

# 방법 1: 픽셀을 튜플로 변환하여 고유한 색상 수 계산
colors = set()
for i in range(h):
    for j in range(w):
        colors.add(tuple(image[i, j]))
print(f"방법 1 - 고유한 색상 수: {len(colors)}")

# 방법 2: 배열 재구조화 및 NumPy의 unique 함수 사용
colors_array = image.reshape(-1, 3)
unique_colors = np.unique(colors_array, axis=0)
print(f"방법 2 - 고유한 색상 수: {len(unique_colors)}")

# 방법 3: 각 색상 채널을 합성하여 고유 값 계산
# RGB 값을 단일 정수로 인코딩: r*(256²) + g*256 + b
colors_hash = image[:,:,0]*65536 + image[:,:,1]*256 + image[:,:,2]
n_colors = len(np.unique(colors_hash))
print(f"방법 3 - 고유한 색상 수: {n_colors}")
