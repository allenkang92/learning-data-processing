# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances

import numpy as np

# 랜덤 좌표 생성 (100개의 2D 점)
Z = np.random.random((100, 2))

# 각 점 사이의 거리 계산 방법 1: 반복문 사용 (비효율적)
distances = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        # 유클리드 거리: sqrt((x2-x1)^2 + (y2-y1)^2)
        distances[i, j] = np.sqrt(((Z[i] - Z[j])**2).sum())

print("거리 행렬 일부:")
print(distances[:5, :5])

# 방법 2: 벡터화된 연산 (효율적)
# 각 좌표 쌍 간의 차이 계산
from scipy.spatial.distance import cdist
distances_vectorized = cdist(Z, Z)
print("\n벡터화된 연산 결과:")
print(distances_vectorized[:5, :5])