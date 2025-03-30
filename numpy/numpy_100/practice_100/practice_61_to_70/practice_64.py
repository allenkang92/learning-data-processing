# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?

import numpy as np

# 원본 벡터
Z = np.ones(10)
print("원본 벡터:", Z)

# 인덱스 벡터 (중복된 인덱스 포함)
indices = np.array([0, 1, 2, 0, 3, 3, 4])
print("인덱스 벡터:", indices)

# 방법 1: 단순 인덱싱 (중복 인덱스가 중복 더해짐)
Z1 = Z.copy()
Z1[indices] += 1
print("방법 1 (단순 인덱싱):", Z1)  # 인덱스 0과 3은 2번씩 더해짐

# 방법 2: np.add.at 사용 (중복 인덱스를 올바르게 처리)
Z2 = Z.copy()
np.add.at(Z2, indices, 1)
print("방법 2 (np.add.at):", Z2)  # 중복된 인덱스에 올바르게 더함
