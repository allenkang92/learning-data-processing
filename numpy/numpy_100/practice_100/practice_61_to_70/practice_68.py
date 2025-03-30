# 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices?

import numpy as np

# 원본 데이터 벡터
D = np.random.randint(0, 10, 10)
print("데이터 벡터 D:", D)

# 각 요소의 하위 집합 인덱스 (0, 1, 2 등)를 포함하는 벡터
S = np.random.randint(0, 3, 10)  # 0, 1, 2 중 하나의 값
print("하위 집합 인덱스 벡터 S:", S)

# 방법 1: 반복문 사용
result_manual = np.zeros(3)  # 3개의 하위 집합
count = np.zeros(3)

for i in range(len(D)):
    result_manual[S[i]] += D[i]
    count[S[i]] += 1

result_manual = result_manual / count
print("방법 1 (반복문) - 각 하위 집합의 평균:", result_manual)

# 방법 2: 넘파이 기능 사용 (더 효율적)
# bincount를 사용하여 각 그룹의 합과 개수 계산
sum_per_group = np.bincount(S, weights=D)
count_per_group = np.bincount(S)
result_numpy = sum_per_group / count_per_group
print("방법 2 (NumPy) - 각 하위 집합의 평균:", result_numpy)