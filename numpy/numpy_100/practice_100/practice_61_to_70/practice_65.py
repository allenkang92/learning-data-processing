# 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?

import numpy as np

# 소스 벡터 X
X = np.random.rand(10)
print("소스 벡터 X:", X)

# 인덱스 벡터 I (0, 1, 또는 2 값 포함)
I = np.random.randint(0, 3, 10)
print("인덱스 벡터 I:", I)

# 결과 저장할 F 배열 (길이 3)
F = np.zeros(3)
print("초기 F 배열:", F)

# 방법 1: 반복문 사용
for i in range(len(X)):
    F[I[i]] += X[i]
print("방법 1 결과 F:", F)

# 방법 2: np.add.at 사용 (더 효율적)
F_vectorized = np.zeros(3)
np.add.at(F_vectorized, I, X)
print("방법 2 결과 F (벡터화):", F_vectorized)