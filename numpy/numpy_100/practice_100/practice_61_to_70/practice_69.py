# 69. How to get the diagonal of a dot product?

import numpy as np

# 두 개의 매트릭스 생성
A = np.random.randint(0, 10, (3, 3))
B = np.random.randint(0, 10, (3, 3))

print("행렬 A:")
print(A)
print("\n행렬 B:")
print(B)

# 방법 1: 행렬 곱셈 후 대각선 추출
C = np.dot(A, B)
diag1 = np.diag(C)
print("\n방법 1 - 행렬 곱셈 후 대각선 추출:")
print("A·B:", C)
print("diag(A·B):", diag1)

# 방법 2: 효율적인 방법 - einsum 사용
# 행렬 곱셈의 대각선 직접 계산 (i,j 인덱스와 j,i 인덱스 곱의 합)
diag2 = np.einsum('ij,ji->i', A, B)
print("\n방법 2 - einsum 사용 (더 효율적):")
print("diag(A·B) 직접 계산:", diag2)

# 방법 3: 벡터화된 연산 
diag3 = np.sum(A * B.T, axis=1)
print("\n방법 3 - 벡터화된 연산:")
print("diag(A·B) 벡터화:", diag3)

# 모든 방법의 결과가 동일한지 확인
print("\n모든 방법의 결과가 동일함:", np.array_equal(diag1, diag2) and np.array_equal(diag1, diag3))
