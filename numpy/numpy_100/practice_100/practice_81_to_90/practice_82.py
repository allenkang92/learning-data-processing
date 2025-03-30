# 82. Compute a matrix rank (★★★)

import numpy as np

# 행렬 랭크(rank)는 선형적으로 독립적인 행/열의 최대 개수를 의미합니다.
# 또는 0이 아닌 특이값(singular value)의 개수로도 정의됩니다.

# 예제 행렬 생성
print("1. 랭크가 2인 행렬 (2x3):")
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(A)

print("\n2. 랭크가 1인 행렬 (첫 번째 행이 두 번째 행의 2배):")
B = np.array([[2, 4, 6], 
              [1, 2, 3]])
print(B)

print("\n3. 랭크가 0인 행렬 (모든 요소가 0):")
C = np.zeros((2, 3))
print(C)

print("\n4. 랜덤 행렬:")
D = np.random.rand(3, 3)
print(D)

# 방법 1: np.linalg.matrix_rank 함수 사용
print("\n방법 1: np.linalg.matrix_rank 함수 사용")
print("A의 랭크:", np.linalg.matrix_rank(A))
print("B의 랭크:", np.linalg.matrix_rank(B))
print("C의 랭크:", np.linalg.matrix_rank(C))
print("D의 랭크:", np.linalg.matrix_rank(D))

# 방법 2: SVD(특이값 분해)를 사용한 랭크 계산
print("\n방법 2: SVD(특이값 분해)를 사용한 랭크 계산")
def matrix_rank_svd(matrix, tol=1e-10):
    # SVD 계산
    s = np.linalg.svd(matrix, compute_uv=False)
    # 임계값보다 큰 특이값 개수 반환
    return np.sum(s > tol)

print("A의 랭크 (SVD):", matrix_rank_svd(A))
print("B의 랭크 (SVD):", matrix_rank_svd(B))
print("C의 랭크 (SVD):", matrix_rank_svd(C))
print("D의 랭크 (SVD):", matrix_rank_svd(D))

# 고유값(eigenvalues)으로도 정방행렬의 랭크를 추정할 수 있습니다.
print("\n정방행렬 D의 고유값:", np.linalg.eigvals(D))

# 응용: 랭크와 행렬 크기의 관계
print("\n행렬 크기와 랭크의 관계:")
print("- 랭크는 항상 min(행 수, 열 수) 이하")
print("- 행렬 A(2x3)의 경우, 최대 랭크는 2")
print("- 정방행렬의 경우, 랭크가 행렬 크기와 같으면 가역(invertible)행렬")

# 가역(invertible)/비가역(singular) 행렬 예시
print("\n행렬 D가 가역인가?", np.linalg.matrix_rank(D) == D.shape[0])

# 행 공간과 열 공간의 기저(basis) 구하기
# 기저는 SVD의 U와 V 행렬에서 얻을 수 있습니다.
U, s, Vt = np.linalg.svd(A)
r = np.sum(s > 1e-10)  # 랭크
print("\nA의 행 공간 기저 (U의 처음", r, "열):")
print(U[:, :r])