# 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). 
# How to compute the sum of of the p matrix products at once? (result has shape (n,1))

import numpy as np

# 문제: p개의 nxn 행렬과 p개의 nx1 벡터를 곱한 결과들의 합을 계산하기

# 예시 데이터 생성
p = 3  # 행렬과 벡터의 개수
n = 4  # 각 행렬과 벡터의 차원

# p개의 (n,n) 행렬 생성
matrices = np.random.rand(p, n, n)
print("p개의 행렬 형태:", matrices.shape)
print("첫 번째 행렬:")
print(matrices[0])

# p개의 (n,1) 벡터 생성
vectors = np.random.rand(p, n, 1)
print("\np개의 벡터 형태:", vectors.shape)
print("첫 번째 벡터:")
print(vectors[0])

# 방법 1: 반복문 사용
print("\n방법 1: 반복문 사용")
result_loop = np.zeros((n, 1))
for i in range(p):
    # i번째 행렬과 i번째 벡터의 행렬곱
    result_loop += np.dot(matrices[i], vectors[i])
    
print("반복문 결과 형태:", result_loop.shape)
print("반복문 결과:")
print(result_loop)

# 방법 2: np.einsum 사용 (아인슈타인 합 표기법)
print("\n방법 2: np.einsum 사용")
# einsum 표기법: 'ijk,ikl->jl'
# i: p개의 행렬/벡터 인덱스
# j,k: n x n 행렬의 행과 열 인덱스
# k,l: n x 1 벡터의 인덱스 (여기서 l은 항상 0이지만 차원을 명시적으로 표현)
result_einsum = np.einsum('ijk,ikl->jl', matrices, vectors)

print("einsum 결과 형태:", result_einsum.shape)
print("einsum 결과:")
print(result_einsum)

# 방법 3: 행렬 연산 사용 (batch matrix multiplication)
print("\n방법 3: 배치 행렬 곱셈 사용")
# (p,n,n) @ (p,n,1) -> (p,n,1) 형태의 배치 행렬곱
batch_result = np.matmul(matrices, vectors)
# (p,n,1) -> (n,1) 형태로 합산
result_batch = np.sum(batch_result, axis=0)

print("배치 행렬곱 결과 형태:", result_batch.shape)
print("배치 행렬곱 결과:")
print(result_batch)

# 결과 검증: 세 방법의 결과가 동일한지 확인
print("\n결과 검증:")
print("반복문과 einsum 결과 동일:", np.allclose(result_loop, result_einsum))
print("einsum과 배치 행렬곱 결과 동일:", np.allclose(result_einsum, result_batch))

# 성능 비교 (간단한 시간 측정)
import time

# 더 큰 문제 크기로 성능 테스트
p_large = 100
n_large = 50
matrices_large = np.random.rand(p_large, n_large, n_large)
vectors_large = np.random.rand(p_large, n_large, 1)

print("\n성능 비교 (p={}, n={}):".format(p_large, n_large))

# 방법 1: 반복문
start = time.time()
result1 = np.zeros((n_large, 1))
for i in range(p_large):
    result1 += np.dot(matrices_large[i], vectors_large[i])
time_loop = time.time() - start
print("반복문 시간: {:.6f}초".format(time_loop))

# 방법 2: einsum
start = time.time()
result2 = np.einsum('ijk,ikl->jl', matrices_large, vectors_large)
time_einsum = time.time() - start
print("einsum 시간: {:.6f}초".format(time_einsum))

# 방법 3: 배치 행렬곱
start = time.time()
result3 = np.sum(np.matmul(matrices_large, vectors_large), axis=0)
time_batch = time.time() - start
print("배치 행렬곱 시간: {:.6f}초".format(time_batch))

print("\n최적의 방법: ", end="")
if time_einsum <= time_loop and time_einsum <= time_batch:
    print("einsum")
elif time_batch <= time_loop and time_batch <= time_einsum:
    print("배치 행렬곱")
else:
    print("반복문")