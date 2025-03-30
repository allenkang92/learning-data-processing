# 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

import numpy as np

# 문제: 16x16 배열에서 4x4 블록의 합을 계산하는 방법

# 16x16 크기의 랜덤 배열 생성
Z = np.random.randint(0, 10, (16, 16))
print("원본 16x16 배열:")
print(Z)

# 방법 1: 반복문 사용
print("\n방법 1: 반복문 사용")
block_size = 4
result1 = np.zeros((4, 4))  # 4x4 블록의 합을 저장할 배열

for i in range(4):  # 블록 행 인덱스
    for j in range(4):  # 블록 열 인덱스
        # 현재 4x4 블록 추출
        block = Z[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        # 블록의 합을 계산하여 결과 배열에 저장
        result1[i, j] = np.sum(block)

print("4x4 블록 합 결과 (반복문):")
print(result1)

# 방법 2: reshape과 sum 조합 사용
print("\n방법 2: reshape과 sum 조합 사용")
# 배열을 (4, 4, 4, 4) 형태로 재구성
# 첫 번째 차원은 행 방향 블록 인덱스 (0~3)
# 두 번째 차원은 열 방향 블록 인덱스 (0~3)
# 세 번째 차원은 블록 내 행 인덱스 (0~3)
# 네 번째 차원은 블록 내 열 인덱스 (0~3)
reshaped = Z.reshape(4, 4, 4, 4)
# 블록 내 모든 요소 합산 (축 2, 3)
result2 = np.sum(reshaped, axis=(2, 3))

print("4x4 블록 합 결과 (reshape & sum):")
print(result2)

# 방법 3: NumPy의 블록 축소 함수 사용
print("\n방법 3: NumPy의 블록 축소 함수 사용")
# 블록 축소를 위한 함수 정의
def block_reduce(arr, block_size, func=np.sum):
    # 행과 열 방향으로 배열 형태 변환
    shape = (arr.shape[0] // block_size[0], block_size[0],
             arr.shape[1] // block_size[1], block_size[1])
    # 배열 재구성
    reshaped = arr.reshape(shape)
    # 지정된 함수(기본값: sum)를 사용하여 블록 축소
    return func(func(reshaped, axis=1), axis=2)

result3 = block_reduce(Z, (4, 4))
print("4x4 블록 합 결과 (block_reduce 함수):")
print(result3)

# 결과 검증
print("\n결과 검증:")
print("방법 1과 방법 2 결과 동일:", np.array_equal(result1, result2))
print("방법 2와 방법 3 결과 동일:", np.array_equal(result2, result3))

# 응용: 다양한 축소 함수 적용
print("\n응용: 다양한 축소 함수 적용")
# 평균
block_means = block_reduce(Z, (4, 4), func=np.mean)
print("4x4 블록 평균:")
print(block_means)

# 최대값
block_max = block_reduce(Z, (4, 4), func=np.max)
print("\n4x4 블록 최대값:")
print(block_max)

# 최소값
block_min = block_reduce(Z, (4, 4), func=np.min)
print("\n4x4 블록 최소값:")
print(block_min)

# 표준편차
block_std = block_reduce(Z, (4, 4), func=np.std)
print("\n4x4 블록 표준편차:")
print(block_std)