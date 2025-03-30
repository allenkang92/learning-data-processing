# 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

import numpy as np

# 10x10 랜덤 행렬 생성
Z = np.random.randint(0, 10, size=(10, 10))
print("원본 10x10 랜덤 행렬:")
print(Z)

# 방법 1: 반복문 사용
print("\n방법 1: 반복문 사용하여 3x3 블록 추출")
blocks_loop = []
for i in range(8):  # 10-3+1=8개의 행 시작점
    for j in range(8):  # 10-3+1=8개의 열 시작점
        block = Z[i:i+3, j:j+3]
        blocks_loop.append(block)
        
print(f"추출된 블록 수: {len(blocks_loop)}")
print("첫 번째 블록:")
print(blocks_loop[0])
print("마지막 블록:")
print(blocks_loop[-1])

# 방법 2: numpy.lib.stride_tricks.as_strided 사용
from numpy.lib.stride_tricks import as_strided

print("\n방법 2: as_strided 사용하여 3x3 블록 추출")
block_shape = (3, 3)
window_shape = (10-3+1, 10-3+1) + block_shape  # (8, 8, 3, 3)

# as_strided를 사용하여 뷰 생성 (메모리 효율적)
blocks_strided = as_strided(Z,
                           shape=window_shape,
                           strides=Z.strides + Z.strides)

# reshape하여 블록 배열로 변환
blocks_reshaped = blocks_strided.reshape(-1, *block_shape)  # (64, 3, 3)

print(f"추출된 블록 수: {len(blocks_reshaped)}")
print("첫 번째 블록:")
print(blocks_reshaped[0])
print("마지막 블록:")
print(blocks_reshaped[-1])

# 방법 3: NumPy 1.20 이상에서 sliding_window_view 사용
try:
    from numpy.lib.stride_tricks import sliding_window_view
    print("\n방법 3: sliding_window_view 사용하여 3x3 블록 추출")
    blocks_window = sliding_window_view(Z, (3, 3))
    blocks_window_reshaped = blocks_window.reshape(-1, 3, 3)
    
    print(f"추출된 블록 수: {len(blocks_window_reshaped)}")
    print("첫 번째 블록:")
    print(blocks_window_reshaped[0])
    print("마지막 블록:")
    print(blocks_window_reshaped[-1])
except ImportError:
    print("sliding_window_view는 NumPy 1.20.0 이상에서만 사용 가능합니다.")

# 결과 검증: 두 방법의 결과가 동일한지 확인
print("\n결과 검증:")
if len(blocks_loop) > 0 and len(blocks_reshaped) > 0:
    are_equal = np.all([np.array_equal(blocks_loop[i], blocks_reshaped[i]) 
                      for i in range(len(blocks_loop))])
    print(f"두 방법의 결과가 동일합니까? {are_equal}")

# 응용: 추출된 모든 3x3 블록의 평균 계산
print("\n응용: 모든 3x3 블록의 평균 계산")
block_means = np.mean(blocks_reshaped, axis=(1, 2))
print("각 블록의 평균값:", block_means[:5], "...")  # 처음 5개만 표시
print("전체 블록 평균의 평균:", np.mean(block_means))

# 응용: 가장 높은 합계를 가진 3x3 블록 찾기
print("\n응용: 가장 높은 합계를 가진 3x3 블록 찾기")
block_sums = np.sum(blocks_reshaped, axis=(1, 2))
max_idx = np.argmax(block_sums)
print("최대 합계를 가진 블록 인덱스:", max_idx)
print("최대 합계 값:", block_sums[max_idx])
print("해당 블록:")
print(blocks_reshaped[max_idx])