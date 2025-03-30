# 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? 

import numpy as np

# Example bincount 배열 (각 인덱스 값이 몇 번 등장하는지 나타냄)
# 예: C[0]=1은 값 0이 1번, C[1]=2는 값 1이 2번, C[2]=3은 값 2가 3번
C = np.array([1, 2, 3, 0, 2])  # 정렬된 bincount 배열
print("Bincount 배열 C:", C)

# 각 인덱스를 해당 빈도수만큼 반복하여 A 생성
# 방법 1: 반복문 사용
A1 = []
for i in range(len(C)):
    A1.extend([i] * C[i])  # 인덱스 i를 C[i]번 반복
A1 = np.array(A1)
print("\n방법 1 - 반복문 결과 A1:", A1)
print("A1의 bincount:", np.bincount(A1))
print("원본 C와 일치:", np.array_equal(np.bincount(A1), C))

# 방법 2: NumPy의 repeat 함수 사용 (더 효율적)
A2 = np.repeat(np.arange(len(C)), C)
print("\n방법 2 - np.repeat 결과 A2:", A2)
print("A2의 bincount:", np.bincount(A2))
print("원본 C와 일치:", np.array_equal(np.bincount(A2), C))
