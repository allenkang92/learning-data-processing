# 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?

import numpy as np

# 원본 벡터
Z = np.array([1, 2, 3, 4, 5])
print("원본 벡터:", Z)

# 방법 1: 반복문과 리스트 컴프리헨션
result = []
for i in range(len(Z)):
    result.append(Z[i])
    if i < len(Z) - 1:  # 마지막 요소가 아니면 0 추가
        result.extend([0, 0, 0])
result1 = np.array(result)
print("방법 1 결과:", result1)

# 방법 2: np.zeros와 인덱싱 사용
# 각 원소 사이에 3개의 0이 필요하므로 총 길이는 원본 길이 + (원본 길이-1)*3
n = len(Z)
result2 = np.zeros(n + (n-1)*3)
result2[::4] = Z  # 0, 4, 8, 12, 16 위치에 원본 값 삽입
print("방법 2 결과:", result2)

# 방법 3: np.repeat과 np.zeros_like 조합
# 각 요소를 [x,0,0,0] 패턴으로 반복하고 마지막 요소는 0 추가 없이
nz = 3  # 삽입할 연속 0의 개수
z = np.zeros(len(Z) + (len(Z)-1)*nz)
z[::nz+1] = Z
print("방법 3 결과:", z)
