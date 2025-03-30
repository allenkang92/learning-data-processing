# 83. How to find the most frequent value in an array?

import numpy as np

# 빈도 계산에 사용할 임의의 배열 생성
arr = np.random.randint(0, 10, size=50)  # 0-9 사이의 숫자 50개
print("원본 배열:")
print(arr)

# 방법 1: np.bincount 사용 (정수 배열에만 적용 가능)
print("\n방법 1: np.bincount 사용")
counts = np.bincount(arr)
print("각 값의 빈도:", counts)
most_frequent = np.argmax(counts)
print(f"가장 빈번한 값: {most_frequent}, 빈도수: {counts[most_frequent]}")

# 방법 2: np.unique 사용 (정수가 아닌 배열에도 적용 가능)
print("\n방법 2: np.unique 사용")
values, counts = np.unique(arr, return_counts=True)
print("고유 값:", values)
print("각 값의 빈도:", counts)
idx = np.argmax(counts)
print(f"가장 빈번한 값: {values[idx]}, 빈도수: {counts[idx]}")

# 방법 3: Counter 클래스 사용 (파이썬 표준 라이브러리)
from collections import Counter
print("\n방법 3: collections.Counter 사용")
counter = Counter(arr)
most_common = counter.most_common(1)[0]  # 가장 흔한 요소 1개 반환
print(f"가장 빈번한 값: {most_common[0]}, 빈도수: {most_common[1]}")

# 여러 값이 동일한 최대 빈도를 가질 경우
print("\n여러 값이 동일한 최대 빈도를 가질 경우:")
# 인위적으로 동일한 빈도를 가진 값이 있는 배열 생성
arr2 = np.array([1, 1, 2, 2, 3, 4, 5])
values, counts = np.unique(arr2, return_counts=True)
max_count = np.max(counts)
max_freq_values = values[counts == max_count]
print(f"동일한 최대 빈도({max_count})를 가진 값들: {max_freq_values}")

# 실제 예제: 이미지에서 가장 흔한 픽셀 값 찾기
print("\n실제 예제: 이미지에서 가장 흔한 픽셀 값 찾기")
# 간단한 그레이스케일 이미지 시뮬레이션
img = np.random.randint(0, 256, size=(10, 10))
print("이미지 배열 (10x10):")
print(img)

# 이미지의 가장 흔한 픽셀 값 찾기
most_common_pixel = np.argmax(np.bincount(img.flatten()))
print(f"가장 흔한 픽셀 값: {most_common_pixel}")

# 고급: 다차원 구조화된 배열에서의 빈도 계산
print("\n고급: 구조화된 배열에서 빈도 계산")
# 구조화된 배열 생성 (x, y 좌표 쌍)
dtype = [('x', float), ('y', float)]
structured_arr = np.array([
    (1.0, 2.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (1.0, 2.0)
], dtype=dtype)
print("구조화된 배열:")
print(structured_arr)

# 구조화된 배열에서 고유값 및 빈도 계산
unique_values, indices, counts = np.unique(
    structured_arr, return_index=True, return_counts=True)
print("고유 값:", unique_values)
print("빈도:", counts)
idx = np.argmax(counts)
print(f"가장 빈번한 값: {unique_values[idx]}, 빈도수: {counts[idx]}")