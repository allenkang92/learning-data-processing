# 75. How to compute averages using a sliding window over an array?

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 데이터 배열 생성
data = np.arange(10)
print("원본 데이터:", data)

# 슬라이딩 윈도우 크기
window_size = 3

# 방법 1: 반복문 사용
result_manual = np.zeros(len(data) - window_size + 1)
for i in range(len(result_manual)):
    result_manual[i] = np.mean(data[i:i + window_size])
print(f"\n방법 1 (반복문) - 윈도우 크기 {window_size}:", result_manual)

# 방법 2: NumPy의 convolve 함수 사용
# 이 방법은 필터와 데이터를 컨볼루션하는 방식
window = np.ones(window_size) / window_size  # 평균을 계산하기 위한 윈도우
result_convolve = np.convolve(data, window, mode='valid')
print(f"방법 2 (np.convolve) - 윈도우 크기 {window_size}:", result_convolve)

# 방법 3: NumPy의 sliding_window_view 사용 (1.20.0 이상)
try:
    # NumPy 1.20.0 이상에서만 사용 가능
    windows = sliding_window_view(data, window_size)
    result_view = np.mean(windows, axis=1)
    print(f"방법 3 (sliding_window_view) - 윈도우 크기 {window_size}:", result_view)
except:
    print("방법 3 (sliding_window_view)는 NumPy 1.20.0 이상에서만 사용 가능합니다.")

# 방법 4: numpy의 cumsum을 이용한 효율적인 방법
cumsum = np.cumsum(data)
result_cumsum = (cumsum[window_size-1:] - np.hstack(([0], cumsum[:-window_size]))) / window_size
print(f"방법 4 (cumsum) - 윈도우 크기 {window_size}:", result_cumsum)

# 방법 5: as_strided를 사용한 뷰 기반 방법
from numpy.lib.stride_tricks import as_strided

def moving_average_strided(a, n):
    # 스트라이드 트릭을 사용한 슬라이딩 윈도우 뷰 생성
    ret = as_strided(a, shape=(len(a) - n + 1, n), strides=(a.strides[0], a.strides[0]))
    return np.mean(ret, axis=1)

result_strided = moving_average_strided(data, window_size)
print(f"방법 5 (as_strided) - 윈도우 크기 {window_size}:", result_strided)
