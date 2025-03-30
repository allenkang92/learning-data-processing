# 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary)

import numpy as np

def extract_subpart(arr, center, shape, fill_value=0):
    """
    배열에서 지정된 중심 주변의 고정된 모양의 하위 배열을 추출합니다.
    필요한 경우 경계를 fill_value로 채웁니다.
    
    매개변수:
    - arr: 입력 배열 (ndarray)
    - center: 추출할 하위 배열의 중심 좌표 (tuple)
    - shape: 추출할 하위 배열의 모양 (tuple)
    - fill_value: 경계를 넘어갈 때 사용할 채움 값 (기본값: 0)
    
    반환:
    - 추출된 하위 배열 (ndarray)
    """
    # 결과 배열 초기화 (fill_value로 채워짐)
    result = np.full(shape, fill_value, dtype=arr.dtype)
    
    # 중심 주변의 인덱스 범위 계산
    ranges = []
    for i, dim in enumerate(shape):
        # 중심에서 각 차원의 시작/끝 인덱스 계산
        start = center[i] - dim // 2
        end = start + dim
        ranges.append((start, end))
    
    # 원본 배열에서 유효한 인덱스 범위 계산
    src_ranges = []
    dst_ranges = []
    
    for i, (start, end) in enumerate(ranges):
        # 원본 배열 범위 내의 시작/끝 인덱스
        src_start = max(0, start)
        src_end = min(arr.shape[i], end)
        
        # 결과 배열에서의 대응되는 인덱스
        dst_start = max(0, -start)
        dst_end = dst_start + (src_end - src_start)
        
        src_ranges.append(slice(src_start, src_end))
        dst_ranges.append(slice(dst_start, dst_end))
    
    # 결과 배열에 원본 배열의 값 복사
    result[tuple(dst_ranges)] = arr[tuple(src_ranges)]
    
    return result

# 테스트를 위한 예제
# 2D 배열 생성
arr_2d = np.arange(1, 17).reshape(4, 4)
print("원본 2D 배열:")
print(arr_2d)

# 중심 (1, 2)에서 3x3 하위 배열 추출
center = (1, 2)
shape = (3, 3)
subpart_2d = extract_subpart(arr_2d, center, shape, fill_value=-1)
print(f"\n중심 {center}에서 {shape} 모양의 하위 배열 추출:")
print(subpart_2d)

# 3D 배열 생성
arr_3d = np.arange(1, 28).reshape(3, 3, 3)
print("\n원본 3D 배열:")
print(arr_3d)

# 중심 (1, 1, 1)에서 2x2x2 하위 배열 추출
center = (1, 1, 1)
shape = (2, 2, 2)
subpart_3d = extract_subpart(arr_3d, center, shape, fill_value=-1)
print(f"\n중심 {center}에서 {shape} 모양의 하위 배열 추출:")
print(subpart_3d)
