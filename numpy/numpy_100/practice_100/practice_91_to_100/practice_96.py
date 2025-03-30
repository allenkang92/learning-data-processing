# 96. Given a two dimensional array, how to extract unique rows? 

import numpy as np
import time
import pandas as pd

# 문제: 2차원 배열에서 고유한 행을 추출하는 방법

# 테스트 배열 생성
np.random.seed(42)
Z = np.random.randint(0, 5, (10, 3))  # 중복 행을 포함할 가능성이 있는 10x3 배열
# 의도적으로 중복 행 추가
Z[5] = Z[0]  # 첫 번째 행을 복제
Z[7] = Z[2]  # 세 번째 행을 복제

# 방법 1: np.unique 함수 사용
start = time.time()
# axis=0 옵션으로 행 방향으로 고유한 행 추출
unique_rows1 = np.unique(Z, axis=0)
time1 = time.time() - start

# 방법 2: set과 튜플 변환 사용
start = time.time()
unique_rows2 = np.array(list({tuple(row) for row in Z}))
time2 = time.time() - start

# 방법 3: Pandas의 drop_duplicates 메서드 사용
start = time.time()
unique_rows3 = pd.DataFrame(Z).drop_duplicates().values
time3 = time.time() - start

# 방법 4: 마스크 사용 (첫 번째 발생 인덱스 찾기)
start = time.time()
# 각 행에 대해 동일한 행의 첫 번째 발생 인덱스 찾기
mask = np.array([np.where(np.all(Z == row, axis=1))[0][0] for row in Z]) == np.arange(len(Z))
unique_rows4 = Z[mask]
time4 = time.time() - start

# 방법 5: 루프와 리스트 컴프리헨션
start = time.time()
seen = set()
unique_rows5 = np.array([x for x in [tuple(row) for row in Z] if not (x in seen or seen.add(x))])
time5 = time.time() - start

# 성능 비교
print("=" * 60)
print(f"방법 1 (np.unique)         : {time1:.8f}초 (기준)")
print(f"방법 2 (set/튜플)           : {time2:.8f}초 ({time1/time2:.2f}x)")
print(f"방법 3 (pandas)           : {time3:.8f}초 ({time1/time3:.2f}x)")
print(f"방법 4 (마스크)             : {time4:.8f}초 ({time1/time4:.2f}x)")
print(f"방법 5 (루프/컴프리헨션)      : {time5:.8f}초 ({time1/time5:.2f}x)")
print("=" * 60)

# 결과 검증 (모든 방법이 동일한 결과를 제공하는지)
# 각 결과 배열을 정렬하여 비교
def sort_array(arr):
    """행별로 정렬된 배열 반환"""
    return arr[np.lexsort(np.transpose(arr)[::-1])]

result1 = np.array_equal(sort_array(unique_rows1), sort_array(unique_rows2))
result2 = np.array_equal(sort_array(unique_rows2), sort_array(unique_rows3))
result3 = np.array_equal(sort_array(unique_rows3), sort_array(unique_rows4))
result4 = np.array_equal(sort_array(unique_rows4), sort_array(unique_rows5))

# 실제 사용 사례 예제: 로그 데이터 중복 제거
# 가상의 로그 데이터 (IP, 이벤트 코드, 사용자 ID)
np.random.seed(42)
log_data = np.random.randint(0, 5, (20, 3))
# 의도적으로 중복 로그 추가
log_data[10:15] = log_data[0:5]  # 처음 5개 로그를 복제

unique_logs = np.unique(log_data, axis=0)

# 일반화된 함수
def get_unique_rows(array, method='numpy', preserve_order=False):
    """
    2차원 배열에서 고유한 행을 추출하는 함수
    
    Parameters:
    -----------
    array : numpy.ndarray
        고유한 행을 추출할 2차원 배열
    method : str, 선택 사항
        사용할 방법 ('numpy', 'set', 'pandas', 'mask', 'loop' 중 하나)
    preserve_order : bool, 선택 사항
        True면 원본 순서 유지, False면 정렬된 결과 반환
        
    Returns:
    --------
    numpy.ndarray
        고유한 행만 포함하는 배열
    """
    if method == 'numpy':
        # np.unique 사용
        if preserve_order:
            # 순서 유지를 위해 인덱스 반환 후 원본 순서로 정렬
            unique_rows, idx = np.unique(array, axis=0, return_index=True)
            return array[np.sort(idx)]
        else:
            return np.unique(array, axis=0)
    
    elif method == 'set':
        # set과 튜플 변환 사용
        unique_tuples = list({tuple(row) for row in array})
        if preserve_order:
            # 원본 순서 유지
            return np.array([row for row in array if tuple(row) in unique_tuples and unique_tuples.remove(tuple(row)) is None])
        else:
            return np.array(unique_tuples)
    
    elif method == 'pandas':
        # pandas 사용
        if preserve_order:
            return pd.DataFrame(array).drop_duplicates(keep='first').values
        else:
            return pd.DataFrame(array).drop_duplicates().values
    
    elif method == 'mask':
        # 마스크 방법
        if preserve_order:
            # 첫 번째 발생 인덱스 찾기
            first_idx = np.array([np.where(np.all(array == row, axis=1))[0][0] for row in array])
            mask = first_idx == np.arange(len(array))
            return array[mask]
        else:
            # 정렬된 결과 원하는 경우, np.unique 사용
            return np.unique(array, axis=0)
    
    elif method == 'loop':
        # 루프 방법 (항상 원본 순서 유지)
        seen = set()
        if preserve_order:
            return np.array([x for x in [tuple(row) for row in array] if not (x in seen or seen.add(x))])
        else:
            # 정렬된 결과 원하는 경우, set 사용 후 변환
            return np.array(list({tuple(row) for row in array}))
    
    else:
        raise ValueError(f"알 수 없는 방법: {method}. 'numpy', 'set', 'pandas', 'mask', 'loop' 중 하나를 사용하세요.")

# 사용 예제
test_array = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]])

result1 = get_unique_rows(test_array)
result2 = get_unique_rows(test_array, preserve_order=True)
result3 = get_unique_rows(test_array, method='pandas', preserve_order=True)