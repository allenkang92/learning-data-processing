# 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])

import numpy as np
import time

# 문제: 10x3 행렬에서 값이 서로 다른(고유한) 행만 추출하기
# 예: [2,2,3]과 같이 중복된 값이 있는 행은 제외

# 테스트 행렬 생성
np.random.seed(42)  # 재현성을 위한 시드 설정
Z = np.random.randint(0, 5, (10, 3))  # 0-4 범위의 값을 가진 10x3 행렬
print("원본 행렬:")
print(Z)

# 방법 1: np.unique 사용하여 각 행의 고유한 값 개수 확인
print("\n방법 1: np.unique 사용하여 각 행의 고유한 값 개수 확인")
start = time.time()
# 각 행마다 고유한 값의 개수가 행의 크기와 같은지 확인
unique_counts = np.array([len(np.unique(row)) for row in Z])
result1 = Z[unique_counts == Z.shape[1]]  # 고유 값 개수가 열 수와 같은 행만 선택
time1 = time.time() - start

print("고유 값 개수 벡터:", unique_counts)
print("결과 행렬:")
print(result1)
print(f"실행 시간: {time1:.8f}초")

# 방법 2: set의 길이 사용
print("\n방법 2: 집합(set)의 길이 사용")
start = time.time()
result2 = np.array([row for row in Z if len(set(row)) == Z.shape[1]])
time2 = time.time() - start

print("결과 행렬:")
print(result2)
print(f"실행 시간: {time2:.8f}초")

# 방법 3: 각 행에서 최댓값과 최솟값의 차이 및 중간값 확인
print("\n방법 3: 정렬 후 차이 확인")
start = time.time()
# 방법 3a: 3열 행렬에 특화된 방법
# 각 행을 정렬하고, 인접한 값들 간의 차이가 모두 0보다 큰지 확인
result3 = []
for row in Z:
    sorted_row = np.sort(row)
    if sorted_row[1] - sorted_row[0] > 0 and sorted_row[2] - sorted_row[1] > 0:
        result3.append(row)
result3 = np.array(result3)
time3 = time.time() - start

print("결과 행렬 (정렬 후 차이 확인):")
print(result3)
print(f"실행 시간: {time3:.8f}초")

# 방법 4: 비교 마스크 사용 (행렬 각 행의 모든 요소 쌍 비교)
print("\n방법 4: 비교 마스크 사용")
start = time.time()
# 각 행에서 모든 가능한 쌍(pair)이 서로 다른지 확인
# 3열 행렬의 경우 (0,1), (0,2), (1,2) 이렇게 3쌍이 있음
result4 = []
for row in Z:
    # 모든 가능한 쌍에 대해 비교
    if row[0] != row[1] and row[0] != row[2] and row[1] != row[2]:
        result4.append(row)
result4 = np.array(result4)
time4 = time.time() - start

print("결과 행렬 (비교 마스크):")
print(result4)
print(f"실행 시간: {time4:.8f}초")

# 모든 방법의 결과 비교
print("\n모든 방법의 결과가 일치하는지 확인:")
methods_agree = True
if len(result1) > 0 and len(result2) > 0:
    methods_agree = np.array_equal(result1, result2)
if len(result2) > 0 and len(result3) > 0 and methods_agree:
    # 정렬이 필요할 수 있음
    methods_agree = (
        len(result2) == len(result3) and 
        all(any(np.array_equal(r2, r3) for r3 in result3) for r2 in result2)
    )
if len(result3) > 0 and len(result4) > 0 and methods_agree:
    methods_agree = (
        len(result3) == len(result4) and 
        all(any(np.array_equal(r3, r4) for r4 in result4) for r3 in result3)
    )
print(f"모든 방법의 결과 일치: {methods_agree}")

# 성능 비교
print("\n성능 비교:")
print("=" * 60)
print(f"{'방법':<30} | {'실행 시간(초)':<15} | {'상대 속도':<10} |")
print("-" * 60)
print(f"{'1. np.unique 사용':<30} | {time1:<15.8f} | {1.0:<10.2f} |")
print(f"{'2. set 길이 사용':<30} | {time2:<15.8f} | {time1/time2:<10.2f} |")
print(f"{'3. 정렬 후 차이 확인':<30} | {time3:<15.8f} | {time1/time3:<10.2f} |")
print(f"{'4. 비교 마스크 사용':<30} | {time4:<15.8f} | {time1/time4:<10.2f} |")
print("=" * 60)

# 일반화된 함수: n-차원 행렬에서 동작
print("\n일반화된 함수 (N차원 행렬에서 동작):")
def extract_rows_with_unique_values(matrix):
    """
    행렬에서 행 내 모든 값이 서로 다른 행만 추출하는 함수
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        입력 행렬
        
    Returns:
    --------
    result : numpy.ndarray
        모든 열이 서로 다른 값을 가진 행들만 포함한 행렬
    """
    # 각 행의 고유값 개수 계산
    unique_counts = np.array([len(np.unique(row)) for row in matrix])
    # 고유 값 개수가 열 수와 같은 행만 선택
    return matrix[unique_counts == matrix.shape[1]]

# 더 큰 행렬에서 테스트
large_matrix = np.random.randint(0, 10, (1000, 5))
start = time.time()
result_large = extract_rows_with_unique_values(large_matrix)
time_large = time.time() - start

print(f"1000x5 행렬에서 고유값을 가진 행 수: {len(result_large)}")
print(f"처리 시간: {time_large:.6f}초")

# 응용 사례: 데이터 필터링 및 데이터 품질 검사
print("\n응용 사례: 학생 점수 데이터에서 중복 평가가 없는 레코드 식별")
# 학생 점수 데이터: 각 행은 [수학, 영어, 과학] 점수를 나타냄
scores = np.array([
    [85, 92, 78],  # 모든 과목 점수가 다름
    [75, 75, 80],  # 수학=영어, 과학은 다름
    [90, 85, 90],  # 수학=과학, 영어는 다름
    [70, 80, 70],  # 수학=과학, 영어는 다름
    [85, 85, 85],  # 모든 과목 점수가 같음
    [95, 88, 92],  # 모든 과목 점수가 다름
    [82, 78, 85]   # 모든 과목 점수가 다름
])

# 서로 다른 점수를 가진 학생 식별
unique_scores = extract_rows_with_unique_values(scores)
print("모든 과목의 점수가 서로 다른 학생 점수:")
print(unique_scores)

# 고려사항 및 주의점

# 1. NaN 값이 있는 경우 np.unique는 NaN을 고유 값으로 처리하지만, 여러 NaN은 서로 같은 값으로 간주하지 않음
# 2. 부동소수점 비교 시 작은 차이로 인해 동일 값이 다르게 인식될 수 있으므로 np.isclose() 고려 필요
# 3. 대용량 데이터에서는 메모리 효율성을 위해 행별로 처리하는 것이 좋음
# 4. 행렬 크기에 따라 최적의 방법이 달라질 수 있으므로 상황에 맞는 접근법 선택이 중요