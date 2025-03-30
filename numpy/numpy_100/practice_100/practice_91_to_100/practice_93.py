# 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?

import numpy as np

# 문제: 크기가 (8,3)인 A와 (2,2)인 B 두 배열이 있을 때,
# B의 각 행에 있는 요소가 순서에 관계없이 A의 행에 모두 포함되어 있는지 찾는 방법

# 테스트 배열 생성
np.random.seed(123)  # 재현 가능한 결과를 위한 시드 설정
A = np.random.randint(0, 10, (8, 3))  # 0-9 범위의 정수를 가진 8x3 배열
B = np.random.randint(0, 10, (2, 2))  # 0-9 범위의 정수를 가진 2x2 배열

print("배열 A (8x3):")
print(A)
print("\n배열 B (2x2):")
print(B)

# 방법 1: 반복문과 np.in1d 사용
print("\n방법 1: 반복문과 np.in1d 사용")
result1 = []
for i, row_A in enumerate(A):
    # A의 각 행이 B의 각 행의 모든 요소를 포함하는지 확인
    contains_all_rows = True
    for row_B in B:
        # B의 행에 있는 모든 요소가 A의 행에 있는지 확인
        if not np.all(np.in1d(row_B, row_A)):
            contains_all_rows = False
            break
    if contains_all_rows:
        result1.append(i)

print("B의 모든 행의 요소를 포함하는 A의 행 인덱스:", result1)
if result1:
    print("해당 행:")
    for idx in result1:
        print(f"A[{idx}] =", A[idx])

# 방법 2: 집합(set) 연산 사용
print("\n방법 2: 집합(set) 연산 사용")
result2 = []
for i, row_A in enumerate(A):
    contains_all_rows = True
    set_A = set(row_A)
    for row_B in B:
        set_B = set(row_B)
        # B의 행의 모든 요소가 A의 행에 있는지 확인
        if not set_B.issubset(set_A):
            contains_all_rows = False
            break
    if contains_all_rows:
        result2.append(i)

print("B의 모든 행의 요소를 포함하는 A의 행 인덱스:", result2)
if result2:
    print("해당 행:")
    for idx in result2:
        print(f"A[{idx}] =", A[idx])

# 방법 3: 벡터화된 연산 사용
print("\n방법 3: 벡터화된 연산 사용")
def contains_all(row_A, B):
    # A의 행이 B의 모든 행의 요소를 포함하는지 확인하는 함수
    mask = np.zeros(B.shape[0], dtype=bool)
    for i, row_B in enumerate(B):
        # A의 행에서 B의 행의 각 요소를 찾기
        mask[i] = np.all(np.isin(row_B, row_A))
    return np.all(mask)

# 모든 A의 행에 대해 함수 적용
result3 = [i for i, row_A in enumerate(A) if contains_all(row_A, B)]

print("B의 모든 행의 요소를 포함하는 A의 행 인덱스:", result3)
if result3:
    print("해당 행:")
    for idx in result3:
        print(f"A[{idx}] =", A[idx])

# 추가 예제: 커스텀 배열로 확인
print("\n추가 예제: 커스텀 배열로 검증")
# 확실히 포함 관계를 보여주는 예제 배열
A_custom = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 3, 5],
    [2, 4, 6],
    [1, 3, 6],
    [2, 5, 9],
    [7, 8, 9],
    [3, 6, 9]
])

B_custom = np.array([
    [1, 3],
    [6, 9]
])

print("커스텀 배열 A:")
print(A_custom)
print("\n커스텀 배열 B:")
print(B_custom)

# 방법 3을 사용한 결과 계산
result_custom = [i for i, row_A in enumerate(A_custom) if contains_all(row_A, B_custom)]

print("\nB의 모든 행의 요소를 포함하는 A의 행 인덱스:", result_custom)
if result_custom:
    print("해당 행:")
    for idx in result_custom:
        print(f"A[{idx}] =", A_custom[idx])
    
    # 검증: 왜 이 행들이 선택되었는지 설명
    for idx in result_custom:
        row = A_custom[idx]
        print(f"\nA[{idx}] = {row} 분석:")
        print(f"B[0] = {B_custom[0]}의 요소가 모두 {row}에 포함됨? {np.all(np.isin(B_custom[0], row))}")
        print(f"B[1] = {B_custom[1]}의 요소가 모두 {row}에 포함됨? {np.all(np.isin(B_custom[1], row))}")


# 1. 대규모 배열에서는 벡터화된 연산(방법 3)이 일반적으로 가장 효율적
# 2. 작은 배열에서는 집합 연산(방법 2)이 이해하기 쉽고 효율적일 수 있음
# 3. np.isin() 또는 np.in1d()는 벡터화된 멤버십 테스트를 제공하여 성능 향상

# 일반화된 함수
print("\n일반화된 함수:")
def find_rows_containing_subarrays(A, B):
    """
    A의 행 중에서 B의 모든 행의 요소를 각각 포함하는 행을 찾는 함수
    
    Parameters:
    -----------
    A : numpy.ndarray
        검색 대상 배열
    B : numpy.ndarray
        A에서 찾을 요소가 있는 배열
        
    Returns:
    --------
    indices : list
        B의 모든 행의 요소를 각각 포함하는 A의 행 인덱스 리스트
    """
    result = []
    for i, row_A in enumerate(A):
        contains_all_rows = True
        for row_B in B:
            if not np.all(np.isin(row_B, row_A)):
                contains_all_rows = False
                break
        if contains_all_rows:
            result.append(i)
    return result

# 테스트
indices = find_rows_containing_subarrays(A, B)
print(f"일반화된 함수 결과: {indices}")

# 주의사항
# 1. A와 B의 데이터 타입이 같아야 정확한 비교 가능
# 2. NaN 값이 있는 경우 np.isin()이 제대로 동작하지 않을 수 있음
# 3. 매우 큰 배열에서는 메모리 사용량 고려 필요