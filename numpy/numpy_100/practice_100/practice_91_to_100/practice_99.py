# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n.

import numpy as np
import time

# 문제: 정수 n과 2D 배열 X가 주어졌을 때, X에서 다항 분포의 결과로 해석될 수 있는 행들을 선택.
# 즉, 정수만을 포함하고 그 합이 n인 행들을 선택

# 예제를 위한 데이터 생성
np.random.seed(42)
n = 10  # 다항 분포의 시행 횟수
rows = 20
cols = 5
X = np.random.randint(0, 5, size=(rows, cols)).astype(float)

# 일부 행은 다항 분포에서 추출된 것처럼 수정
for i in range(5):
    # i번째 행은 합이 n이 되도록 설정
    X[i] = np.random.multinomial(n, np.ones(cols)/cols)

# 일부 행에는 실수값 추가
for i in range(5, 10):
    X[i, np.random.randint(0, cols)] += 0.5

# 일부 행의 합을 n으로 설정하되 실수 포함
for i in range(10, 15):
    total = np.sum(X[i])
    X[i] = X[i] * (n / total)

print("원본 배열 X:")
print(X)
print(f"\n배열 차원: {X.shape}")

# 방법 1: 기본 접근법 (반복문 사용)
print("\n방법 1: 기본 접근법 (반복문 사용)")
start = time.time()

valid_rows_idx1 = []
for i in range(len(X)):
    row = X[i]
    # 모든 값이 정수이고 합이 n인지 확인
    if np.all(np.equal(np.mod(row, 1), 0)) and np.sum(row) == n:
        valid_rows_idx1.append(i)

# 유효한 행 추출
valid_rows1 = X[valid_rows_idx1]

time_method1 = time.time() - start
print(f"방법 1 - 실행 시간: {time_method1:.8f}초")
print(f"방법 1 - 선택된 행 수: {len(valid_rows1)}")
print("선택된 행:")
print(valid_rows1)

# 방법 2: 벡터화된 접근법
print("\n방법 2: 벡터화된 접근법")
start = time.time()

# 각 행이 정수만 포함하는지 확인 (모듈로 연산 사용)
is_integer = np.all(np.equal(np.mod(X, 1), 0), axis=1)

# 각 행의 합이 n인지 확인
row_sums = np.sum(X, axis=1)
sum_equals_n = np.equal(row_sums, n)

# 두 조건을 모두 만족하는 행 선택
valid_rows_mask = np.logical_and(is_integer, sum_equals_n)
valid_rows2 = X[valid_rows_mask]

time_method2 = time.time() - start
print(f"방법 2 - 실행 시간: {time_method2:.8f}초")
print(f"방법 2 - 선택된 행 수: {len(valid_rows2)}")
print("선택된 행:")
print(valid_rows2)

# 방법 3: isclose 사용 (부동소수점 비교를 위한 안전한 접근법)
print("\n방법 3: np.isclose 사용 (부동소수점 비교를 위한 안전한 접근법)")
start = time.time()

# 각 값이 정수에 가까운지 확인 (부동소수점 오차 고려)
is_integer_safe = np.all(np.isclose(X, np.round(X), rtol=1e-10, atol=1e-10), axis=1)

# 각 행의 합이 n에 가까운지 확인 (부동소수점 오차 고려)
sum_equals_n_safe = np.isclose(np.sum(X, axis=1), n, rtol=1e-10, atol=1e-10)

# 두 조건을 모두 만족하는 행 선택
valid_rows_mask_safe = np.logical_and(is_integer_safe, sum_equals_n_safe)
valid_rows3 = X[valid_rows_mask_safe]

time_method3 = time.time() - start
print(f"방법 3 - 실행 시간: {time_method3:.8f}초")
print(f"방법 3 - 선택된 행 수: {len(valid_rows3)}")
print("선택된 행:")
print(valid_rows3)

# 방법 4: numba 사용한 최적화 (대용량 데이터용)
print("\n방법 4: NumPy 마스킹 연산 최적화")
start = time.time()

# 정수 확인을 위한 더 효율적인 방법
X_rounded = np.round(X)
is_integer_fast = np.all(np.abs(X - X_rounded) < 1e-10, axis=1)

# 합계 확인
sum_equals_n_fast = np.abs(np.sum(X, axis=1) - n) < 1e-10

# 조건 결합
valid_rows_mask_fast = is_integer_fast & sum_equals_n_fast
valid_rows4 = X[valid_rows_mask_fast]

time_method4 = time.time() - start
print(f"방법 4 - 실행 시간: {time_method4:.8f}초")
print(f"방법 4 - 선택된 행 수: {len(valid_rows4)}")
print("선택된 행:")
print(valid_rows4)

# 성능 비교
print("\n성능 비교:")
print("=" * 60)
print(f"{'방법':<30} | {'실행 시간(초)':<15} | {'선택된 행 수'}")
print("-" * 60)
print(f"{'1. 기본 접근법 (반복문)':<30} | {time_method1:<15.8f} | {len(valid_rows1)}")
print(f"{'2. 벡터화된 접근법':<30} | {time_method2:<15.8f} | {len(valid_rows2)}")
print(f"{'3. isclose 사용':<30} | {time_method3:<15.8f} | {len(valid_rows3)}")
print(f"{'4. 마스킹 연산 최적화':<30} | {time_method4:<15.8f} | {len(valid_rows4)}")
print("=" * 60)

# 대용량 데이터 테스트
print("\n대용량 데이터 테스트:")
n_large = 100
rows_large = 100000
cols_large = 10
X_large = np.random.randint(0, 50, size=(rows_large, cols_large)).astype(float)

# 10%의 행을 다항 분포에서 온 것처럼 설정
valid_indices = np.random.choice(rows_large, size=rows_large//10, replace=False)
for i in valid_indices:
    X_large[i] = np.random.multinomial(n_large, np.ones(cols_large)/cols_large)

print(f"대용량 배열 차원: {X_large.shape}")
print(f"예상되는 유효 행 수: {len(valid_indices)}")

# 가장 효율적인 방법 (방법 4)으로 테스트
start = time.time()

# 정수 확인
X_large_rounded = np.round(X_large)
is_integer_large = np.all(np.abs(X_large - X_large_rounded) < 1e-10, axis=1)

# 합계 확인
sum_equals_n_large = np.abs(np.sum(X_large, axis=1) - n_large) < 1e-10

# 조건 결합
valid_rows_mask_large = is_integer_large & sum_equals_n_large
valid_rows_large = X_large[valid_rows_mask_large]

time_large = time.time() - start
print(f"대용량 데이터 처리 시간: {time_large:.4f}초")
print(f"선택된 행 수: {len(valid_rows_large)}")

# 선택된 행의 특성 확인
if len(valid_rows_large) > 0:
    print(f"선택된 첫 행: {valid_rows_large[0]}")
    print(f"첫 행의 합: {np.sum(valid_rows_large[0])}")
    print(f"모든 선택된 행의 합이 {n_large}인가? {np.all(np.isclose(np.sum(valid_rows_large, axis=1), n_large))}")

# 일반화된 함수
def select_multinomial_rows(X, n, rtol=1e-10, atol=1e-10):
    """
    2D 배열 X에서 다항 분포의 결과로 해석될 수 있는 행들을 선택
    
    Parameters:
    -----------
    X : numpy.ndarray
        2D 배열
    n : int
        다항 분포의 시행 횟수 (선택될 행의 합)
    rtol : float, optional
        상대 허용 오차 (기본값: 1e-10)
    atol : float, optional
        절대 허용 오차 (기본값: 1e-10)
        
    Returns:
    --------
    valid_rows : numpy.ndarray
        조건을 만족하는, 정수만을 포함하고 합이 n인 행들
    """
    # 정수 확인 (부동소수점 오차 고려)
    X_rounded = np.round(X)
    is_integer = np.all(np.abs(X - X_rounded) < atol, axis=1)
    
    # 합계 확인
    sum_equals_n = np.abs(np.sum(X, axis=1) - n) < atol
    
    # 두 조건을 모두 만족하는 행 선택
    valid_rows_mask = is_integer & sum_equals_n
    
    return X[valid_rows_mask]

# 함수 테스트
print("\n일반화된 함수 테스트:")
test_X = np.array([
    [1, 2, 3, 4],  # 합: 10, 정수, 선택됨
    [1.5, 2.5, 3, 3],  # 합: 10, 비정수, 선택 안됨
    [2, 2, 2, 2],  # 합: 8, 정수, 선택 안됨
    [5, 0, 5, 0],  # 합: 10, 정수, 선택됨
    [0, 0, 0, 10]  # 합: 10, 정수, 선택됨
])

test_n = 10
result = select_multinomial_rows(test_X, test_n)
print(f"테스트 배열:\n{test_X}")
print(f"선택된 행 (합이 {test_n}이고 정수인 행):")
print(result)