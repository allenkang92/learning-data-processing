# 97. 두 벡터 A와 B에 대해 내적(inner product), 외적(outer product), 합(sum), 곱(mul)을 NumPy의 einsum 함수를 사용하여 구현

import numpy as np
import time

# 문제: 두 벡터 A와 B에 대해 내적(inner), 외적(outer), 합(sum), 곱(mul)의 einsum 등가식 작성

# 테스트 벡터 생성
np.random.seed(42)  # 재현성을 위한 시드 설정
A = np.random.rand(5)  # 5차원 랜덤 벡터 A
B = np.random.rand(5)  # 5차원 랜덤 벡터 B

# print("벡터 A:")
# print(A)
# print("\n벡터 B:")
# print(B)

# 1. 내적 (Inner Product) 계산
# print("\n1. 내적 (Inner Product):")
# print("-" * 50)

# 방법 1: np.inner 사용
start = time.time()
inner_result1 = np.inner(A, B)
time_inner1 = time.time() - start
# print(f"np.inner(A, B) = {inner_result1}")
# print(f"실행 시간: {time_inner1:.8f}초")

# 방법 2: einsum 사용
start = time.time()
inner_result2 = np.einsum('i,i->', A, B)  # i 인덱스에 대해 합산
time_inner2 = time.time() - start
# print(f"np.einsum('i,i->', A, B) = {inner_result2}")
# print(f"실행 시간: {time_inner2:.8f}초")

# 방법 3: 기본 연산 사용
start = time.time()
inner_result3 = np.sum(A * B)
time_inner3 = time.time() - start
# print(f"np.sum(A * B) = {inner_result3}")
# print(f"실행 시간: {time_inner3:.8f}초")

# 2. 외적 (Outer Product) 계산
# print("\n2. 외적 (Outer Product):")
# print("-" * 50)

# 방법 1: np.outer 사용
start = time.time()
outer_result1 = np.outer(A, B)
time_outer1 = time.time() - start
# print(f"np.outer(A, B) 결과 형태: {outer_result1.shape}")
# print(f"실행 시간: {time_outer1:.8f}초")

# 방법 2: einsum 사용
start = time.time()
outer_result2 = np.einsum('i,j->ij', A, B)  # 각 i, j 조합의 곱
time_outer2 = time.time() - start
# print(f"np.einsum('i,j->ij', A, B) 결과 형태: {outer_result2.shape}")
# print(f"실행 시간: {time_outer2:.8f}초")

# 방법 3: 기본 연산 사용
start = time.time()
outer_result3 = A[:, np.newaxis] * B[np.newaxis, :]
time_outer3 = time.time() - start
# print(f"A[:, np.newaxis] * B[np.newaxis, :] 결과 형태: {outer_result3.shape}")
# print(f"실행 시간: {time_outer3:.8f}초")

# 결과 비교 (outer product)
# print("\n외적 결과 비교 (첫 두 행만 표시):")
# print("np.outer:")
# print(outer_result1[:2, :2])
# print("np.einsum:")
# print(outer_result2[:2, :2])
# print("기본 연산:")
# print(outer_result3[:2, :2])

# 3. 합(Sum) 계산
# print("\n3. 합(Sum):")
# print("-" * 50)

# 방법 1: np.sum 사용
start = time.time()
sum_result1 = np.sum(A) + np.sum(B)
time_sum1 = time.time() - start
# print(f"np.sum(A) + np.sum(B) = {sum_result1}")
# print(f"실행 시간: {time_sum1:.8f}초")

# 방법 2: einsum 사용
start = time.time()
sum_result2 = np.einsum('i->', A) + np.einsum('i->', B)
time_sum2 = time.time() - start
# print(f"np.einsum('i->', A) + np.einsum('i->', B) = {sum_result2}")
# print(f"실행 시간: {time_sum2:.8f}초")

# 방법 3: 두 벡터를 연결한 후 합계 계산
start = time.time()
sum_result3 = np.einsum('i->', np.concatenate([A, B]))
time_sum3 = time.time() - start
# print(f"np.einsum('i->', np.concatenate([A, B])) = {sum_result3}")
# print(f"실행 시간: {time_sum3:.8f}초")

# 4. 요소별 곱(Element-wise Multiplication) 계산
# print("\n4. 요소별 곱(Element-wise Multiplication):")
# print("-" * 50)

# 방법 1: 기본 연산자 사용
start = time.time()
mul_result1 = A * B
time_mul1 = time.time() - start
# print(f"A * B 결과 형태: {mul_result1.shape}")
# print(f"실행 시간: {time_mul1:.8f}초")

# 방법 2: einsum 사용
start = time.time()
mul_result2 = np.einsum('i,i->i', A, B)
time_mul2 = time.time() - start
# print(f"np.einsum('i,i->i', A, B) 결과 형태: {mul_result2.shape}")
# print(f"실행 시간: {time_mul2:.8f}초")

# 방법 3: np.multiply 사용
start = time.time()
mul_result3 = np.multiply(A, B)
time_mul3 = time.time() - start
# print(f"np.multiply(A, B) 결과 형태: {mul_result3.shape}")
# print(f"실행 시간: {time_mul3:.8f}초")

# 결과 비교 (element-wise multiplication)
# print("\n요소별 곱 결과 비교 (처음 3개 요소만 표시):")
# print("A * B:")
# print(mul_result1[:3])
# print("np.einsum:")
# print(mul_result2[:3])
# print("np.multiply:")
# print(mul_result3[:3])

# 성능 비교 표
# print("\n성능 비교 표:")
# print("=" * 70)
# print(f"{'연산':<20} | {'방법':<30} | {'실행 시간(초)':<15}")
# print("-" * 70)
# # 내적
# print(f"{'내적(Inner)':<20} | {'np.inner':<30} | {time_inner1:<15.8f}")
# print(f"{'내적(Inner)':<20} | {'np.einsum(i,i->)':<30} | {time_inner2:<15.8f}")
# print(f"{'내적(Inner)':<20} | {'np.sum(A * B)':<30} | {time_inner3:<15.8f}")
# # 외적
# print(f"{'외적(Outer)':<20} | {'np.outer':<30} | {time_outer1:<15.8f}")
# print(f"{'외적(Outer)':<20} | {'np.einsum(i,j->ij)':<30} | {time_outer2:<15.8f}")
# print(f"{'외적(Outer)':<20} | {'A[:,newaxis] * B[newaxis,:]':<30} | {time_outer3:<15.8f}")
# # 합계
# print(f"{'합계(Sum)':<20} | {'np.sum(A) + np.sum(B)':<30} | {time_sum1:<15.8f}")
# print(f"{'합계(Sum)':<20} | {'np.einsum(i->) + np.einsum(i->)':<30} | {time_sum2:<15.8f}")
# print(f"{'합계(Sum)':<20} | {'np.einsum(i->, concatenate)':<30} | {time_sum3:<15.8f}")
# # 요소별 곱
# print(f"{'요소별 곱(Mul)':<20} | {'A * B':<30} | {time_mul1:<15.8f}")
# print(f"{'요소별 곱(Mul)':<20} | {'np.einsum(i,i->i)':<30} | {time_mul2:<15.8f}")
# print(f"{'요소별 곱(Mul)':<20} | {'np.multiply':<30} | {time_mul3:<15.8f}")
# print("=" * 70)

# 대용량 벡터에서의 성능 비교
# print("\n대용량 벡터에서의 성능 비교:")
# # 더 큰 벡터 생성
# large_size = 1000000
# large_A = np.random.rand(large_size)
# large_B = np.random.rand(large_size)

# print(f"벡터 크기: {large_size}, 메모리 사용량: {large_A.nbytes / (1024*1024):.2f} MB (각 벡터)")

# # 대용량 벡터에 대한 내적 연산 비교
# operations = [
#     ("내적 - np.inner", lambda: np.inner(large_A, large_B)),
#     ("내적 - np.einsum", lambda: np.einsum('i,i->', large_A, large_B)),
#     ("내적 - np.sum(A*B)", lambda: np.sum(large_A * large_B)),
#     ("합계 - np.sum", lambda: np.sum(large_A) + np.sum(large_B)),
#     ("합계 - np.einsum", lambda: np.einsum('i->', large_A) + np.einsum('i->', large_B)),
#     ("요소별 곱 - A*B", lambda: large_A * large_B),
#     ("요소별 곱 - np.einsum", lambda: np.einsum('i,i->i', large_A, large_B)),
#     ("요소별 곱 - np.multiply", lambda: np.multiply(large_A, large_B))
# ]

# print("\n대용량 벡터 성능 비교:")
# print("=" * 60)
# print(f"{'연산 방법':<30} | {'실행 시간(초)':<15} | {'결과'}")
# print("-" * 60)

# for name, operation in operations:
#     start = time.time()
#     result = operation()
#     exec_time = time.time() - start
#     if isinstance(result, np.ndarray) and result.size > 3:
#         result_str = f"형태: {result.shape}"
#     else:
#         result_str = f"{result}"
#     print(f"{name:<30} | {exec_time:<15.6f} | {result_str}")

# print("=" * 60)

# 종합적인 einsum 유틸리티 함수
def einsum_operations(A, B=None, operation='inner'):
    """
    두 벡터 또는 배열 A와 B에 대해 다양한 einsum 연산을 수행하는 유틸리티 함수
    
    Parameters:
    -----------
    A : numpy.ndarray
        첫 번째 입력 배열
    B : numpy.ndarray, optional
        두 번째 입력 배열 (필요한 경우)
    operation : str
        수행할 연산 ('inner', 'outer', 'sum', 'mul', 'matmul', 'trace', 'transpose')
        
    Returns:
    --------
    result : numpy.ndarray or scalar
        연산 결과
    """
    if operation == 'inner':
        if B is None:
            raise ValueError("내적 계산에는 두 번째 배열이 필요합니다.")
        return np.einsum('i,i->', A, B)
    
    elif operation == 'outer':
        if B is None:
            raise ValueError("외적 계산에는 두 번째 배열이 필요합니다.")
        return np.einsum('i,j->ij', A, B)
    
    elif operation == 'sum':
        return np.einsum('i->', A)
    
    elif operation == 'mul':
        if B is None:
            raise ValueError("요소별 곱에는 두 번째 배열이 필요합니다.")
        return np.einsum('i,i->i', A, B)
    
    elif operation == 'matmul':
        if B is None:
            raise ValueError("행렬 곱에는 두 번째 배열이 필요합니다.")
        return np.einsum('ij,jk->ik', A, B)
    
    elif operation == 'trace':
        return np.einsum('ii->', A)
    
    elif operation == 'transpose':
        return np.einsum('ij->ji', A)
    
    else:
        raise ValueError(f"지원되지 않는 연산: {operation}")

# 함수 테스트
# print("\n유틸리티 함수 테스트:")
test_A = np.array([1, 2, 3])
test_B = np.array([4, 5, 6])
test_matrix = np.array([[1, 2], [3, 4]])

# print(f"벡터 A: {test_A}")
# print(f"벡터 B: {test_B}")
# print(f"행렬 M: \n{test_matrix}")

# print(f"\n내적 (inner): {einsum_operations(test_A, test_B, 'inner')}")
# print(f"벡터 합 (sum): {einsum_operations(test_A, operation='sum')}")
# print(f"행렬 대각합 (trace): {einsum_operations(test_matrix, operation='trace')}")
# print(f"행렬 전치 (transpose): \n{einsum_operations(test_matrix, operation='transpose')}")