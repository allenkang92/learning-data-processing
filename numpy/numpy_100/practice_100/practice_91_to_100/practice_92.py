# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods

import numpy as np
import time

# 문제: 큰 벡터 Z에 대해 3가지 다른 방법으로 Z의 3제곱 계산하기

# 테스트용 대규모 벡터 생성
size = 10_000_000  # 1천만 개 요소
Z = np.random.random(size)

print(f"벡터 Z 크기: {Z.size}개 요소")
print(f"벡터 Z 처음 5개 요소: {Z[:5]}")

# 방법 1: 직접 제곱 연산자 사용
print("\n방법 1: 제곱 연산자 사용 (Z**3)")
start = time.time()
Z_cubed_1 = Z**3
time_method1 = time.time() - start

print(f"계산 시간: {time_method1:.6f}초")
print(f"결과 처음 5개 요소: {Z_cubed_1[:5]}")

# 방법 2: np.power 함수 사용
print("\n방법 2: np.power 함수 사용")
start = time.time()
Z_cubed_2 = np.power(Z, 3)
time_method2 = time.time() - start

print(f"계산 시간: {time_method2:.6f}초")
print(f"결과 처음 5개 요소: {Z_cubed_2[:5]}")

# 방법 3: Z * Z * Z (곱셈 사용)
print("\n방법 3: 곱셈 연산 사용 (Z * Z * Z)")
start = time.time()
Z_cubed_3 = Z * Z * Z
time_method3 = time.time() - start

print(f"계산 시간: {time_method3:.6f}초")
print(f"결과 처음 5개 요소: {Z_cubed_3[:5]}")

# 추가 방법: np.multiply.reduce
print("\n추가 방법: np.multiply.reduce 사용")
start = time.time()
Z_cubed_4 = np.multiply.reduce([Z, Z, Z])
time_method4 = time.time() - start

print(f"계산 시간: {time_method4:.6f}초")
print(f"결과 처음 5개 요소: {Z_cubed_4[:5]}")

# 성능 비교 표
print("\n성능 비교:")
print("=" * 60)
print(f"{'방법':<20} | {'시간(초)':<12} | {'기준 대비 속도':<15} |")
print("-" * 60)
print(f"{'제곱 연산자 (Z**3)':<20} | {time_method1:<12.6f} | {1.0:<15.2f} |")
print(f"{'np.power(Z, 3)':<20} | {time_method2:<12.6f} | {time_method1/time_method2:<15.2f} |")
print(f"{'Z * Z * Z':<20} | {time_method3:<12.6f} | {time_method1/time_method3:<15.2f} |")
print(f"{'np.multiply.reduce':<20} | {time_method4:<12.6f} | {time_method1/time_method4:<15.2f} |")
print("=" * 60)

# 결과 검증: 모든 방법이 동일한 결과를 반환하는지 확인
print("\n결과 검증:")
print("방법 1 == 방법 2:", np.allclose(Z_cubed_1, Z_cubed_2))
print("방법 2 == 방법 3:", np.allclose(Z_cubed_2, Z_cubed_3))
print("방법 3 == 방법 4:", np.allclose(Z_cubed_3, Z_cubed_4))

# 메모리 사용량 및 정밀도 고려
# 1. 제곱 연산자 또는 np.power는 일반적으로 최적화되어 있으며 가독성이 높음
# 2. Z * Z * Z는 중간 결과를 저장하므로, 대규모 배열에서 메모리 사용량이 증가할 수 있음
# 3. 64비트 부동소수점 숫자의 경우, 매우 큰 값에서는 정밀도 손실 가능성 고려 필요
# 4. 가독성과 성능을 고려할 때 Z**3 또는 np.power(Z, 3)이 권장됨

# 숫자 범위와 데이터 타입에 따른 영향
print("\n다양한 데이터 타입에서의 테스트:")
# 정수 타입 벡터에서의 성능 비교
Z_int = np.random.randint(0, 100, size=1_000_000, dtype=np.int32)

start = time.time()
Z_int_cubed_1 = Z_int**3
time_int_1 = time.time() - start

start = time.time()
Z_int_cubed_2 = Z_int * Z_int * Z_int
time_int_2 = time.time() - start

print(f"정수 타입: 제곱 연산자 시간 = {time_int_1:.6f}초, 곱셈 시간 = {time_int_2:.6f}초")
print(f"정수 연산에서 곱셈 대비 제곱 연산자 속도: {time_int_2/time_int_1:.2f}배")

# 숫자 범위에 따른 고려사항
# - 아주 작은 숫자나 아주 큰 숫자의 경우 정밀도 문제 발생 가능
# - 정수 타입의 경우 오버플로우 가능성 고려 필요
# - 계산 중간 결과가 데이터 타입의 범위를 초과하지 않도록 주의
