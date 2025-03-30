# 95. Convert a vector of ints into a matrix binary representation

import numpy as np
import time

# 문제: 정수 벡터를 이진 표현 행렬로 변환하기

# 테스트용 정수 벡터 생성
vector = np.array([0, 1, 2, 3, 15, 16, 127, 128, 255], dtype=np.int32)
print("원본 정수 벡터:", vector)

# 방법 1: 비트 연산자 사용
print("\n방법 1: 비트 연산자 사용")
start = time.time()
n_elements = len(vector)
n_bits = 8  # 각 요소의 비트 수 (1바이트 = 8비트)
binary_repr_matrix1 = np.zeros((n_elements, n_bits), dtype=np.uint8)

for i in range(n_elements):
    for j in range(n_bits):
        # i번째 숫자의 j번째 비트 추출
        # (vector[i] >> j) 는 i번째 숫자를 j비트 만큼 오른쪽으로 쉬프트
        # & 1 연산은 최하위 비트만 남김
        binary_repr_matrix1[i, n_bits - 1 - j] = (vector[i] >> j) & 1

time1 = time.time() - start
print("이진 표현 행렬 (비트 연산):")
print(binary_repr_matrix1)
print(f"실행 시간: {time1:.8f}초")

# 방법 2: NumPy의 unpackbits 사용
print("\n방법 2: np.unpackbits 사용 (8비트)")
start = time.time()
# unpackbits는 uint8 타입의 배열만 처리 가능
vector_uint8 = np.array(vector, dtype=np.uint8)
binary_repr_matrix2 = np.unpackbits(vector_uint8[:, np.newaxis], axis=1)
time2 = time.time() - start

print("이진 표현 행렬 (unpackbits):")
print(binary_repr_matrix2)
print(f"실행 시간: {time2:.8f}초")

# 방법 3: np.binary_repr 문자열 함수 사용 후 변환
print("\n방법 3: np.binary_repr 문자열 함수 사용")
start = time.time()
# binary_repr는 문자열 반환
binary_strings = [np.binary_repr(num, width=8) for num in vector]
# 문자열을 정수 배열로 변환
binary_repr_matrix3 = np.array([[int(bit) for bit in s] for s in binary_strings], dtype=np.uint8)
time3 = time.time() - start

print("이진 표현 행렬 (binary_repr):")
print(binary_repr_matrix3)
print(f"실행 시간: {time3:.8f}초")

# 방법 4: 벡터화된 비트 연산 (고급 방법)
print("\n방법 4: 벡터화된 비트 연산 (고급 방법)")
start = time.time()
# 비트 마스크 생성: 2^0, 2^1, 2^2, ..., 2^7
bits = 1 << np.arange(8, dtype=np.uint8)[::-1]
# 각 숫자에 비트 마스크 적용하여 이진 표현 추출
binary_repr_matrix4 = ((vector[:, np.newaxis] & bits) > 0).astype(np.uint8)
time4 = time.time() - start

print("이진 표현 행렬 (벡터화된 비트 연산):")
print(binary_repr_matrix4)
print(f"실행 시간: {time4:.8f}초")

# 성능 비교 표
print("\n성능 비교:")
print("=" * 60)
print(f"{'방법':<25} | {'실행 시간(초)':<15} | {'상대 속도':<10} |")
print("-" * 60)
print(f"{'1. 비트 연산자':<25} | {time1:<15.8f} | {1.0:<10.2f} |")
print(f"{'2. np.unpackbits':<25} | {time2:<15.8f} | {time1/time2:<10.2f} |")
print(f"{'3. np.binary_repr':<25} | {time3:<15.8f} | {time1/time3:<10.2f} |")
print(f"{'4. 벡터화된 비트 연산':<25} | {time4:<15.8f} | {time1/time4:<10.2f} |")
print("=" * 60)

# 결과 검증
print("\n결과 검증 (모든 방법의 결과가 일치하는지):")
print("방법 1 == 방법 2:", np.array_equal(binary_repr_matrix1, binary_repr_matrix2))
print("방법 2 == 방법 3:", np.array_equal(binary_repr_matrix2, binary_repr_matrix3))
print("방법 3 == 방법 4:", np.array_equal(binary_repr_matrix3, binary_repr_matrix4))

# 더 큰 수 다루기 (8비트 이상)
print("\n더 큰 수 다루기 (8비트 이상):")
large_nums = np.array([256, 1000, 65535, 65536])  # 8비트 이상 필요한 수
print("큰 정수 벡터:", large_nums)

# 16비트로 변환 (더 큰 숫자 표현)
def int_to_binary_matrix(vector, n_bits=16):
    """정수 벡터를 지정된 비트 수로 이진 표현 행렬로 변환"""
    n_elements = len(vector)
    
    # 벡터 내 최대값이 필요로 하는 비트 수 계산
    max_value = np.max(vector)
    required_bits = max(n_bits, int(np.ceil(np.log2(max_value + 1))))
    
    if required_bits > n_bits:
        print(f"경고: 벡터 내 최대값 {max_value}는 {required_bits}비트가 필요하지만, {n_bits}비트가 제공되었습니다.")
        print(f"필요한 비트 수 {required_bits}로 자동 조정합니다.")
        n_bits = required_bits
    
    binary_matrix = np.zeros((n_elements, n_bits), dtype=np.uint8)
    
    for i in range(n_elements):
        # 이진 문자열 얻기
        binary_str = np.binary_repr(vector[i], width=n_bits)
        # 문자열을 정수 배열로 변환
        binary_matrix[i] = [int(bit) for bit in binary_str]
    
    return binary_matrix

# 16비트 이진 표현
binary_large = int_to_binary_matrix(large_nums, n_bits=16)
print("\n16비트 이진 표현 행렬:")
print(binary_large)

# 32비트로 변환 (더 큰 숫자 표현)
binary_large_32bit = int_to_binary_matrix(large_nums, n_bits=32)
print("\n32비트 이진 표현 행렬 (처음 16비트만 표시):")
print(binary_large_32bit[:, :16])  # 앞부분만 표시

# 벡터화된 비트 연산으로 16비트 표현 (방법 4 확장)
def int_to_binary_vectorized(vector, n_bits=8):
    """정수 벡터를 이진 표현 행렬로 변환 (벡터화된 연산 사용)"""
    # 벡터 내 최대값이 필요로 하는 비트 수 계산
    max_value = np.max(vector)
    required_bits = max(n_bits, int(np.ceil(np.log2(max_value + 1))))
    
    if required_bits > n_bits:
        print(f"경고: 벡터 내 최대값 {max_value}는 {required_bits}비트가 필요하지만, {n_bits}비트가 제공되었습니다.")
        print(f"필요한 비트 수 {required_bits}로 자동 조정합니다.")
        n_bits = required_bits
        
    return ((vector[:, np.newaxis] & (1 << np.arange(n_bits)[::-1])) > 0).astype(np.uint8)

binary_large_vectorized = int_to_binary_vectorized(large_nums, n_bits=16)
print("\n벡터화된 16비트 이진 표현:")
print(binary_large_vectorized)

# 비트플레인 분리 예제
print("\n이미지 비트플레인 분리 예제 (개념적):")
# 가상의 8x8 회색조 이미지 (0-255 픽셀 값)
image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
print("8x8 회색조 이미지:")
print(image)

# 이미지의 모든 픽셀을 이진 표현으로 변환
image_flat = image.flatten()
binary_image = np.unpackbits(image_flat[:, np.newaxis], axis=1).reshape(image.shape + (8,))

# 최상위 비트와 최하위 비트 플레인 출력
print("\n최상위 비트 플레인 (MSB):")
print(binary_image[:, :, 0])  # 첫 번째 비트 플레인 (MSB)
print("\n최하위 비트 플레인 (LSB):")
print(binary_image[:, :, 7])  # 마지막 비트 플레인 (LSB)

# 메모리 최적화 고려사항
# 1. np.packbits/unpackbits는 메모리 효율적인 변환 제공
# 2. 큰 배열 처리 시 메모리 사용을 줄이기 위해 청크로 분할 처리 고려
# 3. 불필요한 중간 데이터 구조 피하기
# 4. Boolean 타입(1비트) vs uint8 타입(8비트)의 저장 공간 차이 고려

# 주요 NumPy 함수
# - np.binary_repr(): 정수를 이진 문자열로 변환
# - np.unpackbits(): uint8 배열을 비트로 풀어냄
# - np.packbits(): 비트 배열을 uint8로 압축