# 89. How to get the n largest values of an array (★★★)

import numpy as np
import time

# 문제: 배열에서 n개의 가장 큰 값을 얻는 방법

# 테스트용 배열 생성
arr = np.random.randint(0, 1000, size=100)  # 0-999 사이의 값 100개
print("원본 배열 (처음 10개 요소):")
print(arr[:10])

n = 5  # 찾을 최대값 개수

# 방법 1: np.sort와 슬라이싱 사용
print("\n방법 1: np.sort와 슬라이싱 사용")
start = time.time()
sorted_arr = np.sort(arr)  # 오름차순 정렬
largest_n_sort = sorted_arr[-n:][::-1]  # 끝에서 n개 요소 추출 후 역순으로
time_sort = time.time() - start

print(f"가장 큰 {n}개 값 (정렬 사용):", largest_n_sort)
print(f"실행 시간: {time_sort:.8f}초")

# 방법 2: np.partition 사용 (정렬보다 효율적)
print("\n방법 2: np.partition 사용 (부분 정렬)")
start = time.time()
# arr의 끝에서 n개 요소를 분리하는 인덱스 -n에서 파티션 수행
# 이렇게 하면 n개의 가장 큰 값들이 배열의 끝부분에 위치하게 됨
partitioned = np.partition(arr, -n)
# 파티션된 배열에서 끝 n개 요소 추출
largest_n_partition = partitioned[-n:]
# 추출된 n개 요소를 정렬
largest_n_partition = np.sort(largest_n_partition)[::-1]
time_partition = time.time() - start

print(f"가장 큰 {n}개 값 (파티션 사용):", largest_n_partition)
print(f"실행 시간: {time_partition:.8f}초")

# 방법 3: np.argpartition 사용 (원래 인덱스 유지)
print("\n방법 3: np.argpartition 사용 (인덱스 유지)")
start = time.time()
# -n 위치에서 배열을 파티션하는 인덱스 배열 반환
indices = np.argpartition(arr, -n)[-n:]
# 해당 인덱스의 값 가져오기
largest_n_argpartition = arr[indices]
# 가져온 값들을 내림차순으로 정렬
sorted_indices = np.argsort(largest_n_argpartition)[::-1]
largest_n_argpartition = largest_n_argpartition[sorted_indices]
time_argpartition = time.time() - start

print(f"가장 큰 {n}개 값 (argpartition 사용):", largest_n_argpartition)
print(f"실행 시간: {time_argpartition:.8f}초")

# 방법 4: np.argsort 사용 (인덱스와 값을 모두 얻음)
print("\n방법 4: np.argsort 사용 (인덱스와 값 모두 얻기)")
start = time.time()
sorted_indices = np.argsort(arr)  # 오름차순 인덱스 정렬
largest_n_indices = sorted_indices[-n:][::-1]  # 끝에서 n개 인덱스
largest_n_argsort = arr[largest_n_indices]  # 해당 인덱스의 값
time_argsort = time.time() - start

print(f"가장 큰 {n}개 값 (argsort 사용):", largest_n_argsort)
print(f"인덱스:", largest_n_indices)
print(f"실행 시간: {time_argsort:.8f}초")

# 방법 5: heapq 모듈 사용 (Python 내장)
print("\n방법 5: heapq 모듈 사용 (Python 내장)")
import heapq
start = time.time()
# nlargest 함수로 최대값 n개 찾기
largest_n_heap = heapq.nlargest(n, arr)
time_heap = time.time() - start

print(f"가장 큰 {n}개 값 (heapq 사용):", largest_n_heap)
print(f"실행 시간: {time_heap:.8f}초")

# 결과 검증
print("\n결과 검증 (모든 방법이 동일한 결과를 반환하는지):")
print("방법 1 == 방법 2:", np.array_equal(largest_n_sort, largest_n_partition))
print("방법 2 == 방법 3:", np.array_equal(largest_n_partition, largest_n_argpartition))
print("방법 3 == 방법 4:", np.array_equal(largest_n_argpartition, largest_n_argsort))
print("방법 4 == 방법 5:", np.array_equal(largest_n_argsort, largest_n_heap))

# 성능 비교표
print("\n성능 비교표 (작은 배열):")
print("="*60)
print(f"{'방법':<15} | {'실행 시간(초)':<15} | {'정렬 여부':<10} | {'인덱스 반환':<15} |")
print("-"*60)
print(f"{'sort':<15} | {time_sort:<15.8f} | {'전체':<10} | {'아니오':<15} |")
print(f"{'partition':<15} | {time_partition:<15.8f} | {'부분':<10} | {'아니오':<15} |")
print(f"{'argpartition':<15} | {time_argpartition:<15.8f} | {'부분':<10} | {'예':<15} |")
print(f"{'argsort':<15} | {time_argsort:<15.8f} | {'전체':<10} | {'예':<15} |")
print(f"{'heapq':<15} | {time_heap:<15.8f} | {'부분':<10} | {'아니오':<15} |")
print("="*60)

# 대규모 배열에서의 성능 비교
print("\n대규모 배열에서의 성능 비교 (100만 요소):")
large_arr = np.random.randint(0, 1000000, size=1000000)
n_large = 10

# 시간 측정 함수
def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    return time.time() - start, result

# 각 방법에 대한 시간 측정
time_sort_large, _ = measure_time(lambda: np.sort(large_arr)[-n_large:][::-1])
time_partition_large, _ = measure_time(lambda: np.sort(np.partition(large_arr, -n_large)[-n_large:])[::-1])
time_argpartition_large, _ = measure_time(lambda: large_arr[np.argpartition(large_arr, -n_large)[-n_large:]])
time_argsort_large, _ = measure_time(lambda: large_arr[np.argsort(large_arr)[-n_large:][::-1]])
time_heap_large, _ = measure_time(lambda: heapq.nlargest(n_large, large_arr))

# 성능 비교표 (대규모 배열)
print("="*60)
print(f"{'방법':<15} | {'실행 시간(초)':<15} | {'상대 속도':<10} |")
print("-"*60)
print(f"{'sort':<15} | {time_sort_large:<15.8f} | {1.0:<10.2f} |")  # 기준
print(f"{'partition':<15} | {time_partition_large:<15.8f} | {time_sort_large/time_partition_large:<10.2f} |")
print(f"{'argpartition':<15} | {time_argpartition_large:<15.8f} | {time_sort_large/time_argpartition_large:<10.2f} |")
print(f"{'argsort':<15} | {time_argsort_large:<15.8f} | {time_sort_large/time_argsort_large:<10.2f} |")
print(f"{'heapq':<15} | {time_heap_large:<15.8f} | {time_sort_large/time_heap_large:<10.2f} |")
print("="*60)

# 응용 사례: 이미지 처리에서 가장 밝은 픽셀 찾기
print("\n응용 사례: 이미지 처리에서 가장 밝은 픽셀 찾기")
# 가상의 그레이스케일 이미지 생성 (0-255 값)
image = np.random.randint(0, 256, size=(100, 100))
print(f"이미지 크기: {image.shape}")
# 가장 밝은 10개 픽셀 값 찾기
brightest_values = np.partition(image.flatten(), -10)[-10:]
brightest_values = np.sort(brightest_values)[::-1]
print(f"가장 밝은 10개 픽셀 값: {brightest_values}")

# 인덱스도 함께 찾기
indices = np.argpartition(image.flatten(), -10)[-10:]
pixel_values = image.flatten()[indices]
# 값 기준으로 정렬
sorted_order = np.argsort(pixel_values)[::-1]
brightest_indices = indices[sorted_order]
brightest_values = pixel_values[sorted_order]
# 2D 인덱스 변환
brightest_coordinates = np.unravel_index(brightest_indices, image.shape)
print("\n가장 밝은 10개 픽셀의 좌표와 값:")
for i in range(10):
    print(f"좌표: ({brightest_coordinates[0][i]}, {brightest_coordinates[1][i]}), 값: {brightest_values[i]}")

# 고려사항
# 1. 대용량 데이터 처리 시 메모리 사용량 주의
# 2. 정렬은 계산 비용이 높은 작업이므로 필요한 만큼만 수행
# 3. partition 메서드는 부분 정렬만 수행하여 정확한 순서를 보장하지 않음
# 4. 민감한 데이터의 경우 최대값이 데이터 유출의 원인이 될 수 있으므로 주의
# 5. 전체 데이터가 필요하지 않은 경우 스트리밍 방식의 접근법 고려