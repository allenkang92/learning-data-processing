# 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

import numpy as np
import time
import itertools

# 문제: 임의의 개수 벡터에 대한 데카르트 곱(모든 조합) 생성하기

# 예제 벡터 생성
vec1 = np.array([1, 2, 3])
vec2 = np.array(['a', 'b'])
vec3 = np.array([True, False])
print("벡터 1:", vec1)
print("벡터 2:", vec2)
print("벡터 3:", vec3)

# 방법 1: 중첩 반복문 (작은 예제용)
print("\n방법 1: 중첩 반복문 사용")
print("2개 벡터의 데카르트 곱:")
for i in vec1:
    for j in vec2:
        print(f"({i}, {j})")

print("\n3개 벡터의 데카르트 곱 (일부):")
count = 0
for i in vec1:
    for j in vec2:
        for k in vec3:
            print(f"({i}, {j}, {k})")
            count += 1
            if count >= 6:  # 6개만 출력
                break
        if count >= 6:
            break
    if count >= 6:
        break
print("... 등 총", len(vec1) * len(vec2) * len(vec3), "개 조합")

# 방법 2: NumPy의 meshgrid 사용 (2차원까지만 직관적)
print("\n방법 2: np.meshgrid 사용 (2차원)")
X, Y = np.meshgrid(vec1, vec2)
cartesian_2d = np.column_stack([X.flatten(), Y.flatten()])
print("2차원 데카르트 곱 결과:")
print(cartesian_2d)

# 방법 3: NumPy의 ix_ 인덱서 사용
print("\n방법 3: np.ix_ 인덱서 사용")
# ix_는 sparse mesh grid를 생성합니다.
try:
    # 주의: 모든 배열이 정수 또는 부울 인덱스여야 함
    # 문자열 등을 사용하려면 숫자 인덱스로 변환 필요
    vec1_idx = np.arange(len(vec1))
    vec2_idx = np.arange(len(vec2))
    vec3_idx = np.arange(len(vec3))
    
    grid_idx = np.ix_(vec1_idx, vec2_idx, vec3_idx)
    print("ix_ 결과 형태:", [g.shape for g in grid_idx])
    
    # ix_ 결과를 사용하여 값 조회
    result_shape = tuple(len(v) for v in [vec1, vec2, vec3])
    results = np.zeros(result_shape, dtype=object)
    
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            for k in range(len(vec3)):
                results[i, j, k] = (vec1[i], vec2[j], vec3[k])
    
    print("ix_ 기반 결과 (처음 3개):")
    flat_results = results.flatten()
    for i in range(min(3, len(flat_results))):
        print(flat_results[i])
except Exception as e:
    print(f"ix_ 방법 오류: {e}")

# 방법 4: itertools.product 사용 (가장 범용적이고 권장됨)
print("\n방법 4: itertools.product 사용 (권장)")
start = time.time()
cart_product = list(itertools.product(vec1, vec2, vec3))
time_product = time.time() - start

print("데카르트 곱 결과 (처음 5개):")
for i in range(min(5, len(cart_product))):
    print(cart_product[i])
print(f"총 {len(cart_product)}개 조합, 실행 시간: {time_product:.8f}초")

# 방법 5: NumPy 전용 구현 (모든 타입의 벡터에 적용 가능)
print("\n방법 5: NumPy 전용 구현")

def cartesian_product(*arrays):
    """NumPy 배열의 데카르트 곱 계산"""
    n = len(arrays)
    # 결과 배열의 형태와 크기 결정
    shape = tuple(len(a) for a in arrays)
    total_combinations = np.prod(shape).astype(int)
    
    # 결과를 위한 배열 생성
    result = np.zeros((total_combinations, n), dtype=object)
    
    # 각 차원에 대한 인덱스 배열 생성
    for i, arr in enumerate(arrays):
        # 이 차원의 값을 반복해야 하는 횟수 계산
        repeat_count = np.prod([len(a) for a in arrays[i+1:]]).astype(int) if i < n-1 else 1
        # 이 차원의 값을 타일링해야 하는 횟수 계산
        tile_count = np.prod([len(a) for a in arrays[:i]]).astype(int) if i > 0 else 1
        
        # 값 생성 및 타일링
        indices = np.repeat(np.arange(len(arr)), repeat_count)
        indices = np.tile(indices, tile_count)
        
        # 결과 배열에 추가
        result[:, i] = np.array(arr)[indices]
    
    return result

start = time.time()
cart_product_np = cartesian_product(vec1, vec2, vec3)
time_numpy = time.time() - start

print("NumPy 구현 결과 (처음 5개):")
for i in range(min(5, len(cart_product_np))):
    print(tuple(cart_product_np[i]))
print(f"총 {len(cart_product_np)}개 조합, 실행 시간: {time_numpy:.8f}초")

# 벡터 길이에 따른 성능 비교
print("\n벡터 길이에 따른 성능 비교:")
lengths = [5, 10, 15]
print("=" * 60)
print(f"{'벡터 길이':<12} | {'조합 수':<12} | {'itertools(초)':<15} | {'NumPy(초)':<15} |")
print("-" * 60)

for length in lengths:
    # 길이가 같은 2개 벡터 생성
    a = np.arange(length)
    b = np.arange(length)
    
    # itertools.product 시간 측정
    start = time.time()
    prod_iter = list(itertools.product(a, b))
    time_iter = time.time() - start
    
    # NumPy 구현 시간 측정
    start = time.time()
    prod_np = cartesian_product(a, b)
    time_np = time.time() - start
    
    print(f"{length:<12} | {length*length:<12} | {time_iter:<15.8f} | {time_np:<15.8f} |")

print("=" * 60)

# 실제 응용 사례
print("\n실제 응용 사례:")
print("1. 하이퍼파라미터 그리드 탐색 (머신러닝)")
# 하이퍼파라미터 그리드 정의
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
epochs = [10, 50, 100]

print(f"학습률: {learning_rates}")
print(f"배치 크기: {batch_sizes}")
print(f"에폭: {epochs}")

# 모든 하이퍼파라미터 조합 생성
hyperparameter_grid = list(itertools.product(learning_rates, batch_sizes, epochs))
print(f"\n총 {len(hyperparameter_grid)}개 하이퍼파라미터 조합 (처음 3개):")
for i in range(3):
    lr, bs, ep = hyperparameter_grid[i]
    print(f"조합 {i+1}: 학습률={lr}, 배치 크기={bs}, 에폭={ep}")

print("\n2. RGB 색상 팔레트 생성")
# 8비트 색상 값 (간소화를 위해 값 제한)
red_values = [0, 128, 255]
green_values = [0, 128, 255]
blue_values = [0, 128, 255]

# RGB 색상 조합 생성
rgb_combinations = list(itertools.product(red_values, green_values, blue_values))
print(f"\n총 {len(rgb_combinations)}개 RGB 색상 조합:")
for i, (r, g, b) in enumerate(rgb_combinations):
    print(f"Color {i+1}: RGB({r}, {g}, {b})")

# 고려사항
# 1. 데카르트 곱의 결과는 벡터 크기에 따라 기하급수적으로 증가합니다.
# 2. 매우 큰 벡터에 대해 메모리 부족 문제가 발생할 수 있습니다.
# 3. 필요한 경우 제너레이터 형태로 사용하여 메모리 효율성을 높일 수 있습니다.
# 4. 대용량 데이터에 대해 샘플링이나 분할 처리 기법을 고려하세요.
# 5. NumPy 구현은 dtype=object를 사용하여 성능이 저하될 수 있습니다.
