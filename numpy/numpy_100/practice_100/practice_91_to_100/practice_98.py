# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate

# 문제: 두 벡터(X,Y)로 표현된 경로를 등거리 샘플을 사용하여 샘플링하는 방법

# 테스트용 경로 생성 (비등간격 X, Y 데이터)
np.random.seed(42)  # 재현성을 위한 시드 설정
n_points = 100
# 비선형 경로 생성 (나선형 경로)
t = np.linspace(0, 4*np.pi, n_points)
X = t * np.cos(t)
Y = t * np.sin(t)
# 일부 구간에서 집중된 샘플링 (비등간격 샘플링 시뮬레이션)
noise = np.random.randn(n_points) * 0.1
X += noise
Y += noise

# 원본 경로 계산
original_points = np.column_stack((X, Y))

print(f"원본 경로 포인트 수: {len(X)}")

# 방법 1: 누적 거리 계산 후 선형 보간 사용
print("\n방법 1: 누적 거리 계산 후 선형 보간")
start = time.time()

# 경로를 따라 점 사이의 거리 계산
dists = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
# 누적 거리 계산
cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
# 총 거리
total_dist = cumulative_dist[-1]

# 샘플링할 등거리 지점 생성
n_samples = 50  # 원하는 샘플 수
equal_dist_points = np.linspace(0, total_dist, n_samples)

# 누적 거리를 기준으로 X, Y를 보간
interp_X = np.interp(equal_dist_points, cumulative_dist, X)
interp_Y = np.interp(equal_dist_points, cumulative_dist, Y)

time_method1 = time.time() - start
print(f"방법 1 - 실행 시간: {time_method1:.8f}초")
print(f"방법 1 - 등거리 샘플 수: {len(interp_X)}")

# 방법 2: scipy.interpolate의 CubicSpline 사용
print("\n방법 2: scipy.interpolate의 CubicSpline 사용")
start = time.time()

# 매개변수화 - 누적 거리 기준
t_param = cumulative_dist / total_dist  # 0~1 정규화

# CubicSpline 보간
cs_x = interpolate.CubicSpline(t_param, X)
cs_y = interpolate.CubicSpline(t_param, Y)

# 등거리 샘플링 포인트 생성
t_equal = np.linspace(0, 1, n_samples)
spline_X = cs_x(t_equal)
spline_Y = cs_y(t_equal)

time_method2 = time.time() - start
print(f"방법 2 - 실행 시간: {time_method2:.8f}초")
print(f"방법 2 - 등거리 샘플 수: {len(spline_X)}")

# 방법 3: 적응형 샘플링 (iterative approach)
print("\n방법 3: 적응형 샘플링 (반복 접근법)")
start = time.time()

def adaptive_sampling(X, Y, n_desired_samples):
    """
    적응형 샘플링을 통해 경로를 등거리로 재샘플링
    
    Parameters:
    -----------
    X, Y : 원본 경로 좌표
    n_desired_samples : 원하는 샘플 수
    
    Returns:
    --------
    new_X, new_Y : 등거리 샘플링된 좌표
    """
    # 원본 포인트들
    points = np.column_stack((X, Y))
    
    # 경로 길이 계산
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    
    # 목표 거리 간격
    target_dist = total_dist / (n_desired_samples - 1)
    
    # 첫 번째 포인트 추가
    new_points = [points[0]]
    current_dist = 0
    
    # 적응형 샘플링
    for i in range(1, len(points)):
        # 현재 세그먼트의 거리
        segment_dist = dists[i-1]
        
        # 현재 세그먼트 내에서 샘플링해야 할 포인트가 있는지 확인
        while current_dist + segment_dist >= target_dist:
            # 세그먼트 내에서의 상대적 위치
            alpha = (target_dist - current_dist) / segment_dist
            # 새 포인트 계산
            interp_point = points[i-1] + alpha * (points[i] - points[i-1])
            new_points.append(interp_point)
            # 다음 타겟 거리로 업데이트
            segment_dist -= (target_dist - current_dist)
            current_dist = 0
        
        # 남은 세그먼트 거리 업데이트
        current_dist += segment_dist
    
    # 마지막 포인트가 추가되지 않았다면 추가
    if len(new_points) < n_desired_samples:
        new_points.append(points[-1])
    
    # 최종 결과를 배열로 변환
    new_points = np.array(new_points)
    return new_points[:, 0], new_points[:, 1]

adaptive_X, adaptive_Y = adaptive_sampling(X, Y, n_samples)

time_method3 = time.time() - start
print(f"방법 3 - 실행 시간: {time_method3:.8f}초")
print(f"방법 3 - 등거리 샘플 수: {len(adaptive_X)}")

# 방법 4: 반데르몬드 행렬 방식 (curve fitting approach)
print("\n방법 4: 반데르몬드 행렬 방식 (곡선 피팅)")
start = time.time()

def vandermonde_sampling(X, Y, n_desired_samples, degree=3):
    """
    반데르몬드 행렬을 사용한 다항식 피팅으로 경로를 등거리 샘플링
    
    Parameters:
    -----------
    X, Y : 원본 경로 좌표
    n_desired_samples : 원하는 샘플 수
    degree : 다항식 차수
    
    Returns:
    --------
    new_X, new_Y : 등거리 샘플링된 좌표
    """
    # 경로 길이 계산
    dists = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    
    # 경로 파라미터화 (0~1)
    t = cumulative_dist / total_dist
    
    # 다항식 피팅 (반데르몬드 행렬 사용)
    x_coeffs = np.polyfit(t, X, degree)
    y_coeffs = np.polyfit(t, Y, degree)
    
    # 등거리 샘플링 파라미터
    t_equal = np.linspace(0, 1, n_desired_samples)
    
    # 다항식으로부터 새로운 좌표 계산
    vand_X = np.polyval(x_coeffs, t_equal)
    vand_Y = np.polyval(y_coeffs, t_equal)
    
    return vand_X, vand_Y

vand_X, vand_Y = vandermonde_sampling(X, Y, n_samples)

time_method4 = time.time() - start
print(f"방법 4 - 실행 시간: {time_method4:.8f}초")
print(f"방법 4 - 등거리 샘플 수: {len(vand_X)}")

# 샘플링된 포인트 간 거리 계산 (등거리 확인용)
print("\n샘플링된 포인트 간 거리 통계:")

def compute_distance_stats(X, Y):
    """포인트 간 거리의 통계 계산"""
    dists = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
    return {
        "최소 거리": np.min(dists),
        "최대 거리": np.max(dists),
        "평균 거리": np.mean(dists),
        "표준 편차": np.std(dists),
        "변동 계수(CV)": np.std(dists) / np.mean(dists) * 100  # 낮을수록 더 등거리에 가까움
    }

print("원본 경로 거리 통계:")
original_stats = compute_distance_stats(X, Y)
for key, value in original_stats.items():
    print(f"  {key}: {value:.6f}")

print("\n방법 1 (선형 보간) 거리 통계:")
method1_stats = compute_distance_stats(interp_X, interp_Y)
for key, value in method1_stats.items():
    print(f"  {key}: {value:.6f}")

print("\n방법 2 (CubicSpline) 거리 통계:")
method2_stats = compute_distance_stats(spline_X, spline_Y)
for key, value in method2_stats.items():
    print(f"  {key}: {value:.6f}")

print("\n방법 3 (적응형 샘플링) 거리 통계:")
method3_stats = compute_distance_stats(adaptive_X, adaptive_Y)
for key, value in method3_stats.items():
    print(f"  {key}: {value:.6f}")

print("\n방법 4 (반데르몬드) 거리 통계:")
method4_stats = compute_distance_stats(vand_X, vand_Y)
for key, value in method4_stats.items():
    print(f"  {key}: {value:.6f}")

# 방법 간 성능 비교 표
print("\n성능 및 정확도 비교:")
print("=" * 70)
print(f"{'방법':<20} | {'실행 시간(초)':<15} | {'CV(%)':<10} | {'평균 거리':<10}")
print("-" * 70)
print(f"{'1. 선형 보간':<20} | {time_method1:<15.8f} | {method1_stats['변동 계수(CV)']:<10.2f} | {method1_stats['평균 거리']:<10.6f}")
print(f"{'2. CubicSpline':<20} | {time_method2:<15.8f} | {method2_stats['변동 계수(CV)']:<10.2f} | {method2_stats['평균 거리']:<10.6f}")
print(f"{'3. 적응형 샘플링':<20} | {time_method3:<15.8f} | {method3_stats['변동 계수(CV)']:<10.2f} | {method3_stats['평균 거리']:<10.6f}")
print(f"{'4. 반데르몬드':<20} | {time_method4:<15.8f} | {method4_stats['변동 계수(CV)']:<10.2f} | {method4_stats['평균 거리']:<10.6f}")
print("=" * 70)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# 원본 경로와 각 방법 비교 시각화
methods = [
    ("선형 보간", interp_X, interp_Y, method1_stats),
    ("CubicSpline", spline_X, spline_Y, method2_stats),
    ("적응형 샘플링", adaptive_X, adaptive_Y, method3_stats),
    ("반데르몬드 다항식", vand_X, vand_Y, method4_stats)
]

for i, (name, method_X, method_Y, stats) in enumerate(methods):
    ax = axes[i]
    ax.plot(X, Y, 'k-', alpha=0.3, label='원본 경로')
    ax.plot(method_X, method_Y, 'ro-', markersize=4, label=name)
    
    # 첫 포인트와 마지막 포인트 강조
    ax.plot(method_X[0], method_Y[0], 'go', markersize=6, label='시작점')
    ax.plot(method_X[-1], method_Y[-1], 'bo', markersize=6, label='종료점')
    
    ax.set_title(f"{name} (CV: {stats['변동 계수(CV)']:.2f}%)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

plt.tight_layout()
# plt.savefig('equidistant_sampling_comparison.png', dpi=150)
# plt.show()

# 로봇 경로 계획 응용 예제
# 로봇 팔 웨이포인트 정의
def plan_robot_trajectory(waypoints_x, waypoints_y, n_samples=100):
    """
    주어진 웨이포인트를 기반으로 로봇 경로를 계획
    
    Parameters:
    -----------
    waypoints_x, waypoints_y : 경유점 좌표
    n_samples : 생성할 경로 포인트 수
    
    Returns:
    --------
    path_x, path_y : 등거리 샘플링된 경로 좌표
    """
    # 누적 거리 기반 파라미터화
    dists = np.sqrt(np.diff(waypoints_x)**2 + np.diff(waypoints_y)**2)
    t = np.insert(np.cumsum(dists), 0, 0)
    
    # 적응형 샘플링 적용
    path_x, path_y = adaptive_sampling(waypoints_x, waypoints_y, n_samples)
    
    return path_x, path_y

# 예시 웨이포인트
robot_waypoints_x = np.array([0, 2, 5, 8, 10, 8, 5, 2, 0])
robot_waypoints_y = np.array([0, 2, 3, 2, 0, -2, -3, -2, 0])

# 경로 계획
robot_path_x, robot_path_y = plan_robot_trajectory(robot_waypoints_x, robot_waypoints_y, 200)

# 로봇 경로 시각화
plt.figure(figsize=(10, 6))
plt.plot(robot_waypoints_x, robot_waypoints_y, 'ro-', label='웨이포인트')
plt.plot(robot_path_x, robot_path_y, 'b.-', alpha=0.6, label='등거리 샘플링된 경로')
plt.title('로봇 경로 계획 예시')
plt.xlabel('X 좌표')
plt.ylabel('Y 좌표')
plt.grid(True)
plt.legend()
plt.axis('equal')
# plt.savefig('robot_path_planning.png')
# plt.show()

print("로봇 경로 계획 완료")
print(f"웨이포인트 수: {len(robot_waypoints_x)}")
print(f"생성된 경로 포인트 수: {len(robot_path_x)}")

# 일반화된 등거리 샘플링 함수
def equidistant_sample_path(X, Y, n_samples=50, method='adaptive'):
    """
    경로를 등거리로 샘플링하는 일반화된 함수
    
    Parameters:
    -----------
    X, Y : 원본 경로 좌표
    n_samples : 원하는 샘플 수
    method : 사용할 방법 ('linear', 'spline', 'adaptive', 'vandermonde')
    
    Returns:
    --------
    sampled_X, sampled_Y : 등거리 샘플링된 좌표
    """
    # 경로 길이 계산
    dists = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    t_param = cumulative_dist / total_dist  # 정규화된 매개변수 (0~1)
    
    if method == 'linear':
        # 선형 보간 방법
        equal_dist_points = np.linspace(0, total_dist, n_samples)
        sampled_X = np.interp(equal_dist_points, cumulative_dist, X)
        sampled_Y = np.interp(equal_dist_points, cumulative_dist, Y)
    
    elif method == 'spline':
        # CubicSpline 방법
        cs_x = interpolate.CubicSpline(t_param, X)
        cs_y = interpolate.CubicSpline(t_param, Y)
        t_equal = np.linspace(0, 1, n_samples)
        sampled_X = cs_x(t_equal)
        sampled_Y = cs_y(t_equal)
    
    elif method == 'adaptive':
        # 적응형 샘플링 방법
        sampled_X, sampled_Y = adaptive_sampling(X, Y, n_samples)
    
    elif method == 'vandermonde':
        # 반데르몬드 방법
        sampled_X, sampled_Y = vandermonde_sampling(X, Y, n_samples)
    
    else:
        raise ValueError("지원되지 않는 방법입니다. 'linear', 'spline', 'adaptive', 'vandermonde' 중 하나를 사용하세요.")
    
    return sampled_X, sampled_Y

# 간단한 함수 테스트
print("\n일반화된 함수 테스트:")
test_path_x = np.array([0, 1, 2, 3, 3.5, 4, 5])
test_path_y = np.array([0, 0.5, 0, 1, 2, 2.5, 2])

print("원본 경로 포인트 수:", len(test_path_x))

# 다양한 방법으로 테스트
methods = ['linear', 'spline', 'adaptive', 'vandermonde']
for method in methods:
    sampled_x, sampled_y = equidistant_sample_path(test_path_x, test_path_y, n_samples=10, method=method)
    # 거리 계산
    dists = np.sqrt(np.diff(sampled_x)**2 + np.diff(sampled_y)**2)
    cv = np.std(dists) / np.mean(dists) * 100
    print(f"{method} 방법: 포인트 수={len(sampled_x)}, 변동 계수(CV)={cv:.2f}%")

# 결론 및 권장사항
print("\n결론 및 권장사항:")
print("1. 등거리 샘플링 필요성: 로봇 경로 계획, 3D 모델링, 애니메이션 등에서 중요")
print("2. 간단한 경로: 선형 보간이 빠르고 충분히 정확")
print("3. 복잡한 곡선 경로: CubicSpline 또는 적응형 샘플링 권장")
print("4. 계산 효율성 중요: 적응형 샘플링이 가장 효과적")
print("5. 노이즈 많은 데이터: 스플라인이나 반데르몬드로 평활화 가능")