# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means).

import numpy as np
import time
import matplotlib.pyplot as plt

# 문제: 1D 배열 X의 평균에 대한 부트스트랩 95% 신뢰 구간 계산하기

# 테스트 데이터 생성
np.random.seed(42)
X = np.random.normal(loc=5, scale=2, size=100)  # 평균 5, 표준편차 2인 정규분포에서 100개 샘플 추출

print("원본 데이터 통계:")
print(f"샘플 크기: {len(X)}")
print(f"샘플 평균: {np.mean(X):.4f}")
print(f"샘플 표준편차: {np.std(X, ddof=1):.4f}")

# 방법 1: 기본 부트스트랩 구현
def bootstrap_ci_basic(data, n_resamples=10000, ci=95):
    """
    기본적인 부트스트랩 방법으로 평균의 신뢰 구간 계산
    
    Parameters:
    -----------
    data : numpy.ndarray
        원본 1D 데이터 배열
    n_resamples : int
        부트스트랩 리샘플링 횟수
    ci : float
        신뢰 구간 수준 (0-100)
        
    Returns:
    --------
    tuple
        (하한, 상한) 형태의 신뢰 구간
    """
    data_size = len(data)
    means = np.zeros(n_resamples)
    
    start = time.time()
    
    for i in range(n_resamples):
        # 복원 추출로 리샘플링
        indices = np.random.randint(0, data_size, size=data_size)
        sample = data[indices]
        means[i] = np.mean(sample)
    
    # 백분위수 계산
    alpha = (100 - ci) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)
    
    execution_time = time.time() - start
    
    return lower_bound, upper_bound, execution_time

# 방법 2: NumPy 벡터화 부트스트랩
def bootstrap_ci_vectorized(data, n_resamples=10000, ci=95):
    """
    벡터화된 부트스트랩 방법으로 평균의 신뢰 구간 계산
    
    Parameters:
    -----------
    data : numpy.ndarray
        원본 1D 데이터 배열
    n_resamples : int
        부트스트랩 리샘플링 횟수
    ci : float
        신뢰 구간 수준 (0-100)
        
    Returns:
    --------
    tuple
        (하한, 상한) 형태의 신뢰 구간
    """
    data_size = len(data)
    
    start = time.time()
    
    # 벡터화된 리샘플링
    indices = np.random.randint(0, data_size, size=(n_resamples, data_size))
    samples = data[indices]
    means = np.mean(samples, axis=1)
    
    # 백분위수 계산
    alpha = (100 - ci) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)
    
    execution_time = time.time() - start
    
    return lower_bound, upper_bound, execution_time

# 방법 3: 병렬 처리 부트스트랩 (예시만 제공, 실제 병렬 처리는 구현하지 않음)
def bootstrap_ci_parallel(data, n_resamples=10000, ci=95):
    """
    병렬 처리를 활용한 부트스트랩 방법 (개념적 예시)
    
    실제 병렬 처리는 구현하지 않고, 벡터화된 방법과 동일하게 동작
    """
    return bootstrap_ci_vectorized(data, n_resamples, ci)

# 기본 부트스트랩 신뢰 구간 계산
ci_level = 95
n_resamples = 10000

print(f"\n{ci_level}% 신뢰 구간 계산 (부트스트랩 리샘플링 {n_resamples}회):")

lower1, upper1, time1 = bootstrap_ci_basic(X, n_resamples, ci_level)
print(f"방법 1 (기본 부트스트랩): [{lower1:.4f}, {upper1:.4f}], 실행 시간: {time1:.4f}초")

lower2, upper2, time2 = bootstrap_ci_vectorized(X, n_resamples, ci_level)
print(f"방법 2 (벡터화 부트스트랩): [{lower2:.4f}, {upper2:.4f}], 실행 시간: {time2:.4f}초")

lower3, upper3, time3 = bootstrap_ci_parallel(X, n_resamples, ci_level)
print(f"방법 3 (병렬 부트스트랩 개념): [{lower3:.4f}, {upper3:.4f}], 실행 시간: {time3:.4f}초")

# 이론적 신뢰 구간 (정규 분포 가정)
from scipy import stats

mean = np.mean(X)
std_err = stats.sem(X)  # 표준 오차
theoretical_ci = stats.t.interval(ci_level/100, len(X)-1, loc=mean, scale=std_err)

print(f"이론적 {ci_level}% 신뢰 구간 (t-분포): [{theoretical_ci[0]:.4f}, {theoretical_ci[1]:.4f}]")

# 성능 비교
print("\n성능 비교:")
print(f"기본 부트스트랩 vs 벡터화: {time1/time2:.2f}x 속도 향상")

# 부트스트랩 분포 시각화
plt.figure(figsize=(10, 6))

# 벡터화된 방법으로 부트스트랩 평균 분포 생성
indices = np.random.randint(0, len(X), size=(n_resamples, len(X)))
samples = X[indices]
bootstrap_means = np.mean(samples, axis=1)

plt.hist(bootstrap_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'샘플 평균: {mean:.4f}')
plt.axvline(lower2, color='green', linestyle='-', linewidth=2, label=f'{ci_level}% 신뢰 구간 하한: {lower2:.4f}')
plt.axvline(upper2, color='green', linestyle='-', linewidth=2, label=f'{ci_level}% 신뢰 구간 상한: {upper2:.4f}')

plt.title(f'부트스트랩 평균 분포 (리샘플링 {n_resamples}회)')
plt.xlabel('샘플 평균')
plt.ylabel('빈도')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.savefig('bootstrap_distribution.png', dpi=150)
# plt.show()

# 일반화된 부트스트랩 함수
def bootstrap_statistic_ci(data, statistic_func=np.mean, n_resamples=10000, ci=95):
    """
    임의의 통계량에 대한 부트스트랩 신뢰 구간 계산
    
    Parameters:
    -----------
    data : numpy.ndarray
        원본 1D 데이터 배열
    statistic_func : function
        신뢰 구간을 계산할 통계량 함수 (기본값: np.mean)
    n_resamples : int
        부트스트랩 리샘플링 횟수
    ci : float
        신뢰 구간 수준 (0-100)
        
    Returns:
    --------
    tuple
        (하한, 상한) 형태의 신뢰 구간
    """
    data_size = len(data)
    statistics = np.zeros(n_resamples)
    
    # 벡터화된 리샘플링
    indices = np.random.randint(0, data_size, size=(n_resamples, data_size))
    samples = data[indices]
    
    # 각 리샘플에 대해 통계량 계산
    for i in range(n_resamples):
        statistics[i] = statistic_func(samples[i])
    
    # 백분위수 계산
    alpha = (100 - ci) / 2
    lower_bound = np.percentile(statistics, alpha)
    upper_bound = np.percentile(statistics, 100 - alpha)
    
    return lower_bound, upper_bound

# 다양한 통계량에 대한 부트스트랩 신뢰 구간 계산
print("\n다양한 통계량에 대한 부트스트랩 신뢰 구간:")

# 평균
mean_ci = bootstrap_statistic_ci(X, np.mean)
print(f"평균 {ci_level}% 신뢰 구간: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")

# 중앙값
median_ci = bootstrap_statistic_ci(X, np.median)
print(f"중앙값 {ci_level}% 신뢰 구간: [{median_ci[0]:.4f}, {median_ci[1]:.4f}]")

# 표준편차
std_ci = bootstrap_statistic_ci(X, lambda x: np.std(x, ddof=1))
print(f"표준편차 {ci_level}% 신뢰 구간: [{std_ci[0]:.4f}, {std_ci[1]:.4f}]")

# 사용자 정의 통계량 (예: 절사평균)
trimmed_mean_ci = bootstrap_statistic_ci(X, lambda x: stats.trim_mean(x, 0.1))
print(f"10% 절사평균 {ci_level}% 신뢰 구간: [{trimmed_mean_ci[0]:.4f}, {trimmed_mean_ci[1]:.4f}]")
