# 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?

import numpy as np

# 두 점으로 정의된 여러 개의 직선 생성
# 각 행은 직선을 정의하는 두 점의 (x, y) 좌표
P0 = np.random.uniform(-10, 10, (5, 2))  # 5개 직선의 시작점
P1 = np.random.uniform(-10, 10, (5, 2))  # 5개 직선의 끝점

# 거리를 계산할 점
p = np.random.uniform(-10, 10, (2,))

print("직선의 시작점 P0:")
print(P0)
print("\n직선의 끝점 P1:")
print(P1)
print("\n점 p:", p)

# 방법 1: 반복문 사용
distances1 = np.zeros(len(P0))
for i in range(len(P0)):
    # 직선 벡터 계산
    v = P1[i] - P0[i]
    # 점 p에서 직선의 시작점까지의 벡터
    w = p - P0[i]
    # 직선 벡터의 제곱 길이
    c1 = np.dot(v, v)
    # 벡터 w와 v의 내적
    c2 = np.dot(w, v)
    
    # 점 p에서 직선까지의 최단 거리 계산
    # 직선 매개변수 t 계산 (0에서 1 사이로 클램핑)
    t = max(0, min(1, c2 / c1))
    # 직선 상의 가장 가까운 점 계산
    projection = P0[i] + t * v
    # 점 p와 투영점 사이의 거리 계산
    distances1[i] = np.linalg.norm(p - projection)

print("\n방법 1 (반복문) - 각 직선까지의 거리:")
print(distances1)

# 방법 2: 벡터화된 연산 사용
# 방향 벡터
v = P1 - P0
# 점 p에서 각 직선의 시작점까지의 벡터
w = p - P0
# 내적 계산을 위한 행렬 곱
c1 = np.sum(v * v, axis=1)
c2 = np.sum(w * v, axis=1)
# 매개변수 t 계산
t = np.clip(c2 / c1, 0, 1)
# t 값을 열 벡터로 변환하여 브로드캐스팅
t = t.reshape(-1, 1)
# 직선 상의 가장 가까운 점 계산
projection = P0 + t * v
# 거리 계산
distances2 = np.sqrt(np.sum((p - projection) ** 2, axis=1))

print("\n방법 2 (벡터화) - 각 직선까지의 거리:")
print(distances2)
print("\n두 방법의 결과가 동일함:", np.allclose(distances1, distances2))
