# 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?

import numpy as np

# 여러 직선 정의 (시작점과 끝점)
P0 = np.random.uniform(-10, 10, (3, 2))  # 3개 직선의 시작점
P1 = np.random.uniform(-10, 10, (3, 2))  # 3개 직선의 끝점

# 여러 점 정의
P = np.random.uniform(-10, 10, (4, 2))   # 4개의 점

print("직선 시작점 P0:")
print(P0)
print("\n직선 끝점 P1:")
print(P1)
print("\n점들 P:")
print(P)

# 모든 점에서 모든 직선까지의 거리 계산
# 결과는 (점 수, 직선 수) 형태의 배열이 됨

# 방법 1: 이중 반복문 사용
n_points = len(P)
n_lines = len(P0)
distances1 = np.zeros((n_points, n_lines))

for j in range(n_points):
    for i in range(n_lines):
        # 직선 벡터 계산
        v = P1[i] - P0[i]
        # 점에서 직선의 시작점까지의 벡터
        w = P[j] - P0[i]
        # 직선 벡터의 제곱 길이
        c1 = np.dot(v, v)
        # 벡터 w와 v의 내적
        c2 = np.dot(w, v)
        
        # 점에서 직선까지의 최단 거리 계산
        t = max(0, min(1, c2 / c1))
        projection = P0[i] + t * v
        distances1[j, i] = np.linalg.norm(P[j] - projection)

print("\n방법 1 (이중 반복문) - 각 점에서 각 직선까지의 거리:")
print(distances1)

# 방법 2: 벡터화된 연산 사용
# 차원 확장을 통한 브로드캐스팅 사용
# P0, P1을 (1, n_lines, 2) 형태로, P를 (n_points, 1, 2) 형태로 확장

# 직선 벡터 계산 (n_lines, 2)
v = P1 - P0

# 직선 벡터 제곱 길이 (n_lines,)
c1 = np.sum(v**2, axis=1)

# 모든 점-직선 조합에 대한 계산을 위해 차원 확장
P0_expanded = P0.reshape(1, n_lines, 2)    # (1, n_lines, 2)
v_expanded = v.reshape(1, n_lines, 2)      # (1, n_lines, 2)
P_expanded = P.reshape(n_points, 1, 2)     # (n_points, 1, 2)

# 점에서 직선 시작점까지의 벡터 (n_points, n_lines, 2)
w = P_expanded - P0_expanded

# w와 v의 내적 계산 (n_points, n_lines)
c2 = np.sum(w * v_expanded, axis=2)

# 매개변수 t 계산 (n_points, n_lines)
t = np.clip(c2 / c1.reshape(1, n_lines), 0, 1)

# 직선 상의 가장 가까운 점 계산 (n_points, n_lines, 2)
projection = P0_expanded + t.reshape(n_points, n_lines, 1) * v_expanded

# 거리 계산 (n_points, n_lines)
distances2 = np.sqrt(np.sum((P_expanded - projection)**2, axis=2))

print("\n방법 2 (벡터화) - 각 점에서 각 직선까지의 거리:")
print(distances2)
print("\n두 방법의 결과가 동일함:", np.allclose(distances1, distances2))
