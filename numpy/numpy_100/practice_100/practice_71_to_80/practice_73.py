# 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles

import numpy as np

# 임의의 10개 삼각형 생성 (각 삼각형은 3개의 정점으로 구성)
# 각 삼각형은 (x,y) 좌표의 세 점으로 표현
triangles = np.random.randint(0, 10, (10, 3, 2))
print("삼각형 집합 (10개):")
print(triangles)

# 삼각형의 각 변 추출 (각 삼각형은 3개의 변을 가짐)
# 각 변은 두 점으로 이루어짐 (p1, p2)
edges = np.zeros((10*3, 2, 2))  # 10개 삼각형 * 3개 변 = 30개 변

for i in range(10):
    # 각 삼각형의 3개 변 추출
    # 변1: 정점 0-1, 변2: 정점 1-2, 변3: 정점 2-0
    edges[i*3] = triangles[i, [0, 1]]         # 첫 번째 변
    edges[i*3+1] = triangles[i, [1, 2]]       # 두 번째 변
    edges[i*3+2] = triangles[i, [2, 0]]       # 세 번째 변

print("\n모든 변 (30개):")
print(edges)

# 중복된 변 제거 (각 변을 정렬하여 같은 두 점을 가진 변을 찾음)
# 변을 문자열로 변환하여 집합으로 변환
unique_edges = set()

for edge in edges:
    # 두 점을 정렬하여 (작은 좌표, 큰 좌표) 순서로 저장
    # 이렇게 하면 (a,b)와 (b,a)를 같은 변으로 인식
    edge_tuple = tuple(map(tuple, np.sort(edge, axis=0)))
    unique_edges.add(edge_tuple)

print(f"\n고유한 변의 수: {len(unique_edges)}")
print("고유한 변 목록:")
for edge in unique_edges:
    print(edge)