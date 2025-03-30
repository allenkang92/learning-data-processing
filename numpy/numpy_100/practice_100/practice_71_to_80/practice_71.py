# 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)?

import numpy as np

# (5,5,3) 크기의 배열 생성
arr1 = np.arange(5 * 5 * 3).reshape((5, 5, 3))
print("원본 3D 배열 (5,5,3):")
print(arr1)

# (5,5) 크기의 배열 생성
arr2 = np.arange(5 * 5).reshape((5, 5))
print("\n2D 배열 (5,5):")
print(arr2)

# 문제: 5x5 배열을 5x5x3 배열과 곱하기
# 오류: 단순히 flatten한 후 곱하면 크기가 맞지 않음
# result = np.dot(arr1.flatten(), arr2.flatten())  # ValueError: shapes (75,) and (25,) not aligned

# 올바른 해결책:
# 방법 1: 브로드캐스팅 사용 - 각 채널별로 2D 배열과 곱하기
# arr2를 (5,5,1) 형태로 확장하여 브로드캐스팅
result1 = arr1 * arr2[:, :, np.newaxis]
print("\n방법 1 (브로드캐스팅) 결과 (5,5,3):")
print(result1.shape)  # 출력 형태 확인
print(result1[:2, :2])  # 일부 데이터만 표시

# 방법 2: einsum 사용 - 더 명시적이고 유연한 방법
# ij,ijk->ijk 표기법은 첫 두 차원(i,j)에 대해 곱셈을 수행
result2 = np.einsum('ij,ijk->ijk', arr2, arr1)
print("\n방법 2 (einsum) 결과 (5,5,3):")
print(result2.shape)  # 출력 형태 확인
print(result2[:2, :2])  # 일부 데이터만 표시

# 두 방법의 결과가 동일한지 확인
print("\n두 방법의 결과가 동일함:", np.array_equal(result1, result2))

# 방법 3: 각 채널에 대해 별도로 곱셈 수행
result3 = np.zeros_like(arr1)
for i in range(arr1.shape[2]):  # 각 채널에 대해 반복
    result3[:, :, i] = arr1[:, :, i] * arr2
print("\n방법 3 (반복문) 결과 (5,5,3):")
print(result3.shape)

# 방법 비교 표: 효율성과 가독성
print("\n다양한 방법 비교:")
print("=" * 50)
print("| 방법                | 효율성    | 가독성    | 유연성    |")
print("|---------------------|-----------|-----------|-----------|")
print("| 브로드캐스팅        | 높음      | 중간      | 중간      |")
print("| einsum              | 높음      | 높음      | 높음      |")
print("| 반복문              | 낮음      | 높음      | 높음      |")
print("=" * 50)

# 실제 응용 사례: 이미지 처리
print("\n실제 응용 사례: 이미지 처리")
print("이 연산은 각 픽셀의 RGB 값에 가중치를 적용할 때 유용합니다.")
print("예: 이미지 밝기 조정, 색상 변환, 필터 적용 등")