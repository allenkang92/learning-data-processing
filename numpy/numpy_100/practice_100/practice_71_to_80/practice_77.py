# 77. How to negate a boolean, or to change the sign of a float inplace?

import numpy as np

arr1 = np.array([True, False, True])
arr2 = np.array([1.0, -2.0, 3.0])

# 부울 배열의 부호 변경
arr1 = ~arr1

# 부동 소수점 배열의 부호 변경
arr2 = -arr2

print(arr1)
# [False  True False]
print(arr2)
# [-1.  2. -3.]

# 개선사항:
# 1. 현재 방법은 in-place 연산이 아니라 새 배열을 생성함
# 2. 부울 배열의 진짜 in-place 부정 방법:
#    np.logical_not(arr1, out=arr1)  # 결과를 동일한 배열에 저장
#    또는
#    np.bitwise_not(arr1, out=arr1)  # 비트 연산으로 in-place 수행
# 3. 부동 소수점 배열의 진짜 in-place 부호 변경:
#    np.negative(arr2, out=arr2)  # 결과를 동일한 배열에 저장
# 4. 비교적 간단한 방법:
#    arr2 *= -1  # in-place 곱셈 연산
#    (이 방법은 arr2가 뷰(view)가 아닌 경우에만 실제로 in-place)
# 5. 대용량 배열에서는 in-place 연산이 메모리 효율성 측면에서 중요함
# 6. 주의: NumPy에서 ~ 연산자는 새 배열을 반환하므로 진짜 in-place가 아님
