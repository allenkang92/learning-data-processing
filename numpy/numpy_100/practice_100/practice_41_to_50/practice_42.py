# 42. Consider two random array A and B, check if they are equal


import numpy as np

# 두 랜덤 배열 생성
A = np.random.randint(0, 2, 5)
print(A)
#

B = np.random.randint(0, 2, 5)
print(B)
#

print(A == B)
# [False False False  True False]

print((A == B).all())
# False


# array_equal 함수 사용 (권장)
print("np.array_equal:", np.array_equal(A, B))

# array_equiv 함수 사용 (브로드캐스팅을 고려함)
print("np.array_equiv:", np.array_equiv(A, B))