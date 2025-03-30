# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

import numpy as np

# 문제 설명: Z[i,j] == Z[j,i] 조건을 만족하는 2D 배열 서브클래스 생성
# 이는 대칭 행렬(symmetric matrix)의 특성을 가지도록 하는 것입니다.

# 방법: ndarray를 상속하는 SymmetricArray 클래스 구현
class SymmetricArray(np.ndarray):
    def __new__(cls, input_array):
        # 입력 배열이 2D인지 확인
        input_array = np.asarray(input_array)
        if input_array.ndim != 2:
            raise ValueError("입력 배열은 2차원이어야 합니다.")
        
        # 정사각형 행렬인지 확인
        if input_array.shape[0] != input_array.shape[1]:
            raise ValueError("입력 배열은 정사각형이어야 합니다.")
        
        # 대칭 행렬로 변환 (하삼각행렬 + 상삼각행렬의 전치) / 2
        symmetric = (input_array + input_array.T) / 2
        
        # 대칭 행렬을 ndarray의 인스턴스로 변환
        obj = symmetric.view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
    
    def __setitem__(self, index, value):
        # 인덱스가 튜플이고 길이가 2인지 확인 (2D 배열 인덱싱)
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            # 대각선이 아닌 경우, 대칭성을 유지하기 위해 반대쪽도 설정
            if i != j:
                super(SymmetricArray, self).__setitem__((j, i), value)
        # 항상 원래 위치도 설정
        super(SymmetricArray, self).__setitem__(index, value)

# 테스트 1: 임의의 배열로 SymmetricArray 생성
print("테스트 1: 임의의 배열로 SymmetricArray 생성")
A = np.random.randint(0, 10, (4, 4))
print("원본 배열:")
print(A)

S = SymmetricArray(A)
print("\n대칭 배열 (생성자 사용):")
print(S)

# 테스트 2: 값 설정 시 대칭성 유지 확인
print("\n테스트 2: 값 설정 시 대칭성 유지 확인")
S[0, 1] = 100
print("S[0, 1] = 100 설정 후:")
print(S)
print("S[1, 0]의 값:", S[1, 0])  # S[0, 1]과 동일한 값이어야 함

# 테스트 3: 산술 연산 후에도 대칭성 유지 확인
print("\n테스트 3: 산술 연산 후에도 대칭성 유지 확인")
S2 = S * 2
print("S * 2:")
print(S2)
print("S2의 타입:", type(S2))  # SymmetricArray 타입이 아닐 수 있음

# NumPy의 기본 연산은 결과를 SymmetricArray로 반환하지 않을 수 있으므로
# 필요한 경우 명시적으로 변환해야 함
S2_symmetric = SymmetricArray(S2)
print("\n명시적으로 다시 SymmetricArray로 변환한 결과:")
print(S2_symmetric)

# 대칭 행렬의 특성 확인: 항상 Z == Z.T
print("\n대칭 행렬의 특성 확인: Z == Z.T")
print("S == S.T:", np.array_equal(S, S.T))
print("S2_symmetric == S2_symmetric.T:", np.array_equal(S2_symmetric, S2_symmetric.T))