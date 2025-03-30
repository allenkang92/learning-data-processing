# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

import numpy as np

X = np.array([1, 2, 3])
print("X:", X)

Y = np.array([4, 5, 6])
print("Y:", Y)

# Cauchy 행렬 구성: Cij = 1/(xi - yj)
C = 1.0 / (X.reshape(-1, 1) - Y)
print("Cauchy 행렬 C:")
print(C)