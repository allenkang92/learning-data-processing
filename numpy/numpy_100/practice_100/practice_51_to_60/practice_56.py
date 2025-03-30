# 56. Generate a generic 2D Gaussian-like array

import numpy as np
import matplotlib.pyplot as plt

# 20x20 배열 생성
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
R = np.sqrt(X**2 + Y**2)
Z = np.exp(-R**2)

print("2D 가우시안 배열 모양:", Z.shape)
print("일부 데이터:")
print(Z[:5, :5])

# 시각화 (선택 사항)
plt.imshow(Z, cmap='viridis')
plt.colorbar()
plt.title("2D Gaussian")
# plt.show()  # 그래프를 표시하려면 주석 해제