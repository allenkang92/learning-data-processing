
import numpy as np

rg = np.random.default_rng(1)

a = rg.random((2, 3))

print(a)
# [[0.51182162 0.9504637  0.14415961]
#  [0.94864945 0.31183145 0.42332645]]

print(a.sum())
# 3.290252281866131

print(a.min())
# 0.14415961271963373

print(a.max())
# 0.9504636963259353
