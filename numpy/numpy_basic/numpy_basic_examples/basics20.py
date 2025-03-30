


import numpy as np

print(np.ones(3))
# [1. 1. 1.]

print(np.zeros(3))
# [0. 0. 0.]

rng = np.random.default_rng() # the simplest way to generate random numbers
print(rng.random(3))  
# [0.23326758 0.79946408 0.49052666]


print(np.ones((3, 2)))
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

print(np.zeros((3, 2)))
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]]

print(rng.random((3, 2)))
# [[0.27245449 0.17107541]
#  [0.66332537 0.13205482]
#  [0.09622077 0.95204348]]