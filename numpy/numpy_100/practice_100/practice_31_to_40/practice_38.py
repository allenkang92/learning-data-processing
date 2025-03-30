# 38. Consider a generator function that generates 10 integers and use it to build an array

import numpy as np

arr = np.array((_ for _ in range(10 + 1)))
print(arr)
# [ 0  1  2  3  4  5  6  7  8  9 10]

print(type(arr))
# <generator object <genexpr> at 0x10e329c10>
# <class 'numpy.ndarray'>


# def generate():
#     for x in range(10):
#         yield x
# Z = np.fromiter(generate(),dtype=float,count=-1)
# print(Z)