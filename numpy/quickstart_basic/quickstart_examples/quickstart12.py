
import numpy as np

rg = np.random.default_rng(1) # create instance of default random number generator

a = np.ones((2, 3), dtype = int)

b = rg.random((2, 3))

a *= 3
print(a)
# [[3 3 3]
#  [3 3 3]]

b += a
print(b)
# [[3.51182162 3.9504637  3.14415961]
#  [3.94864945 3.31183145 3.42332645]]

a += b # b is not automatically converted to integer type
print(a)
# line 20, in <module>
#     a += b # b is not automatically converted to integer type
# numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'