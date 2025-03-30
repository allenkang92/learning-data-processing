
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
# (6,)  

a2 = a[np.newaxis, :]
print(a2.shape)
# (1, 6)

row_vector = a[np.newaxis, :]
print(row_vector.shape)
# (1, 6)

col_vector = a[:, np.newaxis]
print(col_vector.shape)
# (6, 1)

a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
# (6,)

b = np.expand_dims(a, axis = 1)
print(b.shape)
# (6, 1)

c = np.expand_dims(a, axis = 0)
print(c.shape)
# (1, 6)