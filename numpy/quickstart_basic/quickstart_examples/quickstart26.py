

import numpy as np

a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

b = a # no new object is created

print(b is a) # a and b are two names for the same ndarray object
# True

def f(x):
    print(id(x))

print(id(a)) # id is a unique identifier of an object
# 4467141104

print(f(a))
# 4467141104

c = a.view()
print(c is a)
# False

print(c.base is a) # c is a view of the data owned by a
# True

print(c.flags.owndata)
# False

c = c.reshape((2, 6)) # a's shape doesn't change, reassigned c is still a view of a
print(a.shape)
# (3, 4)

c[0, 4] = 1234 # a's data changes
print(a)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]

s = a[:, 1:3]
s[:] = 10 # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]

d = a.copy() # a new array object with new data is created
print(d is a)
# False

print(d.base is a) # d doesn't share anything with a 
# False

d[0, 0] = 9999

print(d)
# [[9999   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]

print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]