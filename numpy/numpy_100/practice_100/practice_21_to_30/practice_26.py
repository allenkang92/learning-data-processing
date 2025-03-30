# 26. What is the output of the following script?


# print(sum(range(5),-1))
# -> 0부터 4까지 합. 0, 1, 2, 3, 4
print(sum(range(5)))
# 10
print(sum(range(5),-1))
# 9

from numpy import *
print(sum(range(5),-1))
# 10

# 왜 다르지..?