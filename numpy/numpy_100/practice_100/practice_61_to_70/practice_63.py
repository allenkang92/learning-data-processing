# 63. Create an array class that has a name attribute

import numpy as np

class NamedArray(np.ndarray):
    def __new__(cls, array, name="no_name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "no_name")

# 테스트
a = NamedArray(np.arange(10), "range_array")
print(a.name)
print(a)

# 슬라이싱 시 이름 유지 확인
b = a[1:3]
print(b.name)
print(b)
