# 48. Print the minimum and maximum representable value for each numpy scalar type

import numpy as np

# 일반적인 NumPy 데이터 타입 목록
scalar_types = [np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64,
               np.float16, np.float32, np.float64]

print("NumPy 스칼라 타입의 최소/최대 표현 가능 값:")
print("-" * 50)

for dtype in scalar_types:
    type_info = np.iinfo if np.issubdtype(dtype, np.integer) else np.finfo
    try:
        info = type_info(dtype)
        print(f"{dtype.__name__:10} - 최소값: {info.min:20} - 최대값: {info.max:20}")
    except:
        print(f"{dtype.__name__:10} - 정보를 얻을 수 없습니다.")