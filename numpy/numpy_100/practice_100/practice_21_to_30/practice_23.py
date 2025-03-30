# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)

import numpy as np

color_dtype = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
color_arr = np.array([(255, 0, 0, 255), (0, 255, 0, 255)], dtype=color_dtype)
print(color_arr)
# [(255, 0, 0, 255) (0, 255, 0, 255)]