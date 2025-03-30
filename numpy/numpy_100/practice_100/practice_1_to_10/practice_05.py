# 5. How to get the documentation of the numpy add function from the command line?

import numpy as np

print(help(np.add))
# Help on ufunc:

# add = <ufunc 'add'>
#     add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
#     Add arguments element-wise.
    
#     Parameters
#     ----------
#     x1, x2 : array_like
#         The arrays to be added.
#         If ``x1.shape != x2.shape``, they must be broadcastable to a common