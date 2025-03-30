# 31. How to ignore all numpy warnings (not recommended)?

import numpy as np

# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# Equivalently with a context manager
with np.errstate(all="ignore"):
    np.arange(3) / 0