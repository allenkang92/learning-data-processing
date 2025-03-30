# 17. What is the result of the following expression? (★☆☆)

import numpy as np

# nan은 not a number
# 1. 0 * np.nan
# -> nan.
print(0 * np.nan)
# nan

# 2. np.nan == np.nan
# -> True인가?
print(np.nan == np.nan)
# False, 틀렸다.

# 3. np.inf > np.nan
# -> 모르겠다.
print(np.inf > np.nan)
# False, 틀렸다.

# 4. np.nan - np.nan
# -> 계산 불가? 아니면 nan?
print(np.nan - np.nan)
# nan

# 5. np.nan in set([np.nan])
# -> nan은 set에 포함되지 않을 것 같다.
print(np.nan in set([np.nan]))
# True, 틀렸다.

# 6. 0.3 == 3 * 0.1
# -> False일 거 같다.
print(0.3 == 3 * 0.1)
# False
