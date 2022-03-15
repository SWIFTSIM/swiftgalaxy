import builtins
from typing import Union
import numpy as np

MaskType = Union[None, np.ndarray, slice, 'builtins.ellipsis']
