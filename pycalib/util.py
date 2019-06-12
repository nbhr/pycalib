import numpy as np

def is_shape(a, sz):
    sz0 = a.shape
    if len(sz0) != sz:
        return False
    for i in len(sz0):
        if sz[i] != -1 and sz[i] != sz0[i]:
            return False
    return True

def transpose_to_col(a, m):
    if a.ndim == 1:
        return a.reshape((m, 1))
    
    assert a.ndim == 2, "a is not 2-dim"
    if a.shape[1] == m:
        return a
    else:
        assert a.shape[0] == m, "a is not 2x? or ?x2"
        return a.T