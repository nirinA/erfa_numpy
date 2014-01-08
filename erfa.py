import math
import numpy as np
import _erfa

def ab(pnat, v, s, bm1):
    for a in (pnat, v, s, bm1):
        if not (type(a) == np.ndarray):
            raise _erfa.error('argument is not of type ndarray')
        
    n = pnat.shape[0]
    if (v.shape[0] == n and
        s.shape[0] == n and
        bm1.shape[0] == n):
        return _erfa.ab(pnat, v, s, bm1)
    else:
        raise _erfa.error('incompatible shape for arguments')

def besselian_epoch_jd(epd):
    if not (type(epd) == np.ndarray):
        raise _erfa.error('argument is not of type of ndarray')
    if not epd.shape:
        raise _erfa.error('argument is ndarray of length 0')
    return _erfa.epb2jd(epd)

