import math
import numpy as np
import _erfa

def ab(pnat, v, s, bm1):
    return _erfa.ab(pnat, v, s, bm1)

def besselian_epoch_jd(epd):
    if not (type(epd) == np.ndarray):
        raise _erfa.error('argument is not of type of ndarray')
    if not epd.shape:
        raise _erfa.error('argument is ndarray of length 0')
    return _erfa.epb2jd(epd)

def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    if not (type(epd) == np.ndarray):
        raise _erfa.error('argument is not of type of ndarray')
    if not epd.shape:
        raise _erfa.error('argument is ndarray of length 0')
    return _erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

