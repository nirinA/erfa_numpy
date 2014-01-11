import math
import numpy as np
import _erfa

def check_args(*args):
    for t in args:
        if not (type(t) == np.ndarray):
            raise _erfa.error('arguments are not of type of ndarray')
        try:
            if t.shape[0] != args[0].shape[0]:
                raise _erfa.error('arguments have not the same shape')
        except IndexError:
            raise _erfa.error('cannot compute ndarray of length 0')

def ab(pnat, v, s, bm1):
    check_args(pnat, v, s, bm1)
    return _erfa.ab(pnat, v, s, bm1)

def apcs(date1, date2, pv, ebpv, ehp):
    check_args(date1, date2, pv, ebpv, ehp)
    return _erfa.apcs(date1, date2, pv, ebpv, ehp)

def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    check_args(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)
    return _erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

def besselian_epoch_jd(epd):
    check_args(epd)
    return _erfa.epb2jd(epd)

def s00(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s00(d1, d2, x, y)
