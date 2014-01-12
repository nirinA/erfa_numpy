import math
import numpy as np
import _erfa

def check_args(*args):
    for t in args:
        if not (type(t) == np.ndarray):
            raise _erfa.error('arguments are not of type of ndarray')
        try:
            if t.shape[0] != args[0].shape[0]:
                raise _erfa.error('shape of arguments are not compatible')
        except IndexError:
            raise _erfa.error('cannot compute ndarray of length 0')

def cast_to_int32(i):
    return np.array([n for n in i], dtype='int32')

## ASTROM and LDBODY struct
ASTROM = _erfa.ASTROM

## Astronomy/Calendars
def cal2jd(iy, im, id):
    check_args(iy, im, id)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    return _erfa.cal2jd(iy, im, id)
        
def besselian_epoch_jd(epd):
    check_args(epd)
    return _erfa.epb2jd(epd)

## Astronomy/Astrometry 
def ab(pnat, v, s, bm1):
    check_args(pnat, v, s, bm1)
    return _erfa.ab(pnat, v, s, bm1)

def apcs(date1, date2, pv, ebpv, ehp):
    check_args(date1, date2, pv, ebpv, ehp)
    return _erfa.apcs(date1, date2, pv, ebpv, ehp)

def ld(bm, p, q, e, em, dlim):
    check_args(bm, p, q, e, em, dlim)
    return _erfa.ld(bm, p, q, e, em, dlim)

def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    check_args(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)
    return _erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

## Astronomy/Timescales
def d_tai_utc(iy, im, id, fd):
    check_args(iy, im, id, fd)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    return _erfa.dat(iy, im, id, fd)

## Astronomy/PrecNutPolar
def numat(epsa, dpsi, deps):
    check_args(epsa, dpsi, deps)
    return _erfa.numat(epsa, dpsi, deps)

def nut80(d1, d2):
    check_args(d1, d2)
    return _erfa.nut80(d1, d2)

def obl80(d1, d2):
    check_args(d1, d2)
    return _erfa.obl80(d1, d2)

def plan94():
    pass

def pmat76(d1, d2):
    check_args(d1, d2)
    return _erfa.pmat76(d1, d2)

def s00(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s00(d1, d2, x, y)

def xys06a(d1, d2):
    check_args(d1, d2)
    return _erfa.xys06a(d1, d2)

## VectorMatrix/MatrixOps
def rxr(a, b):
    check_args(a, b)
    return _erfa.rxr(a, b)
