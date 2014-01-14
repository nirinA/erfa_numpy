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
        
def epb2jd(epd):
    check_args(epd)
    return _erfa.epb2jd(epd)

besselian_epoch_jd = epb2jd

def epb(jd1, jd2):
    check_args(jd1, jd2)
    return _erfa.epb(jd1, jd2)

jd_besselian_epoch = epb

def epj(d1, d2):
    check_args(d1, d2)
    return _erfa.epj(d1, d2)

jd_julian_epoch = epj

def epj2jd(epj):
    check_args(epj)
    return _erfa.epj2jd(epj)

julian_epoch_jd = epj2jd

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

## Astronomy/SpaceMotion 
def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    check_args(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)
    return _erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

## Astronomy/Timescales
def dat(iy, im, id, fd):
    check_args(iy, im, id, fd)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    return _erfa.dat(iy, im, id, fd)

d_tai_utc = dat


def d2dtf(scale, ndp, d1, d2):
    if scale.lower() not in ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc'):
        raise _erfa.error('unknown time scale: %s'%scale)
    check_args(d1, d2)
    return _erfa.d2dtf(scale, ndp, d1, d2)

jd_dtf = d2dtf

def dtf2d(scale, iy, im, id, ihr, imn, sec):   
    if scale.lower() not in ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc'):
        raise _erfa.error('unknown time scale: %s'%scale)
    check_args(iy, im, id, ihr, imn, sec)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    ihr = cast_to_int32(ihr)
    imn = cast_to_int32(imn)
    return _erfa.dtf2d(scale, iy, im, id, ihr, imn, sec)

dtf_jd = dtf2d

def taitt(tai1, tai2):
    check_args(tai1, tai2)
    return _erfa.taitt(tai1, tai2)

tai_tt = taitt

def taiut1(tai1, tai2, dta):
    check_args(tai1, tai2, dta)
    return _erfa.taiut1(tai1, tai2, dta)

tai_ut1 = taiut1

def taiutc(tai1, tai2):
    check_args(tai1, tai2)
    return _erfa.taiutc(tai1, tai2)

tai_utc = taiutc

def tcbtdb(tcb1, tcb2):
    check_args(tcb1, tcb2)
    return _erfa.tcbtdb(tcb1, tcb2)

tcb_tdb = tcbtdb

def tcgtt(tcb1, tcb2):
    check_args(tcb1, tcb2)
    return _erfa.tcgtt(tcb1, tcb2)

tcg_tt = tcgtt

def tdbtcb(tdb1, tdb2):
    check_args(tdb1, tdb2)
    return _erfa.tdbtcb(tdb1, tdb2)

tdb_tcb = tdbtcb

def tttai(tt1, tt2):
    check_args(tt1, tt2)
    return _erfa.tttai(tt1, tt2)

tt_tai = tttai

def tttcg(tt1, tt2):
    check_args(tt1, tt2)
    return _erfa.tttcg(tt1, tt2)

tt_tcg = tttcg

def ttut1(tt1, tt2, dt):
    check_args(tt1, tt2, dt)
    return _erfa.ttut1(tt1, tt2, dt)

tt_ut1 = ttut1

def ut1tai(ut11, ut12, dta):
    check_args(ut11, ut12, dta)
    return _erfa.ut1tai(ut11, ut12, dta)

ut1_tai = ut1tai

def ut1tt(ut11, ut12, dt):
    check_args(ut11, ut12, dt)
    return _erfa.ut1tt(ut11, ut12, dt)

ut1_tt = ut1tt

def utctai(utc1, utc2):
    check_args(utc1, utc2)
    return _erfa.utctai(utc1, utc2)

utc_tai = utctai

## Astronomy/PrecNutPolar
def c2ixys(x, y, s):
    check_args(x, y, s)
    return _erfa.c2ixys(x, y, s)

def numat(epsa, dpsi, deps):
    check_args(epsa, dpsi, deps)
    return _erfa.numat(epsa, dpsi, deps)

def nut00a(d1, d2):
    check_args(d1, d2)
    return _erfa.nut00a(d1, d2)

def nut80(d1, d2):
    check_args(d1, d2)
    return _erfa.nut80(d1, d2)

def obl80(d1, d2):
    check_args(d1, d2)
    return _erfa.obl80(d1, d2)

def plan94(d1, d2, np):
    check_args(d1, d2)
    return _erfa.plan94(d1, d2, np)

def pmat76(d1, d2):
    check_args(d1, d2)
    return _erfa.pmat76(d1, d2)

def pn00(d1,d2,dpsi,deps):
    check_args(d1,d2,dpsi,deps)
    return _erfa.pn00(d1,d2,dpsi,deps)

def pom00(xp, yp, sp):
    check_args(xp, yp, sp)
    return _erfa.pom00(xp, yp, sp)

def s00(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s00(d1, d2, x, y)

def s06(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s06(d1, d2, x, y)

def sp00(d1, d2):
    check_args(d1, d2)
    return _erfa.sp00(d1, d2)

def xy06(d1, d2):
    check_args(d1, d2)
    return _erfa.xy06(d1, d2)

def xys00a(d1, d2):
    check_args(d1, d2)
    return _erfa.xys00a(d1, d2)

def xys06a(d1, d2):
    check_args(d1, d2)
    return _erfa.xys06a(d1, d2)

## Astronomy/RotationAndTime
def ee00(d1, d2, epsa, dpsi):
    check_args(d1, d2, epsa, dpsi)
    return _erfa.ee00(d1, d2, epsa, dpsi)

def eqeq94(d1, d2):
    check_args(d1, d2)
    return _erfa.eqeq94(d1, d2)

def era00(d1, d2):
    check_args(d1, d2)
    return _erfa.era00(d1, d2)

def gmst00(uta, utb, tta, ttb):   
    check_args(uta, utb, tta, ttb)
    return _erfa.gmst00(uta, utb, tta, ttb)

def gmst82(d1, d2):
    check_args(d1, d2)
    return _erfa.gmst82(d1, d2)

## VectorMatrix/AngleOps
def a2af(n, a):
    check_args(a)
    return _erfa.a2af(n, a)

def a2tf(n, a):
    check_args(a)
    return _erfa.a2tf(n, a)

## VectorMatrix/BuildRotations 
def rx(phi, r):
    check_args(phi, r)
    return _erfa.rx(phi, r)

def rxp(r, p):
    check_args(r, p)
    return _erfa.rxp(r, p)

def ry(theta, r):
    check_args(theta, r)
    return _erfa.ry(theta, r)

def rz(psi, r):
    check_args(psi, r)
    return _erfa.rz(psi, r)

## VectorMatrix/CopyExtendExtract
def cr(a):
    check_args(a)
    return _erfa.cr(a)

## VectorMatrix/MatrixOps
def anp(a):
    check_args(a)
    return _erfa.anp(a)

def rxr(a, b):
    check_args(a, b)
    return _erfa.rxr(a, b)

__all__ = ['anp', 'rxr']
for a in __all__:
    setattr(eval(a), '__doc__',  getattr(_erfa, a).__doc__)
