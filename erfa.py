'''ERFA wrapper
note:
    af2a has different form of arguments
'''
import math
import numpy as np
import _erfa
from _erfa import AULT , CMPS , D2PI , DAS2R , DAU ,\
     DAYSEC , DC , DD2R , DJ00 , DJC , DJM , DJM0 ,\
     DJM00 , DJM77 , DJY , DMAS2R , DPI , DR2AS ,\
     DR2D , DS2R , DTY , ELB , ELG , GRS80 , SRS , \
     TDB0 , TTMTAI , TURNAS , WGS72 , WGS84, \
     ASTROM, LDBODY, \
     bi00, eform, fk5hip,\
     error

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

## Astronomy/Calendars
def cal2jd(iy, im, id):
    check_args(iy, im, id)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    return _erfa.cal2jd(iy, im, id)
        
def epb(jd1, jd2):
    check_args(jd1, jd2)
    return _erfa.epb(jd1, jd2)

jd_besselian_epoch = epb

def epb2jd(epd):
    check_args(epd)
    return _erfa.epb2jd(epd)

besselian_epoch_jd = epb2jd

def epj(d1, d2):
    check_args(d1, d2)
    return _erfa.epj(d1, d2)

jd_julian_epoch = epj

def epj2jd(epj):
    check_args(epj)
    return _erfa.epj2jd(epj)

julian_epoch_jd = epj2jd

def jd2cal(d1, d2):
    check_args(d1, d2)
    return _erfa.jd2cal(d1, d2)

def jdcalf(ndp, d1, d2):
    check_args(d1, d2)
    return _erfa.jdcalf(ndp, d1, d2)

## Astronomy/Astrometry 
def ab(pnat, v, s, bm1):
    check_args(pnat, v, s, bm1)
    return _erfa.ab(pnat, v, s, bm1)

def apcg(date1, date2, ebpv, ehp):
    check_args(date1, date2, ebpv, ehp)
    return _erfa.apcg(date1, date2, ebpv, ehp)

def apcg13(date1, date2):
    check_args(date1, date2)
    return _erfa.apcg13(date1, date2)

def apci(date1, date2, ebpv, ehp, x, y, s):
    check_args(date1, date2, ebpv, ehp, x, y, s)
    return _erfa.apci(date1, date2, ebpv, ehp, x, y, s)

def apci13(date1, date2):
    check_args(date1, date2)
    return _erfa.apci13(date1, date2)

def apco(date1, date2, ebpv, ehp, x, y, s,theta,elong, phi, hm,xp, yp, sp, refa, refb):
    check_args(date1, date2, ebpv, ehp, x, y, s,theta,elong, phi, hm,xp, yp, sp, refa, refb)
    return _erfa.apco(date1, date2, ebpv, ehp, x, y, s,theta,elong, phi, hm,xp, yp, sp, refa, refb)

def apcs(date1, date2, pv, ebpv, ehp):
    check_args(date1, date2, pv, ebpv, ehp)
    return _erfa.apcs(date1, date2, pv, ebpv, ehp)

def ld(bm, p, q, e, em, dlim):
    check_args(bm, p, q, e, em, dlim)
    return _erfa.ld(bm, p, q, e, em, dlim)

## Astronomy/Ephemerides
def epv00(d1,d2):
    check_args(d1,d2)
    return _erfa.epv00(d1,d2)

def plan94(d1, d2, np):
    check_args(d1, d2)
    return _erfa.plan94(d1, d2, np)

## Astronomy/FundamentalArgs
def fad03(t):
    check_args(t)
    return _erfa.fad03(t)

def fae03(t):
    check_args(t)
    return _erfa.fae03(t)

def faf03(t):
    check_args(t)
    return _erfa.faf03(t)

def faju03(t):
    check_args(t)
    return _erfa.faju03(t)

def fal03(t):
    check_args(t)
    return _erfa.fal03(t)

def falp03(t):
    check_args(t)
    return _erfa.falp03(t)

def fama03(t):
    check_args(t)
    return _erfa.fama03(t)

def fame03(t):
    check_args(t)
    return _erfa.fame03(t)

def fane03(t):
    check_args(t)
    return _erfa.fane03(t)

def faom03(t):
    check_args(t)
    return _erfa.faom03(t)

def fapa03(t):
    check_args(t)
    return _erfa.fapa03(t)

def fasa03(t):
    check_args(t)
    return _erfa.fasa03(t)

def faur03(t):
    check_args(t)
    return _erfa.faur03(t)

def fave03(t):
    check_args(t)
    return _erfa.fave03(t)

## Astronomy/PrecNutPolar
def bp00(d1, d2):
    check_args(d1, d2)
    return _erfa.bp00(d1, d2)

def bp06(d1, d2):
    check_args(d1, d2)
    return _erfa.bp06(d1, d2)

def bpn2xy(rbpn):
    check_args(rbpn)
    return _erfa.bpn2xy(rbpn)

def c2i00a(d1, d2):
    check_args(d1, d2)
    return _erfa.c2i00a(d1, d2)

def c2i00b(d1, d2):
    check_args(d1, d2)
    return _erfa.c2i00b(d1, d2)

def c2i06a(d1, d2):
    check_args(d1, d2)
    return _erfa.c2i06a(d1, d2)

def c2ibpn(d1, d2, rbpn):
    check_args(d1, d2, rbpn)
    return _erfa.c2ibpn(d1, d2, rbpn)

def c2ixy(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.c2ixy(d1, d2, x, y)

def c2ixys(x, y, s):
    check_args(x, y, s)
    return _erfa.c2ixys(x, y, s)

def c2t00a(tta, ttb, uta, utb, xp, yp):
    check_args(tta, ttb, uta, utb, xp, yp)
    return _erfa.c2t00a(tta, ttb, uta, utb, xp, yp)

def c2t00b(tta, ttb, uta, utb, xp, yp):
    check_args(tta, ttb, uta, utb, xp, yp)
    return _erfa.c2t00b(tta, ttb, uta, utb, xp, yp)

def c2t06a(tta, ttb, uta, utb, xp, yp):
    check_args(tta, ttb, uta, utb, xp, yp)
    return _erfa.c2t06a(tta, ttb, uta, utb, xp, yp)

def c2tcio(rc2i, era, rpom):
    check_args(rc2i, era, rpom)
    return _erfa.c2tcio(rc2i, era, rpom)

def c2teqx(rc2i, era, rpom):
    check_args(rc2i, era, rpom)
    return _erfa.c2teqx(rc2i, era, rpom)

def c2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp):
    check_args(tta, ttb, uta, utb, dpsi, deps, xp, yp)
    return _erfa.c2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp)

def c2txy(tta, ttb, uta, utb, x, y, xp, yp):
    check_args(tta, ttb, uta, utb, x, y, xp, yp)
    return _erfa.c2txy(tta, ttb, uta, utb, x, y, xp, yp)

def eo06a(d1, d2):
    check_args(d1, d2)
    return _erfa.eo06a(d1, d2)

def eors(rnpb, s):
    check_args(rnpb, s)
    return _erfa.eors(rnpb, s)

def fw2m(gamb, phib, psi, eps):
    check_args(gamb, phib, psi, eps)
    return _erfa.fw2m(gamb, phib, psi, eps)

def fw2xy(gamb, phib, psi, eps):
    check_args(gamb, phib, psi, eps)
    return _erfa.fw2xy(gamb, phib, psi, eps)

def num00a(d1, d2):
    check_args(d1, d2)
    return _erfa.num00a(d1, d2)

def num00b(d1, d2):
    check_args(d1, d2)
    return _erfa.num00b(d1, d2)

def num06a(d1, d2):
    check_args(d1, d2)
    return _erfa.num06a(d1, d2)

def numat(epsa, dpsi, deps):
    check_args(epsa, dpsi, deps)
    return _erfa.numat(epsa, dpsi, deps)

def nut00a(d1, d2):
    check_args(d1, d2)
    return _erfa.nut00a(d1, d2)

def nut00b(d1, d2):
    check_args(d1, d2)
    return _erfa.nut00b(d1, d2)

def nut06a(d1, d2):
    check_args(d1, d2)
    return _erfa.nut06a(d1, d2)

def nut80(d1, d2):
    check_args(d1, d2)
    return _erfa.nut80(d1, d2)

def nutm80(d1, d2):
    check_args(d1, d2)
    return _erfa.nutm80(d1, d2)

def obl06(d1, d2):
    check_args(d1, d2)
    return _erfa.obl06(d1, d2)

def obl80(d1, d2):
    check_args(d1, d2)
    return _erfa.obl80(d1, d2)

def p06e(d1, d2):
    check_args(d1, d2)
    return _erfa.p06e(d1, d2)

def pb06(d1, d2):
    check_args(d1, d2)
    return _erfa.pb06(d1, d2)

def pfw06(d1, d2):
    check_args(d1, d2)
    return _erfa.pfw06(d1, d2)

def pmat00(d1, d2):
    check_args(d1, d2)
    return _erfa.pmat00(d1, d2)

def pmat06(d1, d2):
    check_args(d1, d2)
    return _erfa.pmat06(d1, d2)

def pmat76(d1, d2):
    check_args(d1, d2)
    return _erfa.pmat76(d1, d2)

def pn00(d1, d2, dpsi, deps):
    check_args(d1, d2, dpsi, deps)
    return _erfa.pn00(d1, d2, dpsi, deps)

def pn00a(d1, d2):
    check_args(d1, d2)
    return _erfa.pn00a(d1, d2)

def pn00b(d1, d2):
    check_args(d1, d2)
    return _erfa.pn00b(d1, d2)

def pn06(d1, d2, dpsi, deps):
    check_args(d1, d2, dpsi, deps)
    return _erfa.pn06(d1, d2, dpsi, deps)

def pn06a(d1, d2):
    check_args(d1, d2)
    return _erfa.pn06a(d1, d2)

def pnm00a(d1, d2):
    check_args(d1, d2)
    return _erfa.pnm00a(d1, d2)

def pnm00b(d1, d2):
    check_args(d1, d2)
    return _erfa.pnm00b(d1, d2)

def pnm06a(d1, d2):
    check_args(d1, d2)
    return _erfa.pnm06a(d1, d2)

def pnm80(d1, d2):
    check_args(d1, d2)
    return _erfa.pnm80(d1, d2)

def pom00(xp, yp, sp):
    check_args(xp, yp, sp)
    return _erfa.pom00(xp, yp, sp)

def pr00(d1,d2):
    check_args(d1,d2)
    return _erfa.pr00(d1,d2)

def prec76(ep01, ep02, ep11, ep12):
    check_args(ep01, ep02, ep11, ep12)
    return _erfa.prec76(ep01, ep02, ep11, ep12)

def s00(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s00(d1, d2, x, y)

def s00a(d1, d2):
    check_args(d1, d2)
    return _erfa.s00a(d1, d2)

def s00b(d1, d2):
    check_args(d1, d2)
    return _erfa.s00b(d1, d2)

def s06(d1, d2, x, y):
    check_args(d1, d2, x, y)
    return _erfa.s06(d1, d2, x, y)

def s06a(d1, d2):
    check_args(d1, d2)
    return _erfa.s06a(d1, d2)

def sp00(d1, d2):
    check_args(d1, d2)
    return _erfa.sp00(d1, d2)

def xy06(d1, d2):
    check_args(d1, d2)
    return _erfa.xy06(d1, d2)

def xys00a(d1, d2):
    check_args(d1, d2)
    return _erfa.xys00a(d1, d2)

def xys00b(d1, d2):
    check_args(d1, d2)
    return _erfa.xys00b(d1, d2)

def xys06a(d1, d2):
    check_args(d1, d2)
    return _erfa.xys06a(d1, d2)

## Astronomy/RotationAndTime
def ee00(d1, d2, epsa, dpsi):
    check_args(d1, d2, epsa, dpsi)
    return _erfa.ee00(d1, d2, epsa, dpsi)

def ee00a(d1, d2):
    check_args(d1, d2)
    return _erfa.ee00a(d1, d2)

def ee00b(d1, d2):
    check_args(d1, d2)
    return _erfa.ee00b(d1, d2)

def ee06a(d1, d2):
    check_args(d1, d2)
    return _erfa.ee06a(d1, d2)

def eect00(d1, d2):
    check_args(d1, d2)
    return _erfa.eect00(d1, d2)

def eqeq94(d1, d2):
    check_args(d1, d2)
    return _erfa.eqeq94(d1, d2)

def era00(d1, d2):
    check_args(d1, d2)
    return _erfa.era00(d1, d2)

def gmst00(uta, utb, tta, ttb):   
    check_args(uta, utb, tta, ttb)
    return _erfa.gmst00(uta, utb, tta, ttb)

def gmst06(uta, utb, tta, ttb):
    check_args(uta, utb, tta, ttb)
    return _erfa.gmst06(uta, utb, tta, ttb)

def gmst82(d1, d2):
    check_args(d1, d2)
    return _erfa.gmst82(d1, d2)

def gst00a(uta, utb, tta, ttb):
    check_args(uta, utb, tta, ttb)
    return _erfa.gst00a(uta, utb, tta, ttb)

def gst00b(uta, utb):
    check_args(uta, utb)
    return _erfa.gst00b(uta, utb)

def gst06(uta, utb, tta, ttb, rnpb):
    check_args(uta, utb, tta, ttb, rnpb)
    return _erfa.gst06(uta, utb, tta, ttb, rnpb)

def gst06a(uta, utb, tta, ttb):
    check_args(uta, utb, tta, ttb)
    return _erfa.gst06a(uta, utb, tta, ttb)

def gst94(uta, utb):
    check_args(uta, utb)
    return _erfa.gst94(uta, utb)

## Astronomy/SpaceMotion 
def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    check_args(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)
    return _erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

def pvstar(pv):
    check_args(pv)
    return _erfa.pvstar(pv)

def starpv(ra, dec, pmr, pmd, px, rv):
    check_args(ra, dec, pmr, pmd, px, rv)
    return _erfa.starpv(ra, dec, pmr, pmd, px, rv) 

def starpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    check_args(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)
    return _erfa.starpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b)

## Astronomy/StarCatalogs
def fk52h(r5, d5, dr5, dd5, px5,rv5):
    check_args(r5, d5, dr5, dd5, px5,rv5)
    return _erfa.fk52h(r5, d5, dr5, dd5, px5,rv5)

def fk5hz(r5, d5, d1, d2):
    check_args(r5, d5, d1, d2)
    return _erfa.fk5hz(r5, d5, d1, d2)

def h2fk5(rh, dh, drh, ddh, pxh, rvh):
    check_args(rh, dh, drh, ddh, pxh, rvh)
    return _erfa.h2fk5(rh, dh, drh, ddh, pxh, rvh)

def hfk5z(rh, dh, d1, d2):
    check_args(rh, dh, d1, d2)
    return _erfa.hfk5z(rh, dh, d1, d2)

## Astronomy/GeodeticGeocentric
def gc2gd(n, xyz):
    check_args(xyz)
    return _erfa.gc2gd(n, xyz)
    
def gd2gc(n, elong, phi, height):
    check_args(elong, phi, height)
    return _erfa.gd2gc(n, elong, phi, height)

def gc2gde(a, f, xyz):
    check_args(xyz)
    return _erfa.gc2gde(a, f, xyz)
    
def gd2gce(a, f, elong, phi, height):
    check_args(elong, phi, height)
    return _erfa.gd2gce(a, f, elong, phi, height)

def pvtob(elong, phi, hm, xp, yp, sp, theta):
    check_args(elong, phi, hm, xp, yp, sp, theta)
    return _erfa.pvtob(elong, phi, hm, xp, yp, sp, theta)

## Astronomy/Timescales
def d2dtf(scale, ndp, d1, d2):
    scale = scale.upper()
    if scale not in ('TAI', 'TCB', 'TCG', 'TDB', 'TT', 'UT1', 'UTC'):
        raise _erfa.error('unknown time scale: %s'%scale)
    check_args(d1, d2)
    return _erfa.d2dtf(scale, ndp, d1, d2)

jd_dtf = d2dtf

def dat(iy, im, id, fd):
    check_args(iy, im, id, fd)
    iy = cast_to_int32(iy)
    im = cast_to_int32(im)
    id = cast_to_int32(id)
    return _erfa.dat(iy, im, id, fd)

d_tai_utc = dat

def dtdb(d1, d2, ut1, elon, u, v):
    check_args(d1, d2, ut1, elon, u, v)
    return _erfa.dtdb(d1, d2, ut1, elon, u, v)

d_tdb_tt = dtdb

def dtf2d(scale, iy, im, id, ihr, imn, sec):   
    scale = scale.upper()
    if scale not in ('TAI', 'TCB', 'TCG', 'TDB', 'TT', 'UT1', 'UTC'):
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

def tdbtt(tdb1, tdb2, dtr):
    check_args(tdb1, tdb2, dtr)
    return _erfa.tdbtt(tdb1, tdb2, dtr)

tdb_tt = tdbtt

def tttai(tt1, tt2):
    check_args(tt1, tt2)
    return _erfa.tttai(tt1, tt2)

tt_tai = tttai

def tttcg(tt1, tt2):
    check_args(tt1, tt2)
    return _erfa.tttcg(tt1, tt2)

tt_tcg = tttcg

def tttdb(tt1, tt2, dtr):
    check_args(tt1, tt2, dtr)
    return _erfa.tttdb(tt1, tt2, dtr)

tt_tdb = tttdb

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

def ut1utc(ut11, ut12, dut1):
    check_args(ut11, ut12, dut1)
    return _erfa.ut1utc(ut11, ut12, dut1)

ut1_utc = ut1utc

def utctai(utc1, utc2):
    check_args(utc1, utc2)
    return _erfa.utctai(utc1, utc2)

utc_tai = utctai

def utcut1(utc1, utc2, dut1):
    check_args(utc1, utc2, dut1)
    return _erfa.utcut1(utc1, utc2, dut1)

utc_ut1 = utcut1

## VectorMatrix/AngleOps
def a2af(n, a):
    check_args(a)
    return _erfa.a2af(n, a)

def a2tf(n, a):
    check_args(a)
    return _erfa.a2tf(n, a)

def af2a(a):
    check_args(a)
    return _erfa.af2a(a)

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
