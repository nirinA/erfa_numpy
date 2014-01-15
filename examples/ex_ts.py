# -*- coding: utf-8 -*-
'''example from sofa_ts_c.pdf
'''
from __future__ import print_function
import math
import numpy as np
import erfa

print('''UTC to TT
transform 2010 July 24, 11:18:07.318 (UTC) into Terrestrial Time (TT)
and report it rounded to 1ms precision''')
# encode UTC date and time into internal format
u1, u2 = erfa.dtf2d('utc',
                    np.array([2010]),
                    np.array([7]),
                    np.array([24]),
                    np.array([11]),
                    np.array([18]),
                    np.array([7.318]))

# transform UTC to TAI, then TAI to TT
a1, a2 = erfa.utctai(u1, u2)
t1, t2 = erfa.taitt(a1, a2)

# decode and report the TT
y, m, d, h = erfa.d2dtf('TT', 3, t1, t2)
print("UTC: 2010 July 24, 11:18:07.318")
for i in range(len(y)):
    print("TT : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))

print('=====')

print('''
TAI to UTC
take a time expressed as TAI, encode it into
the internal format and transform it into UTC''')

# encode TAI date and time into internal format
a1, a2 = erfa.dtf2d("TAI",
                    np.array([2009]),
                    np.array([1]),
                    np.array([1]),
                    np.array([0]),
                    np.array([0]),
                    np.array([33.7]))

# decode and report TAI
y, m, d, h = erfa.d2dtf("TAI", 3, a1, a2)
for i in range(len(y)):
    print("TAI : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))

# transform TAI to UTC
u1, u2 = erfa.taiutc(a1, a2)

# decode and report UTC
y, m, d, h = erfa.d2dtf('utc', 3, u1, u2)
for i in range(len(y)):
    print("UTC : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))

print('=====')

print('''
transform UTC to other times.
an observer at north latitude +19°28'52''.5,
west longitude 155°55'59''.6,
at sea level, on 2006 January 15 at 21:24:37.5 UTC
requires the time in all other supported time scales''')

# site terrestrial coordinates (WGS84)
latnd = +19
latnm = 28
slatn = 52.5
lonwd = -155
lonwm = 55
slonw = 59.6
hm = 0.

# transform to geocentric
phi = erfa.af2a(np.array([[latnd, latnm, slatn]]))
elon = erfa.af2a(np.array([[lonwd, lonwm, slonw]]))
xyz = erfa.gd2gc(1, elon, phi, np.array([hm]))[0]
u = np.array([math.hypot(xyz[0], xyz[1])])
v = np.array([xyz[2]])

# UTC date and time
iy = np.array([2006])
mo = np.array([1])
d = np.array([15])
ih = np.array([21])
im = np.array([24])
sec = np.array([37.5])

# transform into intenal format
utc1, utc2 = erfa.dtf2d("UTC", iy,mo,d,ih,im,sec)

# UT1-UTC from IERS
dut = np.array([.3341])

# UTC -> UT1
ut11, ut12 = erfa.utcut1(utc1, utc2, dut)

# Extract fraction for TDB-TT calculation, later.
ut = np.array([math.fmod(math.fmod(ut11,1.0)+math.fmod(ut12,1.0),1.0)])

# UTC -> TAI -> TT -> TCG
tai1, tai2 = erfa.utctai(utc1, utc2)
tt1, tt2 = erfa.taitt(tai1, tai2)
tcg1, tcg2 = erfa.tttcg(tt1, tt2)

# TDB-TT (using TT as a substitute for TDB).
dtr = erfa.dtdb(tt1, tt2, ut, elon, u, v)

# TT -> TDB -> TCB.
tdb1, tdb2 = erfa.tttdb(tt1, tt2, dtr)
tcb1, tcb2 = erfa.tdbtcb(tdb1, tdb2)

# report
y, m, d, h = erfa.d2dtf('utc', 6, utc1, utc2)
for i in range(len(y)):
    print("UTC : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("ut1", 6, ut11, ut12)
for i in range(len(y)):
    print("UT1 : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("tai", 6, tai1, tai2)      
for i in range(len(y)):
    print("TAI : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("tt", 6, tt1, tt2)
for i in range(len(y)):
    print("TT : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("tcg", 6, tcg1, tcg2)
for i in range(len(y)):
    print("TCG : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("tdb", 6, tdb1, tdb2)
for i in range(len(y)):
    print("TDB : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
y, m, d, h = erfa.d2dtf("tcb", 6, tcb1, tcb2)
for i in range(len(y)):
    print("TCB : %4d/%2.2d/%2.2d, "%(y[i],m[i],d[i]), end =' ')
    print("%3d:%2.2d:%2.2d.%3.3d"%tuple(h[i]))
