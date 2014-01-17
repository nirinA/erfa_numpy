import unittest
try:
    from test import support
except ImportError:
    from test import test_support as support
import math
import numpy as np
import erfa
##import _erfa

class Validate(unittest.TestCase):
    '''
     float:   self.assertAlmostEqual(value, expected, places=diff)       
     int:     self.assertEqual(value, expected)
     char:    self.assertEqual(value, expected)
    '''

## Astronomy/Calendars
    def test_cal2jd(self):
        dmj0, dmj = erfa.cal2jd(np.array([2003]), np.array([6]), np.array([1]))
        self.assertAlmostEqual(dmj0[0], 2400000.5, 9)
        self.assertAlmostEqual(dmj[0], 52791.0, 9)

    def test_d_tdb_tt(self):
        d1 = np.array([2448939.5])
        d2 = np.array([0.123])
        ut1 = np.array([0.76543])
        elon = np.array([5.0123])
        u = np.array([5525.242])
        v = np.array([3190.0])
        d = erfa.d_tdb_tt(d1, d2, ut1, elon, u, v)
        self.assertAlmostEqual(d[0], -0.1280368005936998991e-2, 17)

    def test_besselian_epoch_jd(self):
        dj0, dj1 = erfa.besselian_epoch_jd(np.array([1957.3]))
        self.assertAlmostEqual(dj0[0], 2400000.5, 9)
        self.assertAlmostEqual(dj1[0], 35948.1915101513, 9)

    def test_jd_besselian_epoch(self):
        b = erfa.jd_besselian_epoch(np.array([2415019.8135]), np.array([30103.18648]))
        self.assertAlmostEqual(b[0], 1982.418424159278580, 12)

    def test_jd_julian_epoch(self):
        j = erfa.jd_julian_epoch(np.array([2451545]), np.array([-7392.5]))
        self.assertAlmostEqual(j[0], 1979.760438056125941, 12)

    def test_julian_epoch_jd(self):
        dj0, dj1 = erfa.julian_epoch_jd(np.array([1996.8]))
        self.assertAlmostEqual(dj0[0], 2400000.5, 9)
        self.assertAlmostEqual(dj1[0], 50375.7, 9)

    def test_jd2cal(self):
        y, m, d, fd = erfa.jd2cal(np.array([2400000.5]), np.array([50123.9999]))
        self.assertEqual(y[0], 1996)
        self.assertEqual(m[0], 2)
        self.assertEqual(d[0], 10)
        self.assertAlmostEqual(fd[0], 0.9999, 7)
        
    def test_jdcalf(self):
        y, m, d, fd = erfa.jdcalf(4, np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertEqual(y, 1996)
        self.assertEqual(m, 2)
        self.assertEqual(d, 10)
        self.assertEqual(fd, 9999)

## astrometry tools
    def test_ab(self):
        pnat = np.array([[-0.76321968546737951,-0.60869453983060384,-0.21676408580639883]])
        v = np.array([[2.1044018893653786e-5,-8.9108923304429319e-5,-3.8633714797716569e-5]])
        s = np.array([0.99980921395708788])
        bm1 = np.array([0.99999999506209258])
        ppr = erfa.ab(pnat, v, s, bm1)[0]
        self.assertAlmostEqual(ppr[0], -0.7631631094219556269, places=12)
        self.assertAlmostEqual(ppr[1], -0.6087553082505590832, places=12)
        self.assertAlmostEqual(ppr[2], -0.2167926269368471279, places=12)

    def test_apcs(self):
        date1 = np.array([2456384.5])
        date2 = np.array([0.970031644])
        pv = np.array([[[-1836024.09,1056607.72,-5998795.26],
                        [-77.0361767,-133.310856,0.0971855934]]])
        ebpv = np.array([[[-0.974170438,-0.211520082,-0.0917583024],
                          [0.00364365824,-0.0154287319,-0.00668922024]]])
        ehp = np.array([[-0.973458265,-0.209215307,-0.0906996477]])
        astrom = erfa.apcs(date1, date2, pv, ebpv, ehp)[0]
        self.assertAlmostEqual(astrom.pmt, 13.25248468622587269, places=11)
        self.assertAlmostEqual(astrom.eb[0], -0.9741827110630456169, places=12)
        self.assertAlmostEqual(astrom.eb[1], -0.2115130190136085494, places=12)
        self.assertAlmostEqual(astrom.eb[2], -0.09179840186973175487, places=12)
        self.assertAlmostEqual(astrom.eh[0], -0.9736425571689386099, places=12)
        self.assertAlmostEqual(astrom.eh[1], -0.2092452125849967195, places=12)
        self.assertAlmostEqual(astrom.eh[2], -0.09075578152266466572, places=12)
        self.assertAlmostEqual(astrom.em, 0.9998233241710457140, places=12)
        self.assertAlmostEqual(astrom.v[0], 0.2078704985513566571e-4, places=16)
        self.assertAlmostEqual(astrom.v[1], -0.8955360074245006073e-4, places=16)
        self.assertAlmostEqual(astrom.v[2], -0.3863338980073572719e-4, places=16)
        self.assertAlmostEqual(astrom.bm1, 0.9999999950277561601, places=12)
        self.assertAlmostEqual(astrom.bpn[0][0], 1, 10)
        self.assertAlmostEqual(astrom.bpn[1][0], 0, 10)
        self.assertAlmostEqual(astrom.bpn[2][0], 0, 10)
        self.assertAlmostEqual(astrom.bpn[0][1], 0, 10)
        self.assertAlmostEqual(astrom.bpn[1][1], 1, 10)
        self.assertAlmostEqual(astrom.bpn[2][1], 0, 10)
        self.assertAlmostEqual(astrom.bpn[0][2], 0, 10)
        self.assertAlmostEqual(astrom.bpn[1][2], 0, 10)
        self.assertAlmostEqual(astrom.bpn[2][2], 1, 10)

    def test_ld(self):
        bm = np.array([0.00028574])
        p = np.array([[-0.763276255, -0.608633767, -0.216735543]])
        q = np.array([[-0.763276255, -0.608633767, -0.216735543]])
        e = np.array([[0.76700421, 0.605629598, 0.211937094]])
        em = np.array([8.91276983])
        dlim = np.array([3e-10])
        p1 = erfa.ld(bm, p, q, e, em, dlim)[0]
        self.assertAlmostEqual(p1[0], -0.7632762548968159627, places=12)
        self.assertAlmostEqual(p1[1], -0.6086337670823762701, places=12)
        self.assertAlmostEqual(p1[2], -0.2167355431320546947, places=12)

## Astronomy/SpaceMotion 
    def test_pmsafe(self):
        ra1 = np.array([1.234])
        dec1 = np.array([0.789])
        pmr1 = np.array([1e-5])
        pmd1 = np.array([-2e-5])
        px1 = np.array([1e-2])
        rv1 = np.array([10.0])
        ep1a = np.array([2400000.5])
        ep1b = np.array([48348.5625])
        ep2a = np.array([2400000.5])
        ep2b = np.array([51544.5])
        ra2, dec2, pmr2, pmd2, px2, rv2 = erfa.pmsafe(ra1, dec1, pmr1, pmd1, px1,
                                                      rv1, ep1a, ep1b, ep2a, ep2b)
        self.assertAlmostEqual(ra2[0], 1.234087484501017061, places=12)
        self.assertAlmostEqual(dec2[0], 0.7888249982450468574, places=12)
        self.assertAlmostEqual(pmr2[0], 0.9996457663586073988e-5, places=12)
        self.assertAlmostEqual(pmd2[0], -0.2000040085106737816e-4, places=16)
        self.assertAlmostEqual(px2[0], 0.9999997295356765185e-2, places=12)
        self.assertAlmostEqual(rv2[0], 10.38468380113917014, places=10)

## Astronomy/GeodeticGeocentric
    def test_gd2gc(self):
        e = np.array([3.1])
        p = np.array([-0.5])
        h = np.array([2500.0])
        
        self.assertRaises(erfa.error, erfa.gd2gc, 0, e, p, h)
        xyz = erfa.gd2gc(1,e,p,h)[0]
        self.assertAlmostEqual(xyz[0], -5599000.5577049947, places=7)
        self.assertAlmostEqual(xyz[1], 233011.67223479203, places=7)
        self.assertAlmostEqual(xyz[2], -3040909.4706983363, places=7)
        xyz = erfa.gd2gc(2,e,p,h)[0]
        self.assertAlmostEqual(xyz[0], -5599000.5577260984, places=7)
        self.assertAlmostEqual(xyz[1], 233011.6722356703, places=7)
        self.assertAlmostEqual(xyz[2], -3040909.4706095476, places=7)
        xyz = erfa.gd2gc(3,e,p,h)[0]
        self.assertAlmostEqual(xyz[0], -5598998.7626301490, places=7)
        self.assertAlmostEqual(xyz[1], 233011.5975297822, places=7)
        self.assertAlmostEqual(xyz[2], -3040908.6861467111, places=7)        
        self.assertRaises(erfa.error, erfa.gd2gc, 4, e, p, h)


## Astronomy/Timescales
    def test_d_tai_utc(self):
        d = erfa.d_tai_utc(np.array([2003]), np.array([6]), np.array([1]), np.array([0.0]))
        self.assertAlmostEqual(d[0], 32.0, 9)
        d = erfa.d_tai_utc(np.array([2008]), np.array([1]), np.array([17]), np.array([0.0]))
        self.assertAlmostEqual(d[0], 33.0, 9)

    def test_jd_dtf(self):
        scale = 'UTC'
        d1 = np.array([2400000.5])
        d2 = np.array([49533.99999])
        ndp = 5
        y, m, d, hmsf = erfa.jd_dtf(scale, ndp, d1, d2)
        self.assertEqual(y[0], 1994)
        self.assertEqual(m[0], 6)
        self.assertEqual(d[0], 30)
        self.assertEqual(hmsf[0][0], 23)
        self.assertEqual(hmsf[0][1], 59)
        self.assertEqual(hmsf[0][2], 60)
        self.assertEqual(hmsf[0][3], 13599)

    def test_dtf_jd(self):
        scale = 'UTC'
        y = np.array([1994])
        m = np.array([6])
        d = np.array([30])
        h = np.array([23])
        mn = np.array([59])
        sec = np.array([60.13599])
        jd1, jd2 = erfa.dtf_jd(scale, y, m, d, h, mn, sec)
        self.assertAlmostEqual(jd1[0]+jd2[0], 2449534.49999, 6)

    def test_dtf2d(self):
        scale = 'UTC'
        y = np.array([1994], dtype='int32')
        m = np.array([6], dtype='int32')
        d = np.array([30], dtype='int32')
        h = np.array([23], dtype='int32')
        mn = np.array([59], dtype='int32')
        sec = np.array([60.13599])
        jd1, jd2 = erfa.dtf2d(scale, y, m, d, h, mn, sec)
        self.assertAlmostEqual(jd1[0]+jd2[0], 2449534.49999, 6)

    def test_tai_tt(self):
        t1, t2 = erfa.tai_tt(np.array([2453750.5]), np.array([0.892482639]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.892855139, places=12)

    def test_tai_ut1(self):
        u1, u2 = erfa.tai_ut1(np.array([2453750.5]),
                              np.array([0.892482639]),
                              np.array([-32.6659]))
        self.assertAlmostEqual(u1[0], 2453750.5, places=6)
        self.assertAlmostEqual(u2[0], 0.8921045614537037037, places=12)

    def test_tai_utc(self):
        u1, u2 = erfa.tai_utc(np.array([2453750.5]), np.array([0.892482639]))
        self.assertAlmostEqual(u1[0], 2453750.5, places=6)
        self.assertAlmostEqual(u2[0], 0.8921006945555555556, places=12)

    def test_tcb_tdb(self):
        b1, b2 = erfa.tcb_tdb(np.array([2453750.5]), np.array([0.893019599]))
        self.assertAlmostEqual(b1[0], 2453750.5, places=6)
        self.assertAlmostEqual(b2[0], 0.8928551362746343397, places=12)
        
    def test_tcg_tt(self):
        t1, t2 = erfa.tcg_tt(np.array([2453750.5]), np.array([0.892862531]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.8928551387488816828, places=12)

    def test_tdb_tcb(self):
        b1, b2 = erfa.tdb_tcb(np.array([2453750.5]), np.array([0.892855137]))
        self.assertAlmostEqual(b1[0], 2453750.5, places=6)
        self.assertAlmostEqual(b2[0], 0.8930195997253656716, places=12)

    def test_tdb_tt(self):
        t1, t2 = erfa.tdb_tt(np.array([2453750.5]),
                             np.array([0.892855137]),
                             np.array([-0.000201]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.8928551393263888889, places=12)

    def test_tt_tai(self):
        t1, t2 = erfa.tt_tai(np.array([2453750.5]), np.array([0.892482639]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.892110139, places=12)

    def test_tt_tcg(self):
        t1, t2 = erfa.tt_tcg(np.array([2453750.5]), np.array([0.892482639]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.8924900312508587113, places=12)

    def test_tt_tdb(self):
        t1, t2 = erfa.tt_tdb(np.array([2453750.5]),
                             np.array([0.892855139]),
                             np.array([-0.000201]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.8928551366736111111, places=12)

    def test_tt_ut1(self):
        t1, t2 = erfa.tt_ut1(np.array([2453750.5]),
                             np.array([0.892855139]),
                             np.array([64.8499]))
        self.assertAlmostEqual(t1[0], 2453750.5, places=6)
        self.assertAlmostEqual(t2[0], 0.8921045614537037037, places=12)

    def test_ut1_utc(self):
        a1, a2 = erfa.ut1_utc(np.array([2453750.5]),
                              np.array([0.892104561]),
                              np.array([0.3341]))
        self.assertAlmostEqual(a1[0], 2453750.5, places=6)
        self.assertAlmostEqual(a2[0], 0.8921006941018518519, places=13)
        
    def test_ut1_tai(self):
        a1, a2 = erfa.ut1_tai(np.array([2453750.5]),
                              np.array([0.892104561]),
                              np.array([-32.6659]))
        self.assertAlmostEqual(a1[0], 2453750.5, places=6)
        self.assertAlmostEqual(a2[0], 0.8924826385462962963, places=12)
        
    def test_ut1_tt(self):
        a1, a2 = erfa.ut1_tt(np.array([2453750.5]),
                             np.array([0.892104561]),
                             np.array([64.8499]))
        self.assertAlmostEqual(a1[0], 2453750.5, places=6)
        self.assertAlmostEqual(a2[0], 0.8928551385462962963, places=15)
        
    def test_utc_tai(self):
        u1, u2 = erfa.utc_tai(np.array([2453750.5]), np.array([0.892100694]))
        self.assertAlmostEqual(u1[0], 2453750.5, places=6)
        self.assertAlmostEqual(u2[0], 0.8924826384444444444, places=13)
        
    def test_utc_ut1(self):
        u1, u2 = erfa.utc_ut1(np.array([2453750.5]),
                              np.array([0.892100694]),
                              np.array([0.3341]))
        self.assertAlmostEqual(u1[0], 2453750.5, places=6)
        self.assertAlmostEqual(u2[0], 0.8921045608981481481, places=13)

## Astronomy/Ephemerides
    def test_epv00(self):
        pvh, pvb = erfa.epv00(np.array([2400000.5]), np.array([53411.52501161]))
        self.assertAlmostEqual(pvh[0][0][0], -0.7757238809297706813, places=14)
        self.assertAlmostEqual(pvh[0][0][1], 0.5598052241363340596, places=13) #fail at 14
        self.assertAlmostEqual(pvh[0][0][2], 0.2426998466481686993, places=14)

        self.assertAlmostEqual(pvh[0][1][0], -0.1091891824147313846e-1, places=15)
        self.assertAlmostEqual(pvh[0][1][1], -0.1247187268440845008e-1, places=15)
        self.assertAlmostEqual(pvh[0][1][2], -0.5407569418065039061e-2, places=15)
        
        self.assertAlmostEqual(pvb[0][0][0], -0.7714104440491111971, places=14)
        self.assertAlmostEqual(pvb[0][0][1], 0.5598412061824171323, places=13)  #fail at 14
        self.assertAlmostEqual(pvb[0][0][2], 0.2425996277722452400, places=14)

        self.assertAlmostEqual(pvb[0][1][0], -0.1091874268116823295e-1, places=15)
        self.assertAlmostEqual(pvb[0][1][1], -0.1246525461732861538e-1, places=15)
        self.assertAlmostEqual(pvb[0][1][2], -0.5404773180966231279e-2, places=15)

    def test_plan94(self):
        self.assertRaises(erfa.error, erfa.plan94, np.array([2400000.5]), np.array([-320000]), 0)
        self.assertRaises(erfa.error, erfa.plan94, np.array([2400000.5]), np.array([-320000]), 10)

        pv = erfa.plan94(np.array([2400000.5]), np.array([-320000]), 3)[0]
        self.assertAlmostEqual(pv[0][0], 0.9308038666832975759, places=11)
        self.assertAlmostEqual(pv[0][1], 0.3258319040261346000, places=11)
        self.assertAlmostEqual(pv[0][2], 0.1422794544481140560, places=11)

        self.assertAlmostEqual(pv[1][0], -0.6429458958255170006e-2, places=11)
        self.assertAlmostEqual(pv[1][1], 0.1468570657704237764e-1, places=11)
        self.assertAlmostEqual(pv[1][2], 0.6406996426270981189e-2, places=11)

        pv = erfa.plan94(np.array([2400000.5]), np.array([43999.9]), 1)[0]
        self.assertAlmostEqual(pv[0][0], 0.2945293959257430832, places=11)
        self.assertAlmostEqual(pv[0][1], -0.2452204176601049596, places=11)
        self.assertAlmostEqual(pv[0][2], -0.1615427700571978153, places=11)

        self.assertAlmostEqual(pv[1][0], 0.1413867871404614441e-1, places=11)
        self.assertAlmostEqual(pv[1][1], 0.1946548301104706582e-1, places=11)
        self.assertAlmostEqual(pv[1][2], 0.8929809783898904786e-2, places=11)

## Astronomy/FundamentalArgs
    def test_fad03(self):
        d = erfa.fad03(np.array([0.80]))[0]
        self.assertAlmostEqual(d, 1.946709205396925672, 12)

    def test_fae03(self):
        d = erfa.fae03(np.array([0.80]))[0]
        self.assertAlmostEqual(d, 1.744713738913081846, 12)

    def test_faf03(self):
        f = erfa.faf03(np.array([0.80]))[0]
        self.assertAlmostEqual(f, 0.2597711366745499518, 11) # failed at 12
        
    def test_faju03(self):
        l = erfa.faju03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 5.275711665202481138, 12)
        
    def test_fal03(self):
        l = erfa.fal03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 5.132369751108684150, 12)
        
    def test_falp03(self):
        lp = erfa.falp03(np.array([0.80]))[0]
        self.assertAlmostEqual(lp, 6.226797973505507345, 12)

    def test_fama03(self):
        l = erfa.fama03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 3.275506840277781492, 12)
        
    def test_fame03(self):
        l = erfa.fame03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 5.417338184297289661, 12)
        
    def test_fane03(self):
        l = erfa.fane03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 2.079343830860413523, 12)
        
    def test_faom03(self):
        l = erfa.faom03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, -5.973618440951302183, 12)
        
    def test_fapa03(self):
        l = erfa.fapa03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 0.1950884762240000000e-1, 12)
                
    def test_fasa03(self):
        l = erfa.fasa03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 5.371574539440827046, 12)

    def test_faur03(self):
        l = erfa.faur03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 5.180636450180413523, 12)
        
    def test_fave03(self):
        l = erfa.fave03(np.array([0.80]))[0]
        self.assertAlmostEqual(l, 3.424900460533758000, 12)

## Astronomy/PrecNutPolar
    def test_bi00(self):
        dp, de, dr = erfa.bi00()
        self.assertAlmostEqual(dp, -0.2025309152835086613e-6, places=15)
        self.assertAlmostEqual(de, -0.3306041454222147847e-7, places=15)
        self.assertAlmostEqual(dr, -0.7078279744199225506e-7, places=15)

    def test_bp00(self):
        rb, rp,rbp = erfa.bp00(np.array([2400000.5]), np.array([50123.9999]))
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942498, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078279744199196626e-7, places=16)
        self.assertAlmostEqual(rb[0][0][2], 0.8056217146976134152e-7, places=16)
        self.assertAlmostEqual(rb[0][1][0], 0.7078279477857337206e-7, places=16)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3306041454222136517e-7, places=16)
        self.assertAlmostEqual(rb[0][2][0], -0.8056217380986972157e-7, places=16)
        self.assertAlmostEqual(rb[0][2][1], -0.3306040883980552500e-7, places=16)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999995504864048241, places=12)
        self.assertAlmostEqual(rp[0][0][1], 0.8696113836207084411e-3, places=14)
        self.assertAlmostEqual(rp[0][0][2], 0.3778928813389333402e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], -0.8696113818227265968e-3, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999996218879365258, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.1690679263009242066e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], -0.3778928854764695214e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.1595521004195286491e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999999285984682756, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999995505175087260, places=12)
        self.assertAlmostEqual(rbp[0][0][1], 0.8695405883617884705e-3, places=14)
        self.assertAlmostEqual(rbp[0][0][2], 0.3779734722239007105e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], -0.8695405990410863719e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999996219494925900, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.1360775820404982209e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], -0.3779734476558184991e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.1925857585832024058e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999999285680153377, places=12)

    def test_bp06(self):
        rb, rp,rbp = erfa.bp06(np.array([2400000.5]), np.array([50123.9999]))
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942497, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078368960971557145e-7, places=14)
        self.assertAlmostEqual(rb[0][0][2], 0.8056213977613185606e-7, places=14)
        self.assertAlmostEqual(rb[0][1][0], 0.7078368694637674333e-7, places=14)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3305943742989134124e-7, places=14)
        self.assertAlmostEqual(rb[0][2][0], -0.8056214211620056792e-7, places=14)
        self.assertAlmostEqual(rb[0][2][1], -0.3305943172740586950e-7, places=14)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)

        self.assertAlmostEqual(rp[0][0][0], 0.9999995504864960278, places=12)
        self.assertAlmostEqual(rp[0][0][1], 0.8696112578855404832e-3, places=14)
        self.assertAlmostEqual(rp[0][0][2], 0.3778929293341390127e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], -0.8696112560510186244e-3, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999996218880458820, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.1691646168941896285e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], -0.3778929335557603418e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.1594554040786495076e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999999285984501222, places=12)

        self.assertAlmostEqual(rbp[0][0][0], 0.9999995505176007047, places=12)
        self.assertAlmostEqual(rbp[0][0][1], 0.8695404617348208406e-3, places=14)
        self.assertAlmostEqual(rbp[0][0][2], 0.3779735201865589104e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], -0.8695404723772031414e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999996219496027161, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.1361752497080270143e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], -0.3779734957034089490e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.1924880847894457113e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999999285679971958, places=12)

    def test_bpn2xy(self):
        rbpn = np.array([
            ((9.999962358680738e-1,-2.516417057665452e-3,-1.093569785342370e-3),
             (2.516462370370876e-3,9.999968329010883e-1,4.006159587358310e-5),
             (1.093465510215479e-3,-4.281337229063151e-5,9.999994012499173e-1))
            ])
        x, y = erfa.bpn2xy(rbpn)
        self.assertAlmostEqual(x[0], 1.093465510215479e-3, places=12)
        self.assertAlmostEqual(y[0], -4.281337229063151e-5, places=12)

    def test_c2i00a(self):
        rc2i = erfa.c2i00a(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(rc2i[0][0][0], 0.9999998323037165557, places=12)
        self.assertAlmostEqual(rc2i[0][0][1], 0.5581526348992140183e-9, places=12)
        self.assertAlmostEqual(rc2i[0][0][2], -0.5791308477073443415e-3, places=12)

        self.assertAlmostEqual(rc2i[0][1][0], -0.2384266227870752452e-7, places=12)
        self.assertAlmostEqual(rc2i[0][1][1], 0.9999999991917405258, places=12)
        self.assertAlmostEqual(rc2i[0][1][2], -0.4020594955028209745e-4, places=12)

        self.assertAlmostEqual(rc2i[0][2][0], 0.5791308472168152904e-3, places=12)
        self.assertAlmostEqual(rc2i[0][2][1], 0.4020595661591500259e-4, places=12)
        self.assertAlmostEqual(rc2i[0][2][2], 0.9999998314954572304, places=12)

    def test_c2i00b(self):
        rc2i = erfa.c2i00b(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rc2i[0][0], 0.9999998323040954356, places=12)
        self.assertAlmostEqual(rc2i[0][1], 0.5581526349131823372e-9, places=12)
        self.assertAlmostEqual(rc2i[0][2], -0.5791301934855394005e-3, places=12)

        self.assertAlmostEqual(rc2i[1][0], -0.2384239285499175543e-7, places=12)
        self.assertAlmostEqual(rc2i[1][1], 0.9999999991917574043, places=12)
        self.assertAlmostEqual(rc2i[1][2], -0.4020552974819030066e-4, places=12)

        self.assertAlmostEqual(rc2i[2][0], 0.5791301929950208873e-3, places=12)
        self.assertAlmostEqual(rc2i[2][1], 0.4020553681373720832e-4, places=12)
        self.assertAlmostEqual(rc2i[2][2], 0.9999998314958529887, places=12)

    def test_c2i06a(self):
        rc2i = erfa.c2i06a(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rc2i[0][0], 0.9999998323037159379, places=12)
        self.assertAlmostEqual(rc2i[0][1], 0.5581121329587613787e-9, places=12)
        self.assertAlmostEqual(rc2i[0][2], -0.5791308487740529749e-3, places=12)

        self.assertAlmostEqual(rc2i[1][0], -0.2384253169452306581e-7, places=12)
        self.assertAlmostEqual(rc2i[1][1], 0.9999999991917467827, places=12)
        self.assertAlmostEqual(rc2i[1][2], -0.4020579392895682558e-4, places=12)

        self.assertAlmostEqual(rc2i[2][0], 0.5791308482835292617e-3, places=12)
        self.assertAlmostEqual(rc2i[2][1], 0.4020580099454020310e-4, places=12)
        self.assertAlmostEqual(rc2i[2][2], 0.9999998314954628695, places=12)

    def test_c2ibpn(self):
        rbpn = np.array([
            ((9.999962358680738e-1,-2.516417057665452e-3,-1.093569785342370e-3),
             (2.516462370370876e-3,9.999968329010883e-1,4.006159587358310e-5),
             (1.093465510215479e-3,-4.281337229063151e-5,9.999994012499173e-1))
            ])
        rc2i = erfa.c2ibpn(np.array([2400000.5]), np.array([50123.9999]), rbpn)[0]
        self.assertAlmostEqual(rc2i[0][0], 0.9999994021664089977, places=12)
        self.assertAlmostEqual(rc2i[0][1], -0.3869195948017503664e-8, places=12)
        self.assertAlmostEqual(rc2i[0][2], -0.1093465511383285076e-2, places=12)

        self.assertAlmostEqual(rc2i[1][0], 0.5068413965715446111e-7, places=12)
        self.assertAlmostEqual(rc2i[1][1], 0.9999999990835075686, places=12)
        self.assertAlmostEqual(rc2i[1][2], 0.4281334246452708915e-4, places=12)

        self.assertAlmostEqual(rc2i[2][0], 0.1093465510215479000e-2, places=12)
        self.assertAlmostEqual(rc2i[2][1], -0.4281337229063151000e-4, places=12)
        self.assertAlmostEqual(rc2i[2][2], 0.9999994012499173103, places=12)

    def test_c2ixy(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736])
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        rc2i = erfa.c2ixy(d1, d2, x, y)[0]
        self.assertAlmostEqual(rc2i[0][0], 0.9999998323037157138, places=12)
        self.assertAlmostEqual(rc2i[0][1], 0.5581526349032241205e-9, places=12)
        self.assertAlmostEqual(rc2i[0][2], -0.5791308491611263745e-3, places=12)

        self.assertAlmostEqual(rc2i[1][0], -0.2384257057469842953e-7, places=12)
        self.assertAlmostEqual(rc2i[1][1], 0.9999999991917468964, places=12)
        self.assertAlmostEqual(rc2i[1][2], -0.4020579110172324363e-4, places=12)

        self.assertAlmostEqual(rc2i[2][0], 0.5791308486706011000e-3, places=12)
        self.assertAlmostEqual(rc2i[2][1], 0.4020579816732961219e-4, places=12)
        self.assertAlmostEqual(rc2i[2][2], 0.9999998314954627590, places=12)

    def test_c2ixys(self):
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        s = np.array([-0.1220040848472271978e-7])
        rc2i = erfa.c2ixys(x, y, s)[0]
        self.assertAlmostEqual(rc2i[0][0], 0.9999998323037157138, places=12)
        self.assertAlmostEqual(rc2i[0][1], 0.5581984869168499149e-9, places=12)
        self.assertAlmostEqual(rc2i[0][2], -0.5791308491611282180e-3, places=12)

        self.assertAlmostEqual(rc2i[1][0], -0.2384261642670440317e-7, places=12)
        self.assertAlmostEqual(rc2i[1][1], 0.9999999991917468964, places=12)
        self.assertAlmostEqual(rc2i[1][2], -0.4020579110169668931e-4, places=12)

        self.assertAlmostEqual(rc2i[2][0], 0.5791308486706011000e-3, places=12)
        self.assertAlmostEqual(rc2i[2][1], 0.4020579816732961219e-4, places=12)
        self.assertAlmostEqual(rc2i[2][2], 0.9999998314954627590, places=12)

    def test_c2t00a(self):
        tta = np.array([2400000.5])
        uta = np.array([2400000.5])
        ttb = np.array([53736.0])
        utb = np.array([53736.0])
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        rc2t = erfa.c2t00a(tta, ttb, uta, utb, xp, yp)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128307182668, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806938457836, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555535638688341725e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134135984552, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203649520727, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749801116141056317e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773474014081406921e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961832391770163647e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325501692289, places=12)
        
    def test_c2t00b(self):
        tta = np.array([2400000.5])
        uta = np.array([2400000.5])
        ttb = np.array([53736.0])
        utb = np.array([53736.0])
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        rc2t = erfa.c2t00b(tta, ttb, uta, utb, xp, yp)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128439678965, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806913872359, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555565082458415611e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134115435923, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203784001946, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749793922030017230e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773467471863534901e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961790411549945020e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325505635738, places=12)

    def test_c2t06a(self):
        tta = np.array([2400000.5])
        uta = np.array([2400000.5])
        ttb = np.array([53736.0])
        utb = np.array([53736.0])
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        rc2t = erfa.c2t06a(tta, ttb, uta, utb, xp, yp)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128305897282, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806938592296, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555550962998436505e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134136214897, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203649130832, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749800844905594110e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773474024748545878e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961816829632690581e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325501747785, places=12)

    def test_c2tcio(self):
        c = np.array([
            ((0.9999998323037164738,0.5581526271714303683e-9,-0.5791308477073443903e-3),
             (-0.2384266227524722273e-7,0.9999999991917404296,-0.4020594955030704125e-4),
             (0.5791308472168153320e-3,.4020595661593994396e-4,0.9999998314954572365))
            ])
        era = np.array([1.75283325530307])
        p = np.array([
            ((0.9999999999999674705,-0.1367174580728847031e-10,0.2550602379999972723e-6),
             (0.1414624947957029721e-10,0.9999999999982694954,-0.1860359246998866338e-5),
             (-0.2550602379741215275e-6,0.1860359247002413923e-5,0.9999999999982369658))
            ])
        rc2t = erfa.c2tcio(c, era, p)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128307110439, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806938470149, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555535638685466874e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134135996657, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203649448367, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749801116141106528e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773474014081407076e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961832391772658944e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325501691969, places=12)

    def test_c2teqx(self):
        c = np.array([
            ((0.9999989440476103608,-0.1332881761240011518e-2,-0.5790767434730085097e-3),
             (0.1332858254308954453e-2,0.9999991109044505944,-0.4097782710401555759e-4),
             (0.5791308472168153320e-3,0.4020595661593994396e-4,0.9999998314954572365))
            ])
        gst = np.array([1.754166138040730516])
        p = np.array([
            ((0.9999999999999674705,-0.1367174580728847031e-10,0.2550602379999972723e-6),
             (0.1414624947957029721e-10,0.9999999999982694954,-0.1860359246998866338e-5),
             (-0.2550602379741215275e-6,0.1860359247002413923e-5,0.9999999999982369658))
            ])
        rc2t = erfa.c2teqx(c,gst,p)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128528685730, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806897685071, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555535639982634449e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134095211257, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203871023800, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749801116126438962e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773474014081539467e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961832391768640871e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325501691969, places=12)

    def test_c2tpe(self):
        tta = np.array([2400000.5])
        uta = np.array([2400000.5])
        ttb = np.array([53736.0])
        utb = np.array([53736.0])
        deps = np.array([0.4090789763356509900])
        dpsi = np.array([-0.9630909107115582393e-5])
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        rc2t = erfa.c2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1813677995763029394, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9023482206891683275, places=12)
        self.assertAlmostEqual(rc2t[0][2], -0.3909902938641085751, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834147641476804807, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1659883635434995121, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.7309763898042819705e-1, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.1059685430673215247e-2, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3977631855605078674, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9174875068792735362, places=12)

    def test_c2txy(self):
        tta = np.array([2400000.5])
        uta = np.array([2400000.5])
        ttb = np.array([53736.0])
        utb = np.array([53736.0])
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        rc2t = erfa.c2txy(tta, ttb, uta, utb, x, y, xp, yp)[0]
        self.assertAlmostEqual(rc2t[0][0], -0.1810332128306279253, places=12)
        self.assertAlmostEqual(rc2t[0][1], 0.9834769806938520084, places=12)
        self.assertAlmostEqual(rc2t[0][2], 0.6555551248057665829e-4, places=12)

        self.assertAlmostEqual(rc2t[1][0], -0.9834768134136142314, places=12)
        self.assertAlmostEqual(rc2t[1][1], -0.1810332203649529312, places=12)
        self.assertAlmostEqual(rc2t[1][2], 0.5749800843594139912e-3, places=12)

        self.assertAlmostEqual(rc2t[2][0], 0.5773474028619264494e-3, places=12)
        self.assertAlmostEqual(rc2t[2][1], 0.3961816546911624260e-4, places=12)
        self.assertAlmostEqual(rc2t[2][2], 0.9999998325501746670, places=12)

    def test_eo06a(self):
        eo = erfa.eo06a(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(eo, -0.1332882371941833644e-2, 15)
        
    def test_eors(self):
        r = np.array([
            ((0.9999989440476103608,-0.1332881761240011518e-2,-0.5790767434730085097e-3),
             (0.1332858254308954453e-2,0.9999991109044505944,-0.4097782710401555759e-4),
             (0.5791308472168153320e-3,0.4020595661593994396e-4,0.9999998314954572365))
            ])
        s = np.array([-0.1220040848472271978e-7])
        eo = erfa.eors(r, s)[0]
        self.assertAlmostEqual(eo, -0.1332882715130744606e-2, 15)

    def test_fw2m(self):
        gamb = np.array([-0.2243387670997992368e-5])
        phib = np.array([0.4091014602391312982])
        psi = np.array([-0.9501954178013015092e-3])
        eps = np.array([0.4091014316587367472])
        r = erfa.fw2m(gamb, phib, psi, eps)[0]
        self.assertAlmostEqual(r[0][0], 0.9999995505176007047, places=12)
        self.assertAlmostEqual(r[0][1], 0.8695404617348192957e-3, places=12)
        self.assertAlmostEqual(r[0][2], 0.3779735201865582571e-3, places=12)

        self.assertAlmostEqual(r[1][0], -0.8695404723772016038e-3, places=12)
        self.assertAlmostEqual(r[1][1], 0.9999996219496027161, places=12)
        self.assertAlmostEqual(r[1][2], -0.1361752496887100026e-6, places=12)

        self.assertAlmostEqual(r[2][0], -0.3779734957034082790e-3, places=12)
        self.assertAlmostEqual(r[2][1], -0.1924880848087615651e-6, places=12)
        self.assertAlmostEqual(r[2][2], 0.9999999285679971958, places=12)

    def test_fw2xy(self):
        gamb = np.array([-0.2243387670997992368e-5])
        phib = np.array([0.4091014602391312982])
        psi = np.array([-0.9501954178013015092e-3])
        eps = np.array([0.4091014316587367472])
        x, y = erfa.fw2xy(gamb, phib, psi, eps)
        self.assertAlmostEqual(x[0], -0.3779734957034082790e-3, 14)
        self.assertAlmostEqual(y[0], -0.1924880848087615651e-6, 14)

    def test_num00a(self):
        rmatn = erfa.num00a(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rmatn[0][0], 0.9999999999536227949, places=12)
        self.assertAlmostEqual(rmatn[0][1], 0.8836238544090873336e-5, places=12)
        self.assertAlmostEqual(rmatn[0][2], 0.3830835237722400669e-5, places=12)

        self.assertAlmostEqual(rmatn[1][0], -0.8836082880798569274e-5, places=12)
        self.assertAlmostEqual(rmatn[1][1], 0.9999999991354655028, places=12)
        self.assertAlmostEqual(rmatn[1][2], -0.4063240865362499850e-4, places=12)

        self.assertAlmostEqual(rmatn[2][0], -0.3831194272065995866e-5, places=12)
        self.assertAlmostEqual(rmatn[2][1], 0.4063237480216291775e-4, places=12)
        self.assertAlmostEqual(rmatn[2][2], 0.9999999991671660338, places=12)

    def test_num00b(self):
        rmatn = erfa.num00b(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rmatn[0][0], 0.9999999999536069682, places=12)
        self.assertAlmostEqual(rmatn[0][1], 0.8837746144871248011e-5, places=12)
        self.assertAlmostEqual(rmatn[0][2], 0.3831488838252202945e-5, places=12)

        self.assertAlmostEqual(rmatn[1][0], -0.8837590456632304720e-5, places=12)
        self.assertAlmostEqual(rmatn[1][1], 0.9999999991354692733, places=12)
        self.assertAlmostEqual(rmatn[1][2], -0.4063198798559591654e-4, places=12)

        self.assertAlmostEqual(rmatn[2][0], -0.3831847930134941271e-5, places=12)
        self.assertAlmostEqual(rmatn[2][1], 0.4063195412258168380e-4, places=12)
        self.assertAlmostEqual(rmatn[2][2], 0.9999999991671806225, places=12)

    def test_num06a(self):
        rmatn = erfa.num06a(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rmatn[0][0], 0.9999999999536227668, places=12)
        self.assertAlmostEqual(rmatn[0][1], 0.8836241998111535233e-5, places=12)
        self.assertAlmostEqual(rmatn[0][2], 0.3830834608415287707e-5, places=12)

        self.assertAlmostEqual(rmatn[1][0], -0.8836086334870740138e-5, places=12)
        self.assertAlmostEqual(rmatn[1][1], 0.9999999991354657474, places=12)
        self.assertAlmostEqual(rmatn[1][2], -0.4063240188248455065e-4, places=12)

        self.assertAlmostEqual(rmatn[2][0], -0.3831193642839398128e-5, places=12)
        self.assertAlmostEqual(rmatn[2][1], 0.4063236803101479770e-4, places=12)
        self.assertAlmostEqual(rmatn[2][2], 0.9999999991671663114, places=12)

    def test_numat(self):
        epsa =  np.array([0.4090789763356509900])
        dpsi = np.array([-0.9630909107115582393e-5])
        deps =  np.array([0.4063239174001678826e-4])
        rmatn = erfa.numat(epsa, dpsi, deps)[0]
        self.assertAlmostEqual(rmatn[0][0], 0.9999999999536227949, places=12)
        self.assertAlmostEqual(rmatn[0][1], 0.8836239320236250577e-5, places=12)
        self.assertAlmostEqual(rmatn[0][2], 0.3830833447458251908e-5, places=12)

        self.assertAlmostEqual(rmatn[1][0], -0.8836083657016688588e-5, places=12)
        self.assertAlmostEqual(rmatn[1][1], 0.9999999991354654959, places=12)
        self.assertAlmostEqual(rmatn[1][2], -0.4063240865361857698e-4, places=12)

        self.assertAlmostEqual(rmatn[2][0], -0.3831192481833385226e-5, places=12)
        self.assertAlmostEqual(rmatn[2][1], 0.4063237480216934159e-4, places=12)
        self.assertAlmostEqual(rmatn[2][2], 0.9999999991671660407, places=12)

    def test_nut00a(self):
        dpsi, deps = erfa.nut00a(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9630909107115518431e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4063239174001678710e-4, 13)
        
    def test_nut00b(self):
        dpsi, deps = erfa.nut00b(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9632552291148362783e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4063197106621159367e-4, 13)

    def test_nut06a(self):
        dpsi, deps = erfa.nut06a(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9630912025820308797e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4063238496887249798e-4, 13)
                
    def test_nut80(self):
        dpsi, deps = erfa.nut80(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9643658353226563966e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4060051006879713322e-4, 13)

    def test_nutm80(self):
        rmatn = erfa.nutm80(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(rmatn[0][0], 0.9999999999534999268, places=12)
        self.assertAlmostEqual(rmatn[0][1], 0.8847935789636432161e-5, places=12)
        self.assertAlmostEqual(rmatn[0][2], 0.3835906502164019142e-5, places=12)

        self.assertAlmostEqual(rmatn[1][0], -0.8847780042583435924e-5, places=12)
        self.assertAlmostEqual(rmatn[1][1], 0.9999999991366569963, places=12)
        self.assertAlmostEqual(rmatn[1][2], -0.4060052702727130809e-4, places=12)

        self.assertAlmostEqual(rmatn[2][0], -0.3836265729708478796e-5, places=12)
        self.assertAlmostEqual(rmatn[2][1], 0.4060049308612638555e-4, places=12)
        self.assertAlmostEqual(rmatn[2][2], 0.9999999991684415129, places=12)

    def test_obl06(self):
        obl = erfa.obl06(np.array([2400000.5]), np.array([54388.0]))[0]
        self.assertAlmostEqual(obl, 0.4090749229387258204, 14)
        
    def test_obl80(self):
        obl = erfa.obl80(np.array([2400000.5]), np.array([54388.0]))[0]
        self.assertAlmostEqual(obl, 0.4090751347643816218, 14)
        
    def test_p06e(self):
        eps0,psia,oma,bpa,bqa,pia,bpia,epsa,chia,za,zetaa,thetaa,pa,gam,phi,psi = erfa.p06e(np.array([2400000.5]), np.array([52541.0]))
        self.assertAlmostEqual(eps0[0], 0.4090926006005828715, places=14)
        self.assertAlmostEqual(psia[0], 0.6664369630191613431e-3, places=14)
        self.assertAlmostEqual(oma[0], 0.4090925973783255982, places=14)
        self.assertAlmostEqual(bpa[0], 0.5561149371265209445e-6, places=14)
        self.assertAlmostEqual(bqa[0], -0.6191517193290621270e-5, places=14)
        self.assertAlmostEqual(pia[0], 0.6216441751884382923e-5, places=14)
        self.assertAlmostEqual(bpia[0], 3.052014180023779882, places=14)
        self.assertAlmostEqual(epsa[0], 0.4090864054922431688, places=14)
        self.assertAlmostEqual(chia[0], 0.1387703379530915364e-5, places=14)
        self.assertAlmostEqual(za[0], 0.2921789846651790546e-3, places=14)
        self.assertAlmostEqual(zetaa[0], 0.3178773290332009310e-3, places=14)
        self.assertAlmostEqual(thetaa[0], 0.2650932701657497181e-3, places=14)
        self.assertAlmostEqual(pa[0], 0.6651637681381016344e-3, places=14)
        self.assertAlmostEqual(gam[0], 0.1398077115963754987e-5, places=14)
        self.assertAlmostEqual(phi[0], 0.4090864090837462602, places=14)
        self.assertAlmostEqual(psi[0], 0.6664464807480920325e-3, places=14)

    def test_pb06(self):
        bzeta, bz, btheta = erfa.pb06(np.array([2400000.5]), np.array([50123.9999]))
        self.assertAlmostEqual(bzeta[0], -0.5092634016326478238e-3, places=12)
        self.assertAlmostEqual(bz[0], -0.3602772060566044413e-3, places=12)
        self.assertAlmostEqual(btheta[0], -0.3779735537167811177e-3, places=12)

    def test_pfw06(self):
        gamb, phib, psib, epsa = erfa.pfw06(np.array([2400000.5]), np.array([50123.9999]))
        self.assertAlmostEqual(gamb[0], -0.2243387670997995690e-5, places=16)
        self.assertAlmostEqual(phib[0],  0.4091014602391312808, places=12)
        self.assertAlmostEqual(psib[0], -0.9501954178013031895e-3, places=14)
        self.assertAlmostEqual(epsa[0],  0.4091014316587367491, places=12)

    def test_pmat00(self):
        rbp = erfa.pmat00(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rbp[0][0], 0.9999995505175087260, places=12)
        self.assertAlmostEqual(rbp[0][1], 0.8695405883617884705e-3, places=14)
        self.assertAlmostEqual(rbp[0][2], 0.3779734722239007105e-3, places=14)

        self.assertAlmostEqual(rbp[1][0], -0.8695405990410863719e-3, places=14)
        self.assertAlmostEqual(rbp[1][1], 0.9999996219494925900, places=12)
        self.assertAlmostEqual(rbp[1][2], -0.1360775820404982209e-6, places=14)

        self.assertAlmostEqual(rbp[2][0], -0.3779734476558184991e-3, places=14)
        self.assertAlmostEqual(rbp[2][1], -0.1925857585832024058e-6, places=14)
        self.assertAlmostEqual(rbp[2][2], 0.9999999285680153377, places=12)

    def test_pmat06(self):
        rbp = erfa.pmat06(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rbp[0][0], 0.9999995505176007047, places=12)
        self.assertAlmostEqual(rbp[0][1], 0.8695404617348208406e-3, places=14)
        self.assertAlmostEqual(rbp[0][2], 0.3779735201865589104e-3, places=14)
        self.assertAlmostEqual(rbp[1][0], -0.8695404723772031414e-3, places=14)
        self.assertAlmostEqual(rbp[1][1], 0.9999996219496027161, places=12)
        self.assertAlmostEqual(rbp[1][2], -0.1361752497080270143e-6, places=14)
        self.assertAlmostEqual(rbp[2][0], -0.3779734957034089490e-3, places=14)
        self.assertAlmostEqual(rbp[2][1], -0.1924880847894457113e-6, places=14)
        self.assertAlmostEqual(rbp[2][2], 0.9999999285679971958, places=12)

    def test_pmat76(self):
        rmatp = erfa.pmat76(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rmatp[0][0], 0.9999995504328350733, places=12)
        self.assertAlmostEqual(rmatp[0][1], 0.8696632209480960785e-3, places=14)
        self.assertAlmostEqual(rmatp[0][2], 0.3779153474959888345e-3, places=14)
        self.assertAlmostEqual(rmatp[1][0], -0.8696632209485112192e-3, places=14)
        self.assertAlmostEqual(rmatp[1][1], 0.9999996218428560614, places=12)
        self.assertAlmostEqual(rmatp[1][2], -0.1643284776111886407e-6, places=14)
        self.assertAlmostEqual(rmatp[2][0], -0.3779153474950335077e-3, places=14)
        self.assertAlmostEqual(rmatp[2][1], -0.1643306746147366896e-6, places=14)
        self.assertAlmostEqual(rmatp[2][2], 0.9999999285899790119, places=12)

    def test_pn00(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        dpsi = np.array([-0.9632552291149335877e-5])
        deps = np.array([0.4063197106621141414e-4])
        epsa, rb, rp, rbp, rn, rbpn = erfa.pn00(d1, d2, dpsi, deps)
        self.assertAlmostEqual(epsa[0], 0.4090791789404229916, 12)
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942498, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078279744199196626e-7, places=18)
        self.assertAlmostEqual(rb[0][0][2], 0.8056217146976134152e-7, places=18)
        self.assertAlmostEqual(rb[0][1][0], 0.7078279477857337206e-7, places=18)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3306041454222136517e-7, places=18)
        self.assertAlmostEqual(rb[0][2][0], -0.8056217380986972157e-7, places=18)
        self.assertAlmostEqual(rb[0][2][1], -0.3306040883980552500e-7, places=18)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999989300532289018, places=12)
        self.assertAlmostEqual(rp[0][0][1], -0.1341647226791824349e-2, places=14)
        self.assertAlmostEqual(rp[0][0][2], -0.5829880927190296547e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], 0.1341647231069759008e-2, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999990999908750433, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.3837444441583715468e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], 0.5829880828740957684e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.3984203267708834759e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999998300623538046, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999989300052243993, places=12)
        self.assertAlmostEqual(rbp[0][0][1], -0.1341717990239703727e-2, places=14)
        self.assertAlmostEqual(rbp[0][0][2], -0.5829075749891684053e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], 0.1341718013831739992e-2, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999990998959191343, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.3505759733565421170e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], 0.5829075206857717883e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.4315219955198608970e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999998301093036269, places=12)
        self.assertAlmostEqual(rn[0][0][0], 0.9999999999536069682, places=12)
        self.assertAlmostEqual(rn[0][0][1], 0.8837746144872140812e-5, places=16)
        self.assertAlmostEqual(rn[0][0][2], 0.3831488838252590008e-5, places=16)
        self.assertAlmostEqual(rn[0][1][0], -0.8837590456633197506e-5, places=16)
        self.assertAlmostEqual(rn[0][1][1], 0.9999999991354692733, places=12)
        self.assertAlmostEqual(rn[0][1][2], -0.4063198798559573702e-4, places=15) #failed at 16
        self.assertAlmostEqual(rn[0][2][0], -0.3831847930135328368e-5, places=16)
        self.assertAlmostEqual(rn[0][2][1], 0.4063195412258150427e-4, places=16)
        self.assertAlmostEqual(rn[0][2][2], 0.9999999991671806225, places=12)
        self.assertAlmostEqual(rbpn[0][0][0], 0.9999989440499982806, places=12)
        self.assertAlmostEqual(rbpn[0][0][1], -0.1332880253640848301e-2, places=14)
        self.assertAlmostEqual(rbpn[0][0][2], -0.5790760898731087295e-3, places=14)
        self.assertAlmostEqual(rbpn[0][1][0], 0.1332856746979948745e-2, places=14)
        self.assertAlmostEqual(rbpn[0][1][1], 0.9999991109064768883, places=12)
        self.assertAlmostEqual(rbpn[0][1][2], -0.4097740555723063806e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][0], 0.5791301929950205000e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2][1], 0.4020553681373702931e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][2], 0.9999998314958529887, places=12)

    def test_pn00a(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = erfa.pn00a(d1, d2)
        self.assertAlmostEqual(dpsi[0], -0.9630909107115518431e-5, places=12)
        self.assertAlmostEqual(deps[0],  0.4063239174001678710e-4, places=12)
        self.assertAlmostEqual(epsa[0],  0.4090791789404229916, places=12)
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942498, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078279744199196626e-7, places=16)
        self.assertAlmostEqual(rb[0][0][2], 0.8056217146976134152e-7, places=16)
        self.assertAlmostEqual(rb[0][1][0], 0.7078279477857337206e-7, places=16)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3306041454222136517e-7, places=16)
        self.assertAlmostEqual(rb[0][2][0], -0.8056217380986972157e-7, places=16)
        self.assertAlmostEqual(rb[0][2][1], -0.3306040883980552500e-7, places=16)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999989300532289018, places=12)
        self.assertAlmostEqual(rp[0][0][1], -0.1341647226791824349e-2, places=14)
        self.assertAlmostEqual(rp[0][0][2], -0.5829880927190296547e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], 0.1341647231069759008e-2, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999990999908750433, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.3837444441583715468e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], 0.5829880828740957684e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.3984203267708834759e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999998300623538046, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999989300052243993, places=12)
        self.assertAlmostEqual(rbp[0][0][1], -0.1341717990239703727e-2, places=14)
        self.assertAlmostEqual(rbp[0][0][2], -0.5829075749891684053e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], 0.1341718013831739992e-2, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999990998959191343, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.3505759733565421170e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], 0.5829075206857717883e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.4315219955198608970e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999998301093036269, places=12)
        self.assertAlmostEqual(rn[0][0][0], 0.9999999999536227949, places=12)
        self.assertAlmostEqual(rn[0][0][1], 0.8836238544090873336e-5, places=14)
        self.assertAlmostEqual(rn[0][0][2], 0.3830835237722400669e-5, places=14)
        self.assertAlmostEqual(rn[0][1][0], -0.8836082880798569274e-5, places=14)
        self.assertAlmostEqual(rn[0][1][1], 0.9999999991354655028, places=12)
        self.assertAlmostEqual(rn[0][1][2], -0.4063240865362499850e-4, places=14)
        self.assertAlmostEqual(rn[0][2][0], -0.3831194272065995866e-5, places=14)
        self.assertAlmostEqual(rn[0][2][1], 0.4063237480216291775e-4, places=14)
        self.assertAlmostEqual(rn[0][2][2], 0.9999999991671660338, places=12)
        self.assertAlmostEqual(rbpn[0][0][0], 0.9999989440476103435, places=12)
        self.assertAlmostEqual(rbpn[0][0][1], -0.1332881761240011763e-2, places=14)
        self.assertAlmostEqual(rbpn[0][0][2], -0.5790767434730085751e-3, places=14)
        self.assertAlmostEqual(rbpn[0][1][0], 0.1332858254308954658e-2, places=14)
        self.assertAlmostEqual(rbpn[0][1][1], 0.9999991109044505577, places=12)
        self.assertAlmostEqual(rbpn[0][1][2], -0.4097782710396580452e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][0], 0.5791308472168152904e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2][1], 0.4020595661591500259e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][2], 0.9999998314954572304, places=12)

    def test_pn00b(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = erfa.pn00b(d1, d2)
        self.assertAlmostEqual(dpsi[0], -0.9632552291148362783e-5, places=12)
        self.assertAlmostEqual(deps[0],  0.4063197106621159367e-4, places=12)
        self.assertAlmostEqual(epsa[0],  0.4090791789404229916, places=12)
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942498, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078279744199196626e-7, places=16)
        self.assertAlmostEqual(rb[0][0][2], 0.8056217146976134152e-7, places=16)
        self.assertAlmostEqual(rb[0][1][0], 0.7078279477857337206e-7, places=16)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3306041454222136517e-7, places=16)
        self.assertAlmostEqual(rb[0][2][0], -0.8056217380986972157e-7, places=16)
        self.assertAlmostEqual(rb[0][2][1], -0.3306040883980552500e-7, places=16)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999989300532289018, places=12)
        self.assertAlmostEqual(rp[0][0][1], -0.1341647226791824349e-2, places=14)
        self.assertAlmostEqual(rp[0][0][2], -0.5829880927190296547e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], 0.1341647231069759008e-2, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999990999908750433, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.3837444441583715468e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], 0.5829880828740957684e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.3984203267708834759e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999998300623538046, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999989300052243993, places=12)
        self.assertAlmostEqual(rbp[0][0][1], -0.1341717990239703727e-2, places=14)
        self.assertAlmostEqual(rbp[0][0][2], -0.5829075749891684053e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], 0.1341718013831739992e-2, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999990998959191343, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.3505759733565421170e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], 0.5829075206857717883e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.4315219955198608970e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999998301093036269, places=12)
        self.assertAlmostEqual(rn[0][0][0], 0.9999999999536069682, places=12)
        self.assertAlmostEqual(rn[0][0][1], 0.8837746144871248011e-5, places=14)
        self.assertAlmostEqual(rn[0][0][2], 0.3831488838252202945e-5, places=14)
        self.assertAlmostEqual(rn[0][1][0], -0.8837590456632304720e-5, places=14)
        self.assertAlmostEqual(rn[0][1][1], 0.9999999991354692733, places=12)
        self.assertAlmostEqual(rn[0][1][2], -0.4063198798559591654e-4, places=14)
        self.assertAlmostEqual(rn[0][2][0], -0.3831847930134941271e-5, places=14)
        self.assertAlmostEqual(rn[0][2][1], 0.4063195412258168380e-4, places=14)
        self.assertAlmostEqual(rn[0][2][2], 0.9999999991671806225, places=12)
        self.assertAlmostEqual(rbpn[0][0][0], 0.9999989440499982806, places=12)
        self.assertAlmostEqual(rbpn[0][0][1], -0.1332880253640849194e-2, places=14)
        self.assertAlmostEqual(rbpn[0][0][2], -0.5790760898731091166e-3, places=14)
        self.assertAlmostEqual(rbpn[0][1][0], 0.1332856746979949638e-2, places=14)
        self.assertAlmostEqual(rbpn[0][1][1], 0.9999991109064768883, places=12)
        self.assertAlmostEqual(rbpn[0][1][2], -0.4097740555723081811e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][0], 0.5791301929950208873e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2][1], 0.4020553681373720832e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][2], 0.9999998314958529887, places=12)

    def test_pn06(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        dpsi = np.array([-0.9632552291149335877e-5])
        deps = np.array([0.4063197106621141414e-4])
        epsa, rb, rp, rbp, rn, rbpn = erfa.pn06(d1, d2, dpsi, deps)
        self.assertAlmostEqual(epsa[0], 0.4090789763356509926, places=12)
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942497, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078368960971557145e-7, places=14)
        self.assertAlmostEqual(rb[0][0][2], 0.8056213977613185606e-7, places=14)
        self.assertAlmostEqual(rb[0][1][0], 0.7078368694637674333e-7, places=14)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3305943742989134124e-7, places=14)
        self.assertAlmostEqual(rb[0][2][0], -0.8056214211620056792e-7, places=14)
        self.assertAlmostEqual(rb[0][2][1], -0.3305943172740586950e-7, places=14)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999989300536854831, places=12)
        self.assertAlmostEqual(rp[0][0][1], -0.1341646886204443795e-2, places=14)
        self.assertAlmostEqual(rp[0][0][2], -0.5829880933488627759e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], 0.1341646890569782183e-2, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999990999913319321, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.3835944216374477457e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], 0.5829880833027867368e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.3985701514686976112e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999998300623534950, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999989300056797893, places=12)
        self.assertAlmostEqual(rbp[0][0][1], -0.1341717650545059598e-2, places=14)
        self.assertAlmostEqual(rbp[0][0][2], -0.5829075756493728856e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], 0.1341717674223918101e-2, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999990998963748448, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.3504269280170069029e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], 0.5829075211461454599e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.4316708436255949093e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999998301093032943, places=12)
        self.assertAlmostEqual(rn[0][0][0], 0.9999999999536069682, places=12)
        self.assertAlmostEqual(rn[0][0][1], 0.8837746921149881914e-5, places=14)
        self.assertAlmostEqual(rn[0][0][2], 0.3831487047682968703e-5, places=14)
        self.assertAlmostEqual(rn[0][1][0], -0.8837591232983692340e-5, places=14)
        self.assertAlmostEqual(rn[0][1][1], 0.9999999991354692664, places=12)
        self.assertAlmostEqual(rn[0][1][2], -0.4063198798558931215e-4, places=14)
        self.assertAlmostEqual(rn[0][2][0], -0.3831846139597250235e-5, places=14)
        self.assertAlmostEqual(rn[0][2][1], 0.4063195412258792914e-4, places=14)
        self.assertAlmostEqual(rn[0][2][2], 0.9999999991671806293, places=12)
        self.assertAlmostEqual(rbpn[0][0][0], 0.9999989440504506688, places=12)
        self.assertAlmostEqual(rbpn[0][0][1], -0.1332879913170492655e-2, places=14)
        self.assertAlmostEqual(rbpn[0][0][2], -0.5790760923225655753e-3, places=14)
        self.assertAlmostEqual(rbpn[0][1][0], 0.1332856406595754748e-2, places=14)
        self.assertAlmostEqual(rbpn[0][1][1], 0.9999991109069366795, places=12)
        self.assertAlmostEqual(rbpn[0][1][2], -0.4097725651142641812e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][0], 0.5791301952321296716e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2][1], 0.4020538796195230577e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][2], 0.9999998314958576778, places=12)

    def test_pn06a(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = erfa.pn06a(d1, d2)
        self.assertAlmostEqual(dpsi[0], -0.9630912025820308797e-5, places=12)
        self.assertAlmostEqual(deps[0],  0.4063238496887249798e-4, places=12)
        self.assertAlmostEqual(epsa[0],  0.4090789763356509926, places=12)
        self.assertAlmostEqual(rb[0][0][0], 0.9999999999999942497, places=12)
        self.assertAlmostEqual(rb[0][0][1], -0.7078368960971557145e-7, places=14)
        self.assertAlmostEqual(rb[0][0][2], 0.8056213977613185606e-7, places=14)
        self.assertAlmostEqual(rb[0][1][0], 0.7078368694637674333e-7, places=14)
        self.assertAlmostEqual(rb[0][1][1], 0.9999999999999969484, places=12)
        self.assertAlmostEqual(rb[0][1][2], 0.3305943742989134124e-7, places=14)
        self.assertAlmostEqual(rb[0][2][0], -0.8056214211620056792e-7, places=14)
        self.assertAlmostEqual(rb[0][2][1], -0.3305943172740586950e-7, places=14)
        self.assertAlmostEqual(rb[0][2][2], 0.9999999999999962084, places=12)
        self.assertAlmostEqual(rp[0][0][0], 0.9999989300536854831, places=12)
        self.assertAlmostEqual(rp[0][0][1], -0.1341646886204443795e-2, places=14)
        self.assertAlmostEqual(rp[0][0][2], -0.5829880933488627759e-3, places=14)
        self.assertAlmostEqual(rp[0][1][0], 0.1341646890569782183e-2, places=14)
        self.assertAlmostEqual(rp[0][1][1], 0.9999990999913319321, places=12)
        self.assertAlmostEqual(rp[0][1][2], -0.3835944216374477457e-6, places=14)
        self.assertAlmostEqual(rp[0][2][0], 0.5829880833027867368e-3, places=14)
        self.assertAlmostEqual(rp[0][2][1], -0.3985701514686976112e-6, places=14)
        self.assertAlmostEqual(rp[0][2][2], 0.9999998300623534950, places=12)
        self.assertAlmostEqual(rbp[0][0][0], 0.9999989300056797893, places=12)
        self.assertAlmostEqual(rbp[0][0][1], -0.1341717650545059598e-2, places=14)
        self.assertAlmostEqual(rbp[0][0][2], -0.5829075756493728856e-3, places=14)
        self.assertAlmostEqual(rbp[0][1][0], 0.1341717674223918101e-2, places=14)
        self.assertAlmostEqual(rbp[0][1][1], 0.9999990998963748448, places=12)
        self.assertAlmostEqual(rbp[0][1][2], -0.3504269280170069029e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][0], 0.5829075211461454599e-3, places=14)
        self.assertAlmostEqual(rbp[0][2][1], -0.4316708436255949093e-6, places=14)
        self.assertAlmostEqual(rbp[0][2][2], 0.9999998301093032943, places=12)
        self.assertAlmostEqual(rn[0][0][0], 0.9999999999536227668, places=12)
        self.assertAlmostEqual(rn[0][0][1], 0.8836241998111535233e-5, places=14)
        self.assertAlmostEqual(rn[0][0][2], 0.3830834608415287707e-5, places=14)
        self.assertAlmostEqual(rn[0][1][0], -0.8836086334870740138e-5, places=14)
        self.assertAlmostEqual(rn[0][1][1], 0.9999999991354657474, places=12)
        self.assertAlmostEqual(rn[0][1][2], -0.4063240188248455065e-4, places=14)
        self.assertAlmostEqual(rn[0][2][0], -0.3831193642839398128e-5, places=14)
        self.assertAlmostEqual(rn[0][2][1], 0.4063236803101479770e-4, places=14)
        self.assertAlmostEqual(rn[0][2][2], 0.9999999991671663114, places=12)
        self.assertAlmostEqual(rbpn[0][0][0], 0.9999989440480669738, places=12)
        self.assertAlmostEqual(rbpn[0][0][1], -0.1332881418091915973e-2, places=14)
        self.assertAlmostEqual(rbpn[0][0][2], -0.5790767447612042565e-3, places=14)
        self.assertAlmostEqual(rbpn[0][1][0], 0.1332857911250989133e-2, places=14)
        self.assertAlmostEqual(rbpn[0][1][1], 0.9999991109049141908, places=12)
        self.assertAlmostEqual(rbpn[0][1][2], -0.4097767128546784878e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][0], 0.5791308482835292617e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2][1], 0.4020580099454020310e-4, places=14)
        self.assertAlmostEqual(rbpn[0][2][2], 0.9999998314954628695, places=12)

    def test_pnm00a(self):
        rbpn = erfa.pnm00a(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rbpn[0][0], 0.9999995832793134257, places=12)
        self.assertAlmostEqual(rbpn[0][1], 0.8372384254137809439e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2], 0.3639684306407150645e-3, places=14)
        self.assertAlmostEqual(rbpn[1][0], -0.8372535226570394543e-3, places=14)
        self.assertAlmostEqual(rbpn[1][1], 0.9999996486491582471, places=12)
        self.assertAlmostEqual(rbpn[1][2], 0.4132915262664072381e-4, places=14)
        self.assertAlmostEqual(rbpn[2][0], -0.3639337004054317729e-3, places=14)
        self.assertAlmostEqual(rbpn[2][1], -0.4163386925461775873e-4, places=14)
        self.assertAlmostEqual(rbpn[2][2], 0.9999999329094390695, places=12)

    def test_pnm00b(self):
        rbpn = erfa.pnm00b(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rbpn[0][0], 0.9999995832776208280, places=12)
        self.assertAlmostEqual(rbpn[0][1], 0.8372401264429654837e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2], 0.3639691681450271771e-3, places=14)
        self.assertAlmostEqual(rbpn[1][0], -0.8372552234147137424e-3, places=14)
        self.assertAlmostEqual(rbpn[1][1], 0.9999996486477686123, places=12)
        self.assertAlmostEqual(rbpn[1][2], 0.4132832190946052890e-4, places=14)
        self.assertAlmostEqual(rbpn[2][0], -0.3639344385341866407e-3, places=14)
        self.assertAlmostEqual(rbpn[2][1], -0.4163303977421522785e-4, places=14)
        self.assertAlmostEqual(rbpn[2][2], 0.9999999329092049734, places=12)

    def test_pnm06a(self):
        rbpn = erfa.pnm06a(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rbpn[0][0], 0.9999995832794205484, places=12)
        self.assertAlmostEqual(rbpn[0][1], 0.8372382772630962111e-3, places=14)
        self.assertAlmostEqual(rbpn[0][2], 0.3639684771140623099e-3, places=14)
        self.assertAlmostEqual(rbpn[1][0], -0.8372533744743683605e-3, places=14)
        self.assertAlmostEqual(rbpn[1][1], 0.9999996486492861646, places=12)
        self.assertAlmostEqual(rbpn[1][2], 0.4132905944611019498e-4, places=14)
        self.assertAlmostEqual(rbpn[2][0], -0.3639337469629464969e-3, places=14)
        self.assertAlmostEqual(rbpn[2][1], -0.4163377605910663999e-4, places=14)
        self.assertAlmostEqual(rbpn[2][2], 0.9999999329094260057, places=12)

    def test_pnm80(self):
        rmatpn = erfa.pnm80(np.array([2400000.5]), np.array([50123.9999]))[0]
        self.assertAlmostEqual(rmatpn[0][0], 0.9999995831934611169, places=12)
        self.assertAlmostEqual(rmatpn[0][1], 0.8373654045728124011e-3, places=14)
        self.assertAlmostEqual(rmatpn[0][2], 0.3639121916933106191e-3, places=14)
        self.assertAlmostEqual(rmatpn[1][0], -0.8373804896118301316e-3, places=14)
        self.assertAlmostEqual(rmatpn[1][1], 0.9999996485439674092, places=12)
        self.assertAlmostEqual(rmatpn[1][2], 0.4130202510421549752e-4, places=14)
        self.assertAlmostEqual(rmatpn[2][0], -0.3638774789072144473e-3, places=14)
        self.assertAlmostEqual(rmatpn[2][1], -0.4160674085851722359e-4, places=14)
        self.assertAlmostEqual(rmatpn[2][2], 0.9999999329310274805, places=12)

    def test_pom00(self):
        xp = np.array([2.55060238e-7])
        yp = np.array([1.860359247e-6])
        sp = np.array([-0.1367174580728891460e-10])
        rpom = erfa.pom00(xp, yp, sp)[0]
        self.assertAlmostEqual(rpom[0][0], 0.9999999999999674721, places=12)
        self.assertAlmostEqual(rpom[0][1], -0.1367174580728846989e-10, places=16)
        self.assertAlmostEqual(rpom[0][2], 0.2550602379999972345e-6, places=16)
        self.assertAlmostEqual(rpom[1][0], 0.1414624947957029801e-10, places=16)
        self.assertAlmostEqual(rpom[1][1], 0.9999999999982695317, places=12)
        self.assertAlmostEqual(rpom[1][2], -0.1860359246998866389e-5, places=16)
        self.assertAlmostEqual(rpom[2][0], -0.2550602379741215021e-6, places=16)
        self.assertAlmostEqual(rpom[2][1], 0.1860359247002414021e-5, places=16)
        self.assertAlmostEqual(rpom[2][2], 0.9999999999982370039, places=12)

    def test_pr00(self):
        dpsipr, depspr = erfa.pr00(np.array([2400000.5]), np.array([53736]))
        self.assertAlmostEqual(dpsipr[0], -0.8716465172668347629e-7, places=22)
        self.assertAlmostEqual(depspr[0], -0.7342018386722813087e-8, places=22)

    def test_prec76(self):
        ep01 = np.array([2400000.5])
        ep02 = np.array([33282.0])
        ep11 = np.array([2400000.5])
        ep12 = np.array([51544.0])
        zeta, z, theta = erfa.prec76(ep01, ep02, ep11, ep12)
        self.assertAlmostEqual(zeta[0],  0.5588961642000161243e-2, places=12)
        self.assertAlmostEqual(z[0],     0.5589922365870680624e-2, places=12)
        self.assertAlmostEqual(theta[0], 0.4858945471687296760e-2, places=12)

    def test_s00(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        s = erfa.s00(d1, d2, x, y)
        self.assertAlmostEqual(s[0], -0.1220036263270905693e-7, 18)

    def test_s06(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        s = erfa.s06(d1, d2, x, y)[0]
        self.assertAlmostEqual(s, -0.1220032213076463117e-7, 18)
        
    def test_sp00(self):
        s = erfa.sp00(np.array([2400000.5]), np.array([52541.0]))
        self.assertAlmostEqual(s[0], -0.6216698469981019309e-11, 12)

    def test_xys00a(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x, y, s = erfa.xys00a(d1, d2)
        self.assertAlmostEqual(x[0], 0.5791308472168152904e-3, places=16)
        self.assertAlmostEqual(y[0], 0.4020595661591500259e-4, places=17)
        self.assertAlmostEqual(s[0], -0.1220040848471549623e-7, places=20)

    def test_xy06(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x, y = erfa.xy06(d1, d2)
        self.assertAlmostEqual(x[0], 0.5791308486706010975e-3, places=16)
        self.assertAlmostEqual(y[0], 0.4020579816732958141e-4, places=17)

    def test_xys06a(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x, y, s = erfa.xys06a(d1, d2)
        self.assertAlmostEqual(x[0], 0.5791308482835292617e-3, places=16)
        self.assertAlmostEqual(y[0], 0.4020580099454020310e-4, places=15)
        self.assertAlmostEqual(s[0], -0.1220032294164579896e-7, places=19)

## Astronomy/RotationAndTime
    def test_ee00(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        epsa = np.array([0.4090789763356509900])
        dpsi = np.array([-0.9630909107115582393e-5])
        ee = erfa.ee00(d1, d2, epsa, dpsi)[0]
        self.assertAlmostEqual(ee, -0.8834193235367965479e-5, 18)

    def test_eqeq94(self):
        ee = erfa.eqeq94(np.array([2400000.5]), np.array([41234.0]))
        self.assertAlmostEqual(ee[0], 0.5357758254609256894e-4, 17)

    def test_era00(self):
        era = erfa.era00(np.array([2400000.5]), np.array([54388.0]))
        self.assertAlmostEqual(era[0], 0.4022837240028158102, 12)

    def test_gmst00(self):
        uta = np.array([2400000.5])
        utb = np.array([53736.0])
        tta = np.array([2400000.5])
        ttb = np.array([53736.0])
        g = erfa.gmst00(uta, utb, tta, ttb)[0]
        self.assertAlmostEqual(g, 1.754174972210740592, 14)
        
    def test_gmst06(self):
        uta = np.array([2400000.5])
        utb = np.array([53736.0])
        tta = np.array([2400000.5])
        ttb = np.array([53736.0])
        g = erfa.gmst06(uta, utb, tta, ttb)[0]
        self.assertAlmostEqual(g, 1.754174971870091203, 14)
        
    def test_gmst82(self):
        g = erfa.gmst82(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(g, 1.754174981860675096, 14)

    def test_gst00a(self):
        uta = np.array([2400000.5])
        utb = np.array([53736.0])
        tta = np.array([2400000.5])
        ttb = np.array([53736.0])
        g = erfa.gst00a(uta, utb, tta, ttb)[0]
        self.assertAlmostEqual(g, 1.754166138018281369, 14)

    def test_gst00b(self):
        g = erfa.gst00b(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(g, 1.754166136510680589, 14)

    def test_gst06(self):
        uta = np.array([2400000.5])
        utb = np.array([53736.0])
        tta = np.array([2400000.5])
        ttb = np.array([53736.0])
        rnpb = np.array([
            ((0.9999989440476103608,-0.1332881761240011518e-2,-0.5790767434730085097e-3),
             (0.1332858254308954453e-2,0.9999991109044505944,-0.4097782710401555759e-4),
             (0.5791308472168153320e-3,0.4020595661593994396e-4,0.9999998314954572365))
            ])
        g = erfa.gst06(uta, utb, tta, ttb, rnpb)[0]
        self.assertAlmostEqual(g, 1.754166138018167568, 14)
        
    def test_gst06a(self):
        uta = np.array([2400000.5])
        utb = np.array([53736.0])
        tta = np.array([2400000.5])
        ttb = np.array([53736.0])
        g = erfa.gst06a(uta, utb, tta, ttb)[0]
        self.assertAlmostEqual(g, 1.754166137675019159, 14)
        
    def test_gst94(self):
        g = erfa.gst94(np.array([2400000.5]), np.array([53736.0]))[0]
        self.assertAlmostEqual(g, 1.754166136020645203, 14)

## VectorMatrix/AngleOps
    def test_a2af(self):
        a = np.array([2.345])
        r = erfa.a2af(4, a)[0]
        self.assertEqual(r[0], '+')
        self.assertEqual(r[1][0], 134)
        self.assertEqual(r[1][1], 21)
        self.assertEqual(r[1][2], 30)
        self.assertEqual(r[1][3], 9706)

    def test_a2tf(self):
        a = np.array([-3.01234])
        r = erfa.a2tf(4, a)[0]
        self.assertEqual(r[0], '-')
        self.assertEqual(r[1][0], 11)
        self.assertEqual(r[1][1], 30)
        self.assertEqual(r[1][2], 22)
        self.assertEqual(r[1][3], 6484)

    def test_af2a(self):
        a = np.array([[-45, 13, 27.2], [45, 13, 27.2]])
        r = erfa.af2a(a)
        self.assertAlmostEqual(r[0], -0.7893115794313644842, 15)
        self.assertAlmostEqual(r[1], 0.7893115794313644842, 15)

## VectorMatrix/BuildRotations 
    def test_rx(self):
        phi = np.array([0.3456789])
        r = np.array([
            [[2.0,3.0,2.0],
             [3.0,2.0,3.0],
             [3.0,4.0,5.0]]
            ])
        r = erfa.rx(phi, r)[0]
        self.assertAlmostEqual(r[0][0], 2.0, 0.0)
        self.assertAlmostEqual(r[0][1], 3.0, 0.0)
        self.assertAlmostEqual(r[0][2], 2.0, 0.0)
        self.assertAlmostEqual(r[1][0], 3.839043388235612460, places=12)
        self.assertAlmostEqual(r[1][1], 3.237033249594111899, places=12)
        self.assertAlmostEqual(r[1][2], 4.516714379005982719, places=12)
        self.assertAlmostEqual(r[2][0], 1.806030415924501684, places=12)
        self.assertAlmostEqual(r[2][1], 3.085711545336372503, places=12)
        self.assertAlmostEqual(r[2][2], 3.687721683977873065, places=12)

    def test_rxp(self):
        r = np.array([
            ((2.0,3.0,2.0),
             (3.0,2.0,3.0),
             (3.0,4.0,5.0))
            ])
        p = np.array([
            (0.2,1.5,0.1)
            ])
        rp = erfa.rxp(r, p)[0]
        self.assertAlmostEqual(rp[0], 5.1, places=12)
        self.assertAlmostEqual(rp[1], 3.9, places=12)
        self.assertAlmostEqual(rp[2], 7.1, places=12)

    def test_ry(self):
        theta = np.array([0.3456789])
        r = np.array([
            [[2.0,3.0,2.0],
             [3.0,2.0,3.0],
             [3.0,4.0,5.0]]
            ])
        r = erfa.ry(theta, r)[0]
        self.assertAlmostEqual(r[0][0], 0.8651847818978159930, places=12)
        self.assertAlmostEqual(r[0][1], 1.467194920539316554, places=12)
        self.assertAlmostEqual(r[0][2], 0.1875137911274457342, places=12)
        self.assertAlmostEqual(r[1][0], 3, places=12)
        self.assertAlmostEqual(r[1][1], 2, places=12)
        self.assertAlmostEqual(r[1][2], 3, places=12)
        self.assertAlmostEqual(r[2][0], 3.500207892850427330, places=12)
        self.assertAlmostEqual(r[2][1], 4.779889022262298150, places=12)
        self.assertAlmostEqual(r[2][2], 5.381899160903798712, places=12)

    def test_rz(self):
        psi = np.array([0.3456789])
        r = np.array([
            [[2.0,3.0,2.0],
             [3.0,2.0,3.0],
             [3.0,4.0,5.0]]
            ])
        r = erfa.rz(psi, r)[0]
        self.assertAlmostEqual(r[0][0], 2.898197754208926769, places=12)
        self.assertAlmostEqual(r[0][1], 3.500207892850427330, places=12)
        self.assertAlmostEqual(r[0][2], 2.898197754208926769, places=12)
        self.assertAlmostEqual(r[1][0], 2.144865911309686813, places=12)
        self.assertAlmostEqual(r[1][1], 0.865184781897815993, places=12)
        self.assertAlmostEqual(r[1][2], 2.144865911309686813, places=12)
        self.assertAlmostEqual(r[2][0], 3.0, places=12)
        self.assertAlmostEqual(r[2][1], 4.0, places=12)
        self.assertAlmostEqual(r[2][2], 5.0, places=12)


## VectorMatrix/CopyExtendExtract
    def test_cr(self):
        a = np.array([
            [[2.0,3.0,2.0],
             [3.0,2.0,3.0],
             [3.0,4.0,5.0]]
            ])
        c = erfa.cr(a)[0]
        self.assertAlmostEqual(c[0][0], 2.0, places=12)
        self.assertAlmostEqual(c[0][1], 3.0, places=12)
        self.assertAlmostEqual(c[0][2], 2.0, places=12)
        self.assertAlmostEqual(c[1][0], 3.0, places=12)
        self.assertAlmostEqual(c[1][1], 2.0, places=12)
        self.assertAlmostEqual(c[1][2], 3.0, places=12)
        self.assertAlmostEqual(c[2][0], 3.0, places=12)
        self.assertAlmostEqual(c[2][1], 4.0, places=12)
        self.assertAlmostEqual(c[2][2], 5.0, places=12)

## VectorMatrix/MatrixOps
    def test_rxr(self):
        a = np.array([
            [[2.0,3.0,2.0],
             [3.0,2.0,3.0],
             [3.0,4.0,5.0]]
            ])
        b = np.array([
            [[1.0,2.0,2.0],
             [4.0,1.0,1.0],
             [3.0,0.0,1.0]]
            ])
        atb = erfa.rxr(a, b)[0]
        self.assertAlmostEqual(atb[0][0], 20.0, places=12)
        self.assertAlmostEqual(atb[0][1],  7.0, places=12)
        self.assertAlmostEqual(atb[0][2],  9.0, places=12)
        self.assertAlmostEqual(atb[1][0], 20.0, places=12)
        self.assertAlmostEqual(atb[1][1],  8.0, places=12)
        self.assertAlmostEqual(atb[1][2], 11.0, places=12)
        self.assertAlmostEqual(atb[2][0], 34.0, places=12)
        self.assertAlmostEqual(atb[2][1], 10.0, places=12)
        self.assertAlmostEqual(atb[2][2], 15.0, places=12)

    def test_anp(self):
        r = erfa.anp(np.array([-0.1]))
        self.assertAlmostEqual(r[0], 6.183185307179586477, 15)
        

support.run_unittest(Validate)

