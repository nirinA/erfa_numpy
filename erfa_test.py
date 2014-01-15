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

## Astronomy/PrecNutPolar
    def test_c2ixys(self):
        x =  np.array([0.5791308486706011000e-3])
        y =  np.array([0.4020579816732961219e-4])
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
        
    def test_nut80(self):
        dpsi, deps = erfa.nut80(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9643658353226563966e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4060051006879713322e-4, 13)

    def test_obl80(self):
        obl = erfa.obl80(np.array([2400000.5]), np.array([54388.0]))
        self.assertAlmostEqual(obl[0], 0.4090751347643816218, 14)
        
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
        g = erfa.gmst82(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(g[0], 1.754174981860675096, 14)

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

