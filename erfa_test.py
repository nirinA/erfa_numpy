import unittest
try:
    from test import support
except ImportError:
    from test import test_support as support
import math
import numpy as np
import erfa
import _erfa

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

    def test_besselian_epoch_jd(self):
        dj0, dj1 = erfa.besselian_epoch_jd(np.array([1957.3]))
        self.assertAlmostEqual(dj0[0], 2400000.5, 9)
        self.assertAlmostEqual(dj1[0], 35948.1915101513, 9)

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

## Astronomy/Timescales
    def test_d_tai_utc(self):
        d = erfa.d_tai_utc(np.array([2003]), np.array([6]), np.array([1]), np.array([0.0]))
        self.assertAlmostEqual(d[0], 32.0, 9)
        d = erfa.d_tai_utc(np.array([2008]), np.array([1]), np.array([17]), np.array([0.0]))
        self.assertAlmostEqual(d[0], 33.0, 9)

## Astronomy/PrecNutPolar
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

    def test_nut80(self):
        dpsi, deps = erfa.nut80(np.array([2400000.5]), np.array([53736.0]))
        self.assertAlmostEqual(dpsi[0], -0.9643658353226563966e-5, 13)
        self.assertAlmostEqual(deps[0], 0.4060051006879713322e-4, 13)

    def test_obl80(self):
        obl = erfa.obl80(np.array([2400000.5]), np.array([54388.0]))
        self.assertAlmostEqual(obl[0], 0.4090751347643816218, 14)
##
    def test_plan94(self):
        self.assertRaises(_erfa.error, _erfa.plan94, np.array([2400000.5]), np.array([-320000]), 0)
        self.assertRaises(_erfa.error, _erfa.plan94, np.array([2400000.5]), np.array([-320000]), 10)

        pv = _erfa.plan94(np.array([2400000.5]), np.array([-320000]), 3)[0]
        self.assertAlmostEqual(pv[0][0], 0.9308038666832975759, places=11)
        self.assertAlmostEqual(pv[0][1], 0.3258319040261346000, places=11)
        self.assertAlmostEqual(pv[0][2], 0.1422794544481140560, places=11)

        self.assertAlmostEqual(pv[1][0], -0.6429458958255170006e-2, places=11)
        self.assertAlmostEqual(pv[1][1], 0.1468570657704237764e-1, places=11)
        self.assertAlmostEqual(pv[1][2], 0.6406996426270981189e-2, places=11)

        pv = _erfa.plan94(np.array([2400000.5]), np.array([43999.9]), 1)[0]
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

    def test_s00(self):
        x = np.array([0.5791308486706011000e-3])
        y = np.array([0.4020579816732961219e-4])
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        s = erfa.s00(d1, d2, x, y)
        self.assertAlmostEqual(s[0], -0.1220036263270905693e-7, 18)

    def test_xys06a(self):
        d1 = np.array([2400000.5])
        d2 = np.array([53736.0])
        x, y, s = erfa.xys06a(d1, d2)
        self.assertAlmostEqual(x[0], 0.5791308482835292617e-3, places=16)
        self.assertAlmostEqual(y[0], 0.4020580099454020310e-4, places=15)
        self.assertAlmostEqual(s[0], -0.1220032294164579896e-7, places=19)

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

## Astronomy/RotationAndTime
    def test_eqeq94(self):
        ee = erfa.eqeq94(np.array([2400000.5]), np.array([41234.0]))
        self.assertAlmostEqual(ee[0], 0.5357758254609256894e-4, 17)

support.run_unittest(Validate)

