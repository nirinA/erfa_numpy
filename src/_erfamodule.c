#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structseq.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "erfa.h"
#include "erfam.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/arrayobject.h"

static PyObject *_erfaError;

static int initialized;
/* local prototype */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptr(double **v);

static PyTypeObject AstromType;

static PyTypeObject LdbodyType;
/* local function */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)
{
    double **c, *a;
    int i, n, m;
    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    c = ptrvector(n);
    /* pointer to arrayin data as double */
    a = (double *)arrayin->data;
    for (i=0;i<n;i++) {
        c[i] = a + i*m;
    }
    return c;
}

double **ptrvector(long n)
{
    double **v;
    v = (double **)malloc((size_t)(n *sizeof(double)));
    if (!v) {
        PyErr_SetString(_erfaError, "malloc failed in **ptrvector");
        exit(0);
    }
    return v;
}

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)
{
    /* pointer to arrayin data as double */
    return (double *) arrayin->data; 
}

void free_Carrayptr(double **v){
    free((char *)v);
}

static PyObject *
_to_py_vector(double v[3])
{
    double *cv;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp dims[] = {3};
    int j;
    pyout=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                                                 dsc,
                                                 1,
                                                 dims,
                                                 NULL,
                                                 NULL,
                                                 0,
                                                 NULL);
    if (NULL == pyout)  return NULL;
    cv = pyvector_to_Carrayptrs(pyout);
    for(j=0;j<3;j++) cv[j] = v[j];
    Py_INCREF(pyout); 
    return PyArray_Return(pyout);                     
}

static PyObject *
_to_py_posvel(double pv[2][3])
{
    double **cpv;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp dims[] = {2,3};
    int i, j;
    pyout=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                                                 dsc,
                                                 2,
                                                 dims,
                                                 NULL,
                                                 NULL,
                                                 0,
                                                 NULL);
    if (NULL == pyout)  return NULL;
    cpv = pymatrix_to_Carrayptrs(pyout);
    for (i=0;i<2;i++) {
        for(j=0;j<3;j++) {
            cpv[i][j] = pv[i][j];
        }
    }
    //free?
    Py_INCREF(pyout); 
    return PyArray_Return(pyout);                     
}

static PyObject *
_to_py_matrix(double m[3][3])
{
    double **cm;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp dims[] = {3,3};
    int i, j;
    pyout=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                                                 dsc,
                                                 2,
                                                 dims,
                                                 NULL,
                                                 NULL,
                                                 0,
                                                 NULL);
    if (NULL == pyout)  return NULL;
    cm = pymatrix_to_Carrayptrs(pyout);
    for (i=0;i<3;i++) {
        for(j=0;j<3;j++) {
            cm[i][j] = m[i][j];
        }
    }
    //free?
    Py_INCREF(pyout); 
    return PyArray_Return(pyout);                     
}

static PyObject *
_to_py_astrom(eraASTROM *a)
{
    PyObject *v = PyStructSequence_New(&AstromType);
    if (v == NULL)
        return NULL;
#define SET(i,val) PyStructSequence_SET_ITEM(v, i, PyFloat_FromDouble((double) val))
    SET(0, a->pmt);
    PyStructSequence_SET_ITEM(v, 1, _to_py_vector(a->eb));
    PyStructSequence_SET_ITEM(v, 2, _to_py_vector(a->eh));
    SET(3, a->em);
    PyStructSequence_SET_ITEM(v, 4, _to_py_vector(a->v));
    SET(5, a->bm1);
    PyStructSequence_SET_ITEM(v, 6, _to_py_matrix(a->bpn));
    SET(7, a->along);
    SET(8, a->phi);
    SET(9, a->xpl);
    SET(10, a->ypl);
    SET(11, a->sphi);
    SET(12, a->cphi);
    SET(13, a->diurab);
    SET(14, a->eral);
    SET(15, a->refa);
    SET(16, a->refb);
    if (PyErr_Occurred()) {
        Py_XDECREF(v);
        return NULL;
    }
#undef SET
    return v;
}

static PyStructSequence_Field ASTROM_type_fields[] = {
    {"pmt", "PM time interval (SSB, Julian years)"},
    {"eb", "SSB to observer (vector, au)"},
    {"eh", "Sun to observer (unit vector)"},
    {"em", "distance from Sun to observer (au)"},
    {"v", "barycentric observer velocity (vector, c"},
    {"bm1", "sqrt(1-|v|^2): reciprocal of Lorenz factor"},
    {"bpn", "bias-precession-nutation matrix"},
    {"along", "longitude + s' + dERA(DUT) (radians)"},
    {"phi", "geodetic latitude (radians)"},
    {"xpl", "polar motion xp wrt local meridian (radians)"},
    {"ypl", "polar motion yp wrt local meridian (radians)"},
    {"sphi", "sine of geodetic latitude"},
    {"cphi", "cosine of geodetic latitude"},
    {"diurab", "magnitude of diurnal aberration vector"},
    {"eral", "local Earth rotation angle (radians)"},
    {"refa", "refraction constant A (radians)"},
    {"refb", "refraction constant B (radians)"},
    {0}
};

static PyStructSequence_Desc ASTROM_type_desc = {
    "_erfa.ASTROM",
    "Star-independent astrometry parameters\n"
"(Vectors eb, eh, em and v are all with respect to BCRS axes.)",
    ASTROM_type_fields,
    17,
};

static PyStructSequence_Field LDBODY_type_fields[] = {
    {"bm", "mass of the body (solar masses)"},
    {"dl", "deflection limiter (radians^2/2)"},
    {"pv", "barycentric PV of the body (au, au/day)"},
    {0}
};

static PyStructSequence_Desc LDBODY_type_desc = {
    "_erfa.LDBODY",
    "Body parameters for light deflection",
    LDBODY_type_fields,
    3,
};
static PyObject *
_erfa_ab(PyObject *self, PyObject *args)
{
    double *pnat, *v, *s, *bm1, *ppr;
    PyObject *pypnat, *pyv, *pys, *pybm1;
    PyObject *apnat, *av, *as, *abm1;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pypnat, &PyArray_Type, &pyv,
                                 &PyArray_Type, &pys, &PyArray_Type, &pybm1))
        return NULL;
    apnat = PyArray_FROM_OTF(pypnat, NPY_DOUBLE, NPY_IN_ARRAY);
    av = PyArray_FROM_OTF(pyv, NPY_DOUBLE, NPY_IN_ARRAY);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_IN_ARRAY);
    abm1 = PyArray_FROM_OTF(pybm1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (apnat == NULL || av == NULL || as == NULL || abm1 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(apnat);
    dims = PyArray_DIMS(apnat);
    if (dims[0] != PyArray_DIMS(av)[0] ||
        dims[0] != PyArray_DIMS(as)[0] ||
        dims[0] != PyArray_DIMS(abm1)[0]) {
        PyErr_SetString(_erfaError, "arguments have not the same shape");
        return NULL;
    }    
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) {
        Py_DECREF(pyout);
        return NULL;
    }
    pnat = (double *)PyArray_DATA(apnat);
    v = (double *)PyArray_DATA(av);
    s = (double *)PyArray_DATA(as);
    bm1 = (double *)PyArray_DATA(abm1);
    ppr = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        eraAb(&pnat[i*3], &v[i*3], s[i], bm1[i], &ppr[i*3]);
    }
    Py_DECREF(apnat);
    Py_DECREF(av);
    Py_DECREF(as);
    Py_DECREF(abm1);
    //Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(apnat);
    Py_XDECREF(av);
    Py_XDECREF(as);
    Py_XDECREF(abm1);
    return NULL;
}

PyDoc_STRVAR(_erfa_ab_doc,
"\nab(pnat[3], v[3], s, bm1) -> ppr\n"
"Apply aberration to transform natural direction into proper direction.\n"
"Given:\n"
"    pnat       natural direction to the source (unit vector)\n"
"    v          observer barycentric velocity in units of c\n"
"    s          distance between the Sun and the observer (au)\n"
"    bm1        sqrt(1-|v|^2): reciprocal of Lorenz factor\n"
"Returned:\n"
"    ppr        proper direction to source (unit vector)");

static PyObject *
_erfa_epb2jd(PyObject *self, PyObject *args)
{
    double *epb, *jd0, *jd1;
    PyObject *pyepb;
    PyObject *aepb;
    PyArrayObject *pyjd0, *pyjd1;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyepb))
        return NULL;
    aepb = PyArray_FROM_OTF(pyepb, NPY_DOUBLE, NPY_IN_ARRAY);
    if (aepb == NULL) {
        Py_DECREF(aepb);
        return NULL;
    }
    ndim = PyArray_NDIM(aepb);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        return NULL;
    }            
    dims = PyArray_DIMS(aepb);
    pyjd0 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyjd1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyjd0 || NULL == pyjd1)  return NULL;
    epb = (double *)PyArray_DATA(aepb);
    jd0 = (double *)PyArray_DATA(pyjd0);
    jd1 = (double *)PyArray_DATA(pyjd1);
    for (i=0;i<dims[0];i++) {
        eraEpb2jd(epb[i], &jd0[i], &jd1[i]);
    }
    return Py_BuildValue("OO", pyjd0, pyjd1);
}

PyDoc_STRVAR(_erfa_epb2jd_doc,
"\nepb2jd(epb) -> 2400000.5 djm\n"
"Given:\n"
"    epb        Besselian Epoch,\n"
"Returned:\n"
"    djm        Modified Julian Date");

static PyObject *
_erfa_apcs(PyObject *self, PyObject *args)
{
    double *date1, *date2, pv[2][3], ebpv[2][3], ehp[3];
    PyObject *adate1, *adate2, *apv, *aebpv, *aehp;
    PyObject *pydate1, *pydate2, *pypv, *pyebpv, *pyehp;
    PyObject *pv_iter = NULL, *ebpv_iter = NULL, *ehp_iter = NULL;
    PyObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, d1[] = {3}, d2[] = {2,3}; //d1[2], d2[3];
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2,
                                 &PyArray_Type, &pypv,
                                 &PyArray_Type, &pyebpv,
                                 &PyArray_Type, &pyehp))      
        return NULL;
    adate1 = PyArray_FROM_OTF(pydate1, NPY_DOUBLE, NPY_IN_ARRAY);
    adate2 = PyArray_FROM_OTF(pydate2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (adate1 == NULL || adate2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(adate1);
    dims = PyArray_DIMS(adate1);
/*    d1[0] = dims[0];
    d1[1] = 3;
    d2[0] = dims[0];
    d2[1] = 2;
    d2[2] = 3;*/
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    date1 = (double *)PyArray_DATA(adate1);
    date2 = (double *)PyArray_DATA(adate2);

    pv_iter = PyArray_IterNew((PyObject *)pypv);
    ebpv_iter = PyArray_IterNew((PyObject *)pyebpv);
    ehp_iter = PyArray_IterNew((PyObject *)pyehp);
    if (pv_iter == NULL || ebpv_iter == NULL || ehp_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k,l;
        double e, p, h;
        for (l=0;l<3;l++) {
            aehp = PyArray_GETITEM(pyehp, PyArray_ITER_DATA(ehp_iter));
            Py_INCREF(aehp);
            h = (double)PyFloat_AsDouble(aehp);
            if (h == -1 && PyErr_Occurred()) return NULL;
            ehp[l] = h;
            PyArray_ITER_NEXT(ehp_iter);
        }
        for (j=0;j<2;j++) {
            for (k=0;k<3;k++) {
                apv = PyArray_GETITEM(pypv, PyArray_ITER_DATA(pv_iter));
                aebpv = PyArray_GETITEM(pyebpv, PyArray_ITER_DATA(ebpv_iter));
                
                if (apv == NULL || aebpv == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(apv);Py_INCREF(aebpv);
                p = (double)PyFloat_AsDouble(apv);
                if (p == -1 && PyErr_Occurred()) return NULL;
                pv[j][k] = p;
                e = (double)PyFloat_AsDouble(aebpv);
                if (e == -1 && PyErr_Occurred()) return NULL;
                ebpv[j][k] = e;
                PyArray_ITER_NEXT(pv_iter); 
                PyArray_ITER_NEXT(ebpv_iter); 
            }
        }
        eraApcs(date1[i], date2[i], pv, ebpv, ehp, &astrom);
        if (PyList_SetItem(pyout, i, _to_py_astrom(&astrom))) {
            PyErr_SetString(_erfaError, "cannot set astrom into list");
            goto fail;
        }       
    }
    Py_DECREF(adate1);
    Py_DECREF(adate2);
    Py_DECREF(pv_iter);
    Py_DECREF(ebpv_iter);
    Py_DECREF(ehp_iter);
    Py_INCREF(pyout);
    return (PyObject*)pyout;//_to_py_astrom(&astrom); //

fail:
    Py_XDECREF(adate1);
    Py_XDECREF(adate2);
    Py_XDECREF(pv_iter);
    Py_XDECREF(ebpv_iter);
    Py_XDECREF(ehp_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_apcs_doc,
"\napcs(date1, date2, pv[2][3], ebpv[2][3], ehp[3]) -> astrom\n"
"For an observer whose geocentric position and velocity are known,\n"
"prepare star-independent astrometry parameters for transformations\n"
"between ICRS and GCRS.  The Earth ephemeris is supplied by the caller.\n"
"\n"
"The parameters produced by this function are required in the\n"
"parallax, light deflection and aberration parts of the astrometric\n"
"transformation chain.\n"
"Given:\n"
"    date1      TDB as a 2-part...\n"
"    date2      ...Julian Date\n"
"    pv         observer's geocentric pos/vel (m, m/s)\n"
"    ebpv       Earth barycentric pos/vel (au, au/day)\n"
"    ehp        Earth heliocentric position (au)\n"
"Returned:\n"
"    astrom     star-independent astrometry parameters");

static PyObject *
_erfa_pmsafe(PyObject *self, PyObject *args)
{
    double *ra1, *dec1, *pmr1, *pmd1, *px1, *rv1, *ep1a, *ep1b, *ep2a, *ep2b;
    double *ra2, *dec2, *pmr2, *pmd2, *px2, *rv2;
    PyObject *pyra1, *pydec1, *pypmr1, *pypmd1, *pypx1, *pyrv1, *pyep1a, *pyep1b, *pyep2a, *pyep2b;
    PyObject *ara1, *adec1, *apmr1, *apmd1, *apx1, *arv1, *aep1a, *aep1b, *aep2a, *aep2b;
    PyArrayObject *pyra2, *pydec2, *pypmr2, *pypmd2, *pypx2, *pyrv2;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i, j;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!",
                                 &PyArray_Type, &pyra1, &PyArray_Type, &pydec1,
                                 &PyArray_Type, &pypmr1, &PyArray_Type, &pypmd1,
                                 &PyArray_Type, &pypx1, &PyArray_Type, &pyrv1,
                                 &PyArray_Type, &pyep1a, &PyArray_Type, &pyep1b,
                                 &PyArray_Type, &pyep2a, &PyArray_Type, &pyep2b))
        return NULL;
    ara1 = PyArray_FROM_OTF(pyra1, NPY_DOUBLE, NPY_IN_ARRAY);
    adec1 = PyArray_FROM_OTF(pydec1, NPY_DOUBLE, NPY_IN_ARRAY);
    apmr1 = PyArray_FROM_OTF(pypmr1, NPY_DOUBLE, NPY_IN_ARRAY);
    apmd1 = PyArray_FROM_OTF(pypmd1, NPY_DOUBLE, NPY_IN_ARRAY);
    apx1 = PyArray_FROM_OTF(pypx1, NPY_DOUBLE, NPY_IN_ARRAY);
    arv1 = PyArray_FROM_OTF(pyrv1, NPY_DOUBLE, NPY_IN_ARRAY);
    aep1a = PyArray_FROM_OTF(pyep1a, NPY_DOUBLE, NPY_IN_ARRAY);
    aep1b = PyArray_FROM_OTF(pyep1b, NPY_DOUBLE, NPY_IN_ARRAY);
    aep2a = PyArray_FROM_OTF(pyep2a, NPY_DOUBLE, NPY_IN_ARRAY);
    aep2b = PyArray_FROM_OTF(pyep2b, NPY_DOUBLE, NPY_IN_ARRAY);
    if (ara1 == NULL || adec1 == NULL || apmr1 == NULL || apmd1 == NULL || apx1 == NULL ||
        arv1 == NULL || aep1a == NULL || aep1b == NULL || aep2a == NULL || aep2b == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ara1);
    dims = PyArray_DIMS(ara1);
    if (dims[0] != PyArray_DIMS(ara1)[0] || dims[0] != PyArray_DIMS(adec1)[0] ||
        dims[0] != PyArray_DIMS(apmr1)[0] || dims[0] != PyArray_DIMS(apmd1)[0] ||
        dims[0] != PyArray_DIMS(apx1)[0] || dims[0] != PyArray_DIMS(arv1)[0] ||
        dims[0] != PyArray_DIMS(aep1a)[0] || dims[0] != PyArray_DIMS(aep1b)[0] ||
        dims[0] != PyArray_DIMS(aep2a)[0] || dims[0] != PyArray_DIMS(aep2b)[0]) {
        PyErr_SetString(_erfaError, "arguments have not the same shape");
        return NULL;
    }    

    pyra2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydec2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypmr2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypmd2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypx2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyrv2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyra2 || NULL == pydec2 || NULL == pypmr2 ||
        NULL == pypmd2 || NULL == pypx2 || NULL == pyrv2)  return NULL;

    ra1 = (double *)PyArray_DATA(ara1);
    dec1 = (double *)PyArray_DATA(adec1);
    pmr1 = (double *)PyArray_DATA(apmr1);
    pmd1 = (double *)PyArray_DATA(apmd1);
    px1 = (double *)PyArray_DATA(apx1);
    rv1 = (double *)PyArray_DATA(arv1);
    ep1a = (double *)PyArray_DATA(aep1a);
    ep1b = (double *)PyArray_DATA(aep1b);
    ep2a = (double *)PyArray_DATA(aep2a);
    ep2b = (double *)PyArray_DATA(aep2b);

    ra2 = (double *)PyArray_DATA(pyra2);
    dec2 = (double *)PyArray_DATA(pydec2);
    pmr2 = (double *)PyArray_DATA(pypmr2);
    pmd2 = (double *)PyArray_DATA(pypmd2);
    px2 = (double *)PyArray_DATA(pypx2);
    rv2 = (double *)PyArray_DATA(pyrv2);

    for (i=0;i<dims[0];i++) {
        j = eraPmsafe(ra1[i], dec1[i], pmr1[i], pmd1[i], px1[i],
                      rv1[i], ep1a[i], ep1b[i], ep2a[i], ep2b[i],
                      &ra2[i], &dec2[i], &pmr2[i], &pmd2[i], &px2[i], &rv2[i]);
        if (j == -1) {
            PyErr_SetString(_erfaError, "system error (should not occur)");
            return NULL;
        }
        else if (j == 1) {
            PyErr_SetString(_erfaError, "distance overridden");
            return NULL;
        }
        else if (j == 2) {
            PyErr_SetString(_erfaError, "excessive velocity");
            return NULL;
        }
        else if (j == 4) {
            PyErr_SetString(_erfaError, "solution didn't converge");
            return NULL;
        }
    }
    Py_DECREF(ara1);
    Py_DECREF(adec1);
    Py_DECREF(apmr1);
    Py_DECREF(apmd1);
    Py_DECREF(apx1);
    Py_DECREF(arv1);
    Py_DECREF(aep1a);
    Py_DECREF(aep1b);
    Py_DECREF(aep2a);
    Py_DECREF(aep2b);
    return Py_BuildValue("OOOOOO", pyra2, pydec2, pypmr2, pypmd2, pypx2, pyrv2);

fail:
    Py_XDECREF(ara1);
    Py_XDECREF(adec1);
    Py_XDECREF(apmr1);
    Py_XDECREF(apmd1);
    Py_XDECREF(apx1);
    Py_XDECREF(arv1);
    Py_XDECREF(aep1a);
    Py_XDECREF(aep1b);
    Py_XDECREF(aep2a);
    Py_XDECREF(aep2b);
    return NULL;
}

PyDoc_STRVAR(_erfa_pmsafe_doc,
"\npmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b, -> ra2, dec2, pmr2, pmd2, px2, rv2)\n"
"Star proper motion:  update star catalog data for space motion, with\n"
"special handling to handle the zero parallax case.\n"
"Given:\n"
"    ra1    right ascension (radians), before\n"
"    dec1   declination (radians), before\n"
"    pmr1   RA proper motion (radians/year), before\n"
"    pmd1   Dec proper motion (radians/year), before\n"
"    px1    parallax (arcseconds), before\n"
"    rv1    radial velocity (km/s, +ve = receding), before\n"
"    ep1a   ''before'' epoch, part A\n"
"    ep1b   ''before'' epoch, part B\n"
"    ep2a   ''after'' epoch, part A\n"
"    ep2b   ''after'' epoch, part B\n"
"Returned:\n"
"    ra2    right ascension (radians), after\n"
"    dec2   declination (radians), after\n"
"    pmr2   RA proper motion (radians/year), after\n"
"    pmd2   Dec proper motion (radians/year), after\n"
"    px2    parallax (arcseconds), after\n"
"    rv2    radial velocity (km/s, +ve = receding), after");

static PyObject *
_erfa_plan94(PyObject *self, PyObject *args)
{
    double *d1, *d2, **cpv, pv[2][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2, *pv_result;
    PyArrayObject *pyout = NULL;
    int np, status;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dims_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &pyd1, &PyArray_Type, &pyd2, &np)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dims_out[0] = dims[0];
    dims_out[1] = 2;
    dims_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dims_out, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    cpv = (double **)PyArray_DATA(pyout);
    if (NULL == cpv) {
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        status = eraPlan94(d1[i], d2[i], np, &cpv[i*6]);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal np,  not in range(1,8) for planet");
            return NULL;
        }
        switch (status) {
        case 1:
            PyErr_WarnEx(PyExc_Warning, "year outside range(1000:3000)", 1);
            break;
        case 2:
            PyErr_WarnEx(PyExc_Warning,  "computation failed to converge", 1);
            break;
        default:
            break;
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(cpv);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(cpv);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_plan94_doc,
"\nplan94(d1, d2, np) -> pv\n\n"
"Approximate heliocentric position and velocity of a nominated major\n"
"planet:  Mercury, Venus, EMB, Mars, Jupiter, Saturn, Uranus or\n"
"Neptune (but not the Earth itself).\n"
"Given:\n"
"    d1         TDB date part A\n"
"    d2         TDB date part B\n"
"    np         planet (1=Mercury, 2=Venus, 3=EMB, 4=Mars,\n"
"                       5=Jupiter, 6=Saturn, 7=Uranus, 8=Neptune)\n"
"Returned:\n"
"    pv         planet p,v (heliocentric, J2000.0, AU,AU/d)");

static PyObject *
_erfa_s00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *x, *y, *s;
    PyObject *pyd1, *pyd2, *pyx, *pyy;
    PyObject *ad1, *ad2, *ax, *ay;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pyx, &PyArray_Type, &pyy))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || ax == NULL || ay == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad1)[0] || dims[0] != PyArray_DIMS(ad2)[0] ||
        dims[0] != PyArray_DIMS(ax)[0] || dims[0] != PyArray_DIMS(ay)[0]) {
        PyErr_SetString(_erfaError, "arguments have not the same shape");
        return NULL;
    }    

    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) {
        Py_DECREF(pyout);
        return NULL;
    }
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    s = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        s[i] = eraS00(d1[i], d2[i], x[i], y[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(ax);
    Py_DECREF(ay);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_s00_doc,
"\ns00(d1, d2, x, y) -> s\n\n"
"The CIO locator s, positioning the Celestial Intermediate Origin on\n"
"the equator of the Celestial Intermediate Pole, given the CIP's X,Y\n"
"coordinates.  Compatible with IAU 2000A precession-nutation.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"   x,y     CIP coordinates\n"
"Returned:\n"
"   s       the CIO locator s in radians");


static PyMethodDef _erfa_methods[] = {
    {"ab", _erfa_ab, METH_VARARGS, _erfa_ab_doc},
    {"epb2jd", _erfa_epb2jd, METH_VARARGS, _erfa_epb2jd_doc},
    {"apcs", _erfa_apcs, METH_VARARGS, _erfa_apcs_doc},
    {"pmsafe", _erfa_pmsafe, METH_VARARGS, _erfa_pmsafe_doc},
    {"plan94", _erfa_plan94, METH_VARARGS, _erfa_plan94_doc},
    {"s00", _erfa_s00, METH_VARARGS, _erfa_s00_doc},
    {NULL,		NULL}		/* sentinel */
};

PyDoc_STRVAR(module_doc,
"This module provides ERFA,\n\
the Essential Routine for Fundamental Astronomy,\n\
interface to Python\n\
");

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef _erfamodule = {
	PyModuleDef_HEAD_INIT,
	"_erfa",
	module_doc,
	-1,
	_erfa_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__erfa(void)
{
	PyObject *m;
	m = PyModule_Create(&_erfamodule);
	import_array();
	if (m == NULL)
            return NULL;
#else
PyMODINIT_FUNC
init_erfa(void)
{
	PyObject *m;
	m = Py_InitModule3("_erfa", _erfa_methods, module_doc);
	import_array();
	if (m == NULL)
            goto finally;
#endif
        _erfaError = PyErr_NewException("_erfa.error", NULL, NULL);
        Py_INCREF(_erfaError);
        PyModule_AddObject(m, "error", _erfaError);
        PyModule_AddObject(m, "DPI", PyFloat_FromDouble(ERFA_DPI));
        PyModule_AddObject(m, "D2PI", PyFloat_FromDouble(ERFA_D2PI));
        PyModule_AddObject(m, "DR2D", PyFloat_FromDouble(ERFA_DR2D));
        PyModule_AddObject(m, "DD2R", PyFloat_FromDouble(ERFA_DD2R));
        PyModule_AddObject(m, "DR2AS", PyFloat_FromDouble(ERFA_DR2AS));
        PyModule_AddObject(m, "DAS2R", PyFloat_FromDouble(ERFA_DAS2R));
        PyModule_AddObject(m, "DS2R", PyFloat_FromDouble(ERFA_DS2R));
        PyModule_AddObject(m, "TURNAS", PyFloat_FromDouble(ERFA_TURNAS));
        PyModule_AddObject(m, "DMAS2R", PyFloat_FromDouble(ERFA_DMAS2R));
        PyModule_AddObject(m, "DTY", PyFloat_FromDouble(ERFA_DTY));
        PyModule_AddObject(m, "DAYSEC", PyFloat_FromDouble(ERFA_DAYSEC));
        PyModule_AddObject(m, "DJY", PyFloat_FromDouble(ERFA_DJY));
        PyModule_AddObject(m, "DJC", PyFloat_FromDouble(ERFA_DJC));
        PyModule_AddObject(m, "DJM", PyFloat_FromDouble(ERFA_DJM));
        PyModule_AddObject(m, "DJ00", PyFloat_FromDouble(ERFA_DJ00));
        PyModule_AddObject(m, "DJM0", PyFloat_FromDouble(ERFA_DJM0));
        PyModule_AddObject(m, "DJM00", PyFloat_FromDouble(ERFA_DJM00));
        PyModule_AddObject(m, "DJM77", PyFloat_FromDouble(ERFA_DJM77));
        PyModule_AddObject(m, "TTMTAI", PyFloat_FromDouble(ERFA_TTMTAI));
        PyModule_AddObject(m, "DAU", PyFloat_FromDouble(ERFA_DAU));
        PyModule_AddObject(m, "CMPS", PyFloat_FromDouble(ERFA_CMPS));
        PyModule_AddObject(m, "AULT", PyFloat_FromDouble(ERFA_AULT));
        PyModule_AddObject(m, "DC", PyFloat_FromDouble(ERFA_DC));
        PyModule_AddObject(m, "ELG", PyFloat_FromDouble(ERFA_ELG));
        PyModule_AddObject(m, "ELB", PyFloat_FromDouble(ERFA_ELB));
        PyModule_AddObject(m, "TDB0", PyFloat_FromDouble(ERFA_TDB0));
        PyModule_AddObject(m, "SRS", PyFloat_FromDouble(ERFA_SRS));
        PyModule_AddObject(m, "WGS84", PyLong_FromLong(ERFA_WGS84));
        PyModule_AddObject(m, "GRS80", PyLong_FromLong(ERFA_GRS80));
        PyModule_AddObject(m, "WGS72", PyLong_FromLong(ERFA_WGS72));

        if (!initialized) {
            PyStructSequence_InitType(&AstromType, &ASTROM_type_desc);
            PyStructSequence_InitType(&LdbodyType, &LDBODY_type_desc);

        }
        Py_INCREF(&AstromType);
        PyModule_AddObject(m, "ASTROM", (PyObject*) &AstromType);
        Py_INCREF(&LdbodyType);
        PyModule_AddObject(m, "LDBODY", (PyObject*) &LdbodyType);

        initialized = 1;

#if PY_VERSION_HEX >= 0x03000000
        return m;
#else
        finally:
        return;
#endif
}
