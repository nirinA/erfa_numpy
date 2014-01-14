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
_to_py_scalar(double v)
{
    double *cv;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp dims[] = {1};
    pyout = (PyArrayObject *) PyArray_Zeros(1, dims, dsc, 0);
    if (NULL == pyout)  return NULL;
    cv = (double *)PyArray_DATA(pyout);
    cv[0] = v;
    Py_INCREF(pyout); 
    return PyArray_Return(pyout);                     
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

static PyObject *
_a2xf_object(char sign, int idmsf[4])
{
    int *a;
    PyArrayObject *pyout;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_INT);
    npy_intp dims[] = {4};
    int j;
    pyout=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                                                 dsc,
                                                 1,
                                                 dims,
                                                 NULL,
                                                 NULL,
                                                 0,
                                                 NULL);
    if (NULL == pyout) return NULL;
    a = (int *)PyArray_DATA(pyout);
    for(j=0;j<4;j++) a[j] = idmsf[j];
    Py_INCREF(pyout);                     
#if PY_VERSION_HEX >= 0x03000000
    return Py_BuildValue("CO", sign, pyout);
#else
    return Py_BuildValue("cO", sign, pyout);
#endif
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
    apnat = PyArray_FROM_OTF(pypnat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    av = PyArray_FROM_OTF(pyv, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    abm1 = PyArray_FROM_OTF(pybm1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (apnat == NULL || av == NULL || as == NULL || abm1 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(apnat);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(apnat);
    if (dims[0] != PyArray_DIMS(as)[0] ||
        dims[0] != PyArray_DIMS(abm1)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) {
        Py_DECREF(pyout);
        goto fail;
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
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(apnat);
    Py_XDECREF(av);
    Py_XDECREF(as);
    Py_XDECREF(abm1);
    return NULL;
}

PyDoc_STRVAR(_erfa_ab_doc,
"\nab(pnat[3], v[3], s, bm1) -> ppr[3]\n"
"Apply aberration to transform natural direction into proper direction.\n"
"Given:\n"
"    pnat       natural direction to the source (unit vector)\n"
"    v          observer barycentric velocity in units of c\n"
"    s          distance between the Sun and the observer (au)\n"
"    bm1        sqrt(1-|v|^2): reciprocal of Lorenz factor\n"
"Returned:\n"
"    ppr        proper direction to source (unit vector)");

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
    npy_intp *dims;
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2,
                                 &PyArray_Type, &pypv,
                                 &PyArray_Type, &pyebpv,
                                 &PyArray_Type, &pyehp))      
        return NULL;
    adate1 = PyArray_FROM_OTF(pydate1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adate2 = PyArray_FROM_OTF(pydate2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (adate1 == NULL || adate2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(adate1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(adate1);
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
            if (h == -1 && PyErr_Occurred()) goto fail;
            ehp[l] = h;
            Py_DECREF(aehp);
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
                if (p == -1 && PyErr_Occurred()) goto fail;
                pv[j][k] = p;
                e = (double)PyFloat_AsDouble(aebpv);
                if (e == -1 && PyErr_Occurred()) goto fail;
                ebpv[j][k] = e;
                Py_DECREF(apv);Py_DECREF(aebpv);
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
_erfa_ld(PyObject *self, PyObject *args)
{
    double *bm, *p, *q, *e, *em, *dlim, *p1;
    PyObject *pybm, *pyp, *pyq, *pye, *pyem, *pydlim;
    PyObject *abm, *ap, *aq, *ae, *aem, *adlim;
    PyArrayObject *pyp1;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pybm, &PyArray_Type, &pyp,
                                 &PyArray_Type, &pyq, &PyArray_Type, &pye,
                                 &PyArray_Type, &pyem, &PyArray_Type, &pydlim))
        return NULL;
    abm = PyArray_FROM_OTF(pybm, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ap = PyArray_FROM_OTF(pyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aq = PyArray_FROM_OTF(pyq, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ae = PyArray_FROM_OTF(pye, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aem = PyArray_FROM_OTF(pyem, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adlim = PyArray_FROM_OTF(pydlim, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (abm == NULL || ap == NULL || aq == NULL ||
        ae == NULL || aem == NULL || adlim == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ap);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ap);
    if (dims[0] != PyArray_DIMS(ap)[0] ||
        dims[0] != PyArray_DIMS(aq)[0] || dims[0] != PyArray_DIMS(ae)[0] ||
        dims[0] != PyArray_DIMS(aem)[0] || dims[0] != PyArray_DIMS(adlim)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyp1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyp1) {
        Py_DECREF(pyp1);
        goto fail;
    }
    bm = (double *)PyArray_DATA(abm);
    p = (double *)PyArray_DATA(ap);
    q = (double *)PyArray_DATA(aq);
    e = (double *)PyArray_DATA(ae);
    em = (double *)PyArray_DATA(aem);
    dlim = (double *)PyArray_DATA(adlim);
    p1 = (double *)PyArray_DATA(pyp1);
    for (i=0;i<dims[0];i++) {
        eraLd(bm[i], &p[i*3], &q[i*3], &e[i*3], em[i], dlim[i], &p1[i*3]);
    }
    Py_DECREF(abm);
    Py_DECREF(ap);
    Py_DECREF(aq);
    Py_DECREF(ae);
    Py_DECREF(aem);
    Py_DECREF(adlim);
    Py_INCREF(pyp1);
    return (PyObject *)pyp1;

fail:
    Py_XDECREF(abm);
    Py_XDECREF(ap);
    Py_XDECREF(aq);
    Py_XDECREF(ae);
    Py_XDECREF(aem);
    Py_XDECREF(adlim);
    return NULL;
}

PyDoc_STRVAR(_erfa_ld_doc,
"\nld(bm, p[3], q[3], e[3], em, dlim) -> p1[3]\n"
"Apply light deflection by a solar-system body, as part of\n"
"transforming coordinate direction into natural direction.\n"
"Given:\n"
"    bm     mass of the gravitating body (solar masses)\n"
"    p      direction from observer to source (unit vector)\n"
"    q      direction from body to source (unit vector)\n"
"    e      direction from body to observer (unit vector)\n"
"    em     distance from body to observer (au)\n"
"    dlim   deflection limiter\n"
"Returned:\n"
"    p1     observer to deflected source (unit vector)");

static PyObject *
_erfa_pmsafe(PyObject *self, PyObject *args)
{
    double *ra1, *dec1, *pmr1, *pmd1, *px1, *rv1, *ep1a, *ep1b, *ep2a, *ep2b;
    double *ra2, *dec2, *pmr2, *pmd2, *px2, *rv2;
    PyObject *pyra1, *pydec1, *pypmr1, *pypmd1, *pypx1, *pyrv1, *pyep1a, *pyep1b, *pyep2a, *pyep2b;
    PyObject *ara1, *adec1, *apmr1, *apmd1, *apx1, *arv1, *aep1a, *aep1b, *aep2a, *aep2b;
    PyArrayObject *pyra2 = NULL, *pydec2 = NULL, *pypmr2 = NULL;
    PyArrayObject *pypmd2 = NULL, *pypx2 = NULL, *pyrv2 = NULL;
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
    ara1 = PyArray_FROM_OTF(pyra1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adec1 = PyArray_FROM_OTF(pydec1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apmr1 = PyArray_FROM_OTF(pypmr1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apmd1 = PyArray_FROM_OTF(pypmd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apx1 = PyArray_FROM_OTF(pypx1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arv1 = PyArray_FROM_OTF(pyrv1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep1a = PyArray_FROM_OTF(pyep1a, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep1b = PyArray_FROM_OTF(pyep1b, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep2a = PyArray_FROM_OTF(pyep2a, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep2b = PyArray_FROM_OTF(pyep2b, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ara1 == NULL || adec1 == NULL || apmr1 == NULL || apmd1 == NULL || apx1 == NULL ||
        arv1 == NULL || aep1a == NULL || aep1b == NULL || aep2a == NULL || aep2b == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ara1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ara1);
    if (dims[0] != PyArray_DIMS(adec1)[0] ||
        dims[0] != PyArray_DIMS(apmr1)[0] || dims[0] != PyArray_DIMS(apmd1)[0] ||
        dims[0] != PyArray_DIMS(apx1)[0] || dims[0] != PyArray_DIMS(arv1)[0] ||
        dims[0] != PyArray_DIMS(aep1a)[0] || dims[0] != PyArray_DIMS(aep1b)[0] ||
        dims[0] != PyArray_DIMS(aep2a)[0] || dims[0] != PyArray_DIMS(aep2b)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyra2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydec2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypmr2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypmd2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypx2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyrv2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyra2 || NULL == pydec2 || NULL == pypmr2 ||
        NULL == pypmd2 || NULL == pypx2 || NULL == pyrv2)  goto fail;

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
            goto fail;
        }
        else if (j == 1) {
            PyErr_SetString(_erfaError, "distance overridden");
            goto fail;
        }
        else if (j == 2) {
            PyErr_SetString(_erfaError, "excessive velocity");
            goto fail;
        }
        else if (j == 4) {
            PyErr_SetString(_erfaError, "solution didn't converge");
            goto fail;
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
    Py_INCREF(pyra2);
    Py_INCREF(pydec2);
    Py_INCREF(pypmr2);
    Py_INCREF(pypmd2);
    Py_INCREF(pypx2);
    Py_INCREF(pyrv2);
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
    Py_XDECREF(pyra2);
    Py_XDECREF(pydec2);
    Py_XDECREF(pypmr2);
    Py_XDECREF(pypmd2);
    Py_XDECREF(pypx2);
    Py_XDECREF(pyrv2);
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
_erfa_c2ixys(PyObject *self, PyObject *args)
{
    double *x, *y, *s, rc2i[3][3];
    PyObject *pyx, *pyy, *pys;
    PyObject *ax, *ay, *as;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyx,
                                 &PyArray_Type, &pyy,
                                 &PyArray_Type, &pys))
        return NULL;

    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (as == NULL || ax == NULL || ay == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ax);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ax);
    if (dims[0] != PyArray_DIMS(as)[0] || dims[0] != PyArray_DIMS(ay)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    s = (double *)PyArray_DATA(as);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraC2ixys(x[i], y[i], s[i], rc2i);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(as);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(as);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2ixys_doc,
"\nc2ixys(x, y, s) -> rc2i\n\n"
"Form the celestial to intermediate-frame-of-date matrix\n"
" given the CIP X,Y and the CIO locator s.\n"
"Given:\n"
"    x, y       Celestial Intermediate Pole\n"
"    s          CIO locator \n"
"Returned:\n"
"   rc2i        celestial-to-intermediate matrix");

static PyObject *
_erfa_cal2jd(PyObject *self, PyObject *args)
{
    int *iy, *im, *id;
    double *dmj0, *dmj1;
    int status;
    PyObject *pyiy, *pyim, *pyid;
    PyObject *aiy, *aim, *aid;
    PyArrayObject *pydmj0 = NULL, *pydmj1 = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                                 &PyArray_Type, &pyiy,
                                 &PyArray_Type, &pyim,
                                 &PyArray_Type, &pyid))
        return NULL;
    aiy = PyArray_FROM_OTF(pyiy, NPY_INT, NPY_ARRAY_IN_ARRAY);
    aim = PyArray_FROM_OTF(pyim, NPY_INT, NPY_ARRAY_IN_ARRAY);
    aid = PyArray_FROM_OTF(pyid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (NULL == aiy || NULL == aim || NULL == aid) goto fail;
    ndim = PyArray_NDIM(aiy);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aiy);
    pydmj0 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydmj1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pydmj0 || NULL == pydmj1) {
        goto fail;
    }
    iy = (int *)PyArray_DATA(aiy);
    im = (int *)PyArray_DATA(aim);
    id = (int *)PyArray_DATA(aid);
    dmj0 = (double *)PyArray_DATA(pydmj0);
    dmj1 = (double *)PyArray_DATA(pydmj1);
    for (i=0;i<dims[0];i++) {
        status = eraCal2jd(iy[i], im[i], id[i], &dmj0[i], &dmj1[i]);
        if (status < 0){
            if (status == -1){
                PyErr_SetString(_erfaError, "bad year");
                goto fail;
            }
            else if (status == -2){
                PyErr_SetString(_erfaError, "bad month");
                goto fail;
            }
            else if (status == -3){
                PyErr_SetString(_erfaError, "bad day");
                goto fail;
            }
        }
    }
    Py_DECREF(aiy);
    Py_DECREF(aim);
    Py_DECREF(aid);
    Py_INCREF(pydmj0);
    Py_INCREF(pydmj1);
    return Py_BuildValue("OO", pydmj0, pydmj1);

fail:
    Py_XDECREF(aiy);
    Py_XDECREF(aim);
    Py_XDECREF(aid);
    Py_XDECREF(pydmj0);
    Py_XDECREF(pydmj1);
    return NULL;
}

PyDoc_STRVAR(_erfa_cal2jd_doc,
"\ncal2jd(year, month, day) -> 2400000.5, djm\n\n"
"Gregorian Calendar to Julian Date.\n"
"Given:\n"
"    year, month      day in Gregorian calendar\n"
"Returned:\n"
"    2400000.5,djm    MJD zero-point and Modified Julian Date for 0 hrs");

static PyObject *
_erfa_d2dtf(PyObject *self, PyObject *args)
{
    double *d1, *d2;
    char *scale = "UTC";
    int n, status;
    int iy, im, id, ihmsf[4];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyiy = NULL, *pyim = NULL, *pyid = NULL, *pyihmsf = NULL;
    PyObject *iy_iter = NULL, *im_iter = NULL, *id_iter = NULL, *ihmsf_iter = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_LONG);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "siO!O!",
                                 &scale, &n,
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2))
        return NULL;
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyiy = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyim = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyid = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    dim_out[0] = dims[0];
    dim_out[1] = 4;
    pyihmsf = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyiy || NULL == pyim || NULL == pyid || NULL == pyihmsf) goto fail;
    iy_iter = PyArray_IterNew((PyObject*)pyiy);
    im_iter = PyArray_IterNew((PyObject*)pyim);
    id_iter = PyArray_IterNew((PyObject*)pyid);
    ihmsf_iter = PyArray_IterNew((PyObject*)pyihmsf);
    if (iy_iter == NULL || im_iter == NULL || id_iter == NULL || ihmsf_iter == NULL) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    
    for (i=0;i<dims[0];i++) {
        status = eraD2dtf(scale, n, d1[i], d2[i], &iy, &im, &id, ihmsf);
        if (status > 0){
            PyErr_SetString(_erfaError, "doubious year: date predates UTC or too far in the future");
            goto fail;
        }
        if (status < 0){
            PyErr_SetString(_erfaError, "unaceptable date");
            goto fail;
        }
        int k;
        if (PyArray_SETITEM(pyiy, PyArray_ITER_DATA(iy_iter), PyLong_FromLong((long)iy))) {
            PyErr_SetString(_erfaError, "unable to set iy");
            goto fail;
        }
        PyArray_ITER_NEXT(iy_iter);
        if (PyArray_SETITEM(pyim, PyArray_ITER_DATA(im_iter), PyLong_FromLong((long)im))) {
            PyErr_SetString(_erfaError, "unable to set im");
            goto fail;
        }
        PyArray_ITER_NEXT(im_iter);
        if (PyArray_SETITEM(pyid, PyArray_ITER_DATA(id_iter), PyLong_FromLong((long)id))) {
            PyErr_SetString(_erfaError, "unable to set id");
            goto fail;
        }
        PyArray_ITER_NEXT(id_iter);
        for (k=0;k<4;k++) {
            if (PyArray_SETITEM(pyihmsf, PyArray_ITER_DATA(ihmsf_iter), PyLong_FromLong((long)ihmsf[k]))) {
                PyErr_SetString(_erfaError, "unable to set ihmsf");
                goto fail;
            }
            PyArray_ITER_NEXT(ihmsf_iter);
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(iy_iter); Py_DECREF(im_iter); Py_DECREF(id_iter); Py_DECREF(ihmsf_iter);
    Py_INCREF(pyiy); Py_INCREF(pyim); Py_INCREF(pyid); Py_INCREF(pyihmsf);
    return Py_BuildValue("OOOO", pyiy, pyim, pyid, pyihmsf);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(iy_iter); Py_XDECREF(im_iter); Py_XDECREF(id_iter); Py_XDECREF(ihmsf_iter);
    Py_XDECREF(pyiy); Py_XDECREF(pyim); Py_XDECREF(pyid); Py_XDECREF(pyihmsf);
    return NULL;
}

PyDoc_STRVAR(_erfa_d2dtf_doc,
"\nd2dtf(scale, n, d1, d2) -> year, month, day, hours, minutes, seconds, fraction\n\n"
"Format for output a 2-part Julian Date (or in the case of UTC a\n"
"quasi-JD form that includes special provision for leap seconds).\n"
"Given:\n"
"    scale      time scale ID, default to ''UTC''\n"
"    ndp        resolution\n"
"    d1,d2      time as a 2-part Julian Date\n"
"Returned:\n"
"   iy,im,id    year, month, day in Gregorian calendar\n"
"   ihmsf       hours, minutes, seconds, fraction\n");

static PyObject *
_erfa_dat(PyObject *self, PyObject *args)
{
    int *iy, *im, *id, status;
    double *fd, *deltat;
    PyObject *pyiy, *pyim, *pyid, *pyfd;
    PyObject *aiy, *aim, *aid, *afd;
    PyArrayObject *pydeltat = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pyiy,
                                 &PyArray_Type, &pyim,
                                 &PyArray_Type, &pyid,
                                 &PyArray_Type, &pyfd))
        return NULL;
    aiy = PyArray_FROM_OTF(pyiy, NPY_INT, NPY_ARRAY_IN_ARRAY);
    aim = PyArray_FROM_OTF(pyim, NPY_INT, NPY_ARRAY_IN_ARRAY);
    aid = PyArray_FROM_OTF(pyid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    afd = PyArray_FROM_OTF(pyfd, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (NULL == aiy || NULL == aim || NULL == aid || NULL == afd) goto fail;
    ndim = PyArray_NDIM(aiy);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aiy);
    pydeltat = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pydeltat) {
        Py_DECREF(pydeltat);
        goto fail;
    }
    iy = (int *)PyArray_DATA(aiy);
    im = (int *)PyArray_DATA(aim);
    id = (int *)PyArray_DATA(aid);
    fd = (double *)PyArray_DATA(afd);
    deltat = (double *)PyArray_DATA(pydeltat);

    for (i=0;i<dims[0];i++) {
        status = eraDat(iy[i], im[i], id[i], fd[i], &deltat[i]);
        if (status > 0){
            PyErr_SetString(_erfaError, "doubious year: date before UTC:1960 January 1.0.");
            goto fail;
        }
        else if (status < 0){
            if (status == -1){
                PyErr_SetString(_erfaError, "unaceptable date, bad year");
                goto fail;
            }
            else if (status == -2){
                PyErr_SetString(_erfaError, "unaceptable date, bad month");
                goto fail;
            }
            else if (status == -3){
                PyErr_SetString(_erfaError, "unaceptable date, bad day");
                goto fail;
            }      
            else if (status == -4){
                PyErr_SetString(_erfaError, "bad fraction day, should be < 1.");
                goto fail;
            }      
        }
    }
    Py_DECREF(aiy);
    Py_DECREF(aim);
    Py_DECREF(aid);
    Py_DECREF(afd);
    Py_INCREF(pydeltat);
    return (PyObject *)pydeltat;

fail:
    Py_XDECREF(aiy);
    Py_XDECREF(aim);
    Py_XDECREF(aid);
    Py_XDECREF(afd);
    Py_XDECREF(pydeltat);
    return NULL;
}

PyDoc_STRVAR(_erfa_dat_doc,
"\ndat(y,m,d,fd) -> deltat\n\n"
"For a given UTC date, calculate delta(AT) = TAI-UTC.\n"
"Given:\n"
"    y          year\n"
"    m          month\n"
"    d          day\n"
"    fd         fraction of day\n"
"Returned:\n"
"    deltat     TAI minus UTC, seconds");

static PyObject *
_erfa_dtf2d(PyObject *self, PyObject *args)
{
    int *iy, *im, *id, *ihr, *imn, status;
    double *sec, *d1, *d2;
    char *scale;
    PyObject *pyiy, *pyim, *pyid, *pyihr, *pyimn, *pysec;
    PyObject *aiy, *aim, *aid, *aihr, *aimn, *asec;
    PyArrayObject *pyd1 = NULL, *pyd2 = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!",
                                 &scale,
                                 &PyArray_Type, &pyiy,
                                 &PyArray_Type, &pyim,
                                 &PyArray_Type, &pyid,
                                 &PyArray_Type, &pyihr,
                                 &PyArray_Type, &pyimn,
                                 &PyArray_Type, &pysec))
        return NULL;
    aiy = PyArray_FROM_OTF(pyiy, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    aim = PyArray_FROM_OTF(pyim, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    aid = PyArray_FROM_OTF(pyid, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    aihr = PyArray_FROM_OTF(pyihr, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    aimn = PyArray_FROM_OTF(pyimn, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    asec = PyArray_FROM_OTF(pysec, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (NULL == aiy || NULL == aim || NULL == aid ||
        NULL == aihr || NULL == aim || NULL == asec) goto fail;
    ndim = PyArray_NDIM(aiy);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aiy);
    pyd1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyd2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyd1 || NULL == pyd2) {
        goto fail;
    }
    iy = (int *)PyArray_DATA(aiy);
    im = (int *)PyArray_DATA(aim);
    id = (int *)PyArray_DATA(aid);
    ihr = (int *)PyArray_DATA(aihr);
    imn = (int *)PyArray_DATA(aimn);
    sec = (double *)PyArray_DATA(asec);
    d1 = (double *)PyArray_DATA(pyd1);
    d2 = (double *)PyArray_DATA(pyd2);

    for (i=0;i<dims[0];i++) {
        status = eraDtf2d(scale, iy[i], im[i], id[i], ihr[i], imn[i], sec[i], &d1[i], &d2[i]);
        if (status == 3) {
            PyErr_SetString(_erfaError, "both of next two");
            goto fail;
        }
        else if (status == 2) {
            PyErr_SetString(_erfaError, "time is after end of day");
            goto fail;
        }
        else if (status == 1) {
            PyErr_SetString(_erfaError, "dubious year");
            goto fail;
        }
        else if (status == -1) {
            PyErr_SetString(_erfaError, "bad year");
            goto fail;
        }
        else if (status == -2) {
            PyErr_SetString(_erfaError, "bad month");
            goto fail;
        }
        else if (status == -3) {
            PyErr_SetString(_erfaError, "bad day");
            goto fail;
        }
        else if (status == -4) {
            PyErr_SetString(_erfaError, "bad hour");
            goto fail;
        }
        else if (status == -5) {
            PyErr_SetString(_erfaError, "bad minute");
            goto fail;
        }
        else if (status == -6) {
            PyErr_SetString(_erfaError, "bad second (<0)");
            goto fail;
        }
    }
    Py_DECREF(aiy);
    Py_DECREF(aim);
    Py_DECREF(aid);
    Py_DECREF(aihr);
    Py_DECREF(aimn);
    Py_DECREF(asec);
    Py_INCREF(pyd1);
    Py_INCREF(pyd2);
    return Py_BuildValue("OO", pyd1, pyd2);

fail:
    Py_XDECREF(aiy);
    Py_XDECREF(aim);
    Py_XDECREF(aid);
    Py_XDECREF(aihr);
    Py_XDECREF(aimn);
    Py_XDECREF(asec);
    Py_XDECREF(pyd1);
    Py_XDECREF(pyd2);
    return NULL;
}

PyDoc_STRVAR(_erfa_dtf2d_doc,
"\ndtf2d(scale, iy, im, id, ihr, imn, sec) -> d1, d2\n\n"
"Encode date and time fields into 2-part Julian Date (or in the case\n"
"of UTC a quasi-JD form that includes special provision for leap seconds).\n"
"Given:\n"
"    y,m,d   year, month, day in Gregorian calendar\n"
"    hr,mn   hour, minute\n"
"    sec     seconds\n"
"    scale   optional time scale ID, default ''UTC''\n"
"Returned:\n"
"   d1,d2     2-part Julian Date");

static PyObject *
_erfa_ee00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *epsa, *dpsi, *ee;
    PyObject *pyd1, *pyd2, *pyepsa, *pydpsi;
    PyObject *ad1, *ad2, *aepsa, *adpsi;
    PyArrayObject *pyee = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pyepsa,
                                 &PyArray_Type, &pydpsi))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aepsa = PyArray_FROM_OTF(pyepsa, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adpsi = PyArray_FROM_OTF(pydpsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || aepsa == NULL || adpsi == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0] ||
        dims[0] != PyArray_DIMS(aepsa)[0] ||
        dims[0] != PyArray_DIMS(adpsi)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyee = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyee) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    epsa = (double *)PyArray_DATA(aepsa);
    dpsi = (double *)PyArray_DATA(adpsi);
    ee = (double *)PyArray_DATA(pyee);
    for (i=0;i<dims[0];i++) {
        ee[i] = eraEe00(d1[i], d2[i], epsa[i], dpsi[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(aepsa);
    Py_DECREF(adpsi);
    Py_INCREF(pyee);
    return (PyObject *)pyee;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(aepsa);
    Py_XDECREF(adpsi);
    Py_XDECREF(pyee);
    return NULL;
}

PyDoc_STRVAR(_erfa_ee00_doc,
"\nee00(d1,d2,epsa,dpsi) -> ee\n\n"
"The equation of the equinoxes compatible with IAU 2000 resolutions,\n"
"given the nutation in longitude and the mean obliquity.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"    epsa       mean obliquity\n"
"    dpsi       nutation in longitude\n"
"Returned:\n"
"    ee         equation of the equinoxes");

static PyObject *
_erfa_epb(PyObject *self, PyObject *args)
{
    double *d1, *d2, *b;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyb = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyb = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyb) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    b = (double *)PyArray_DATA(pyb);
    for (i=0;i<dims[0];i++) {
        b[i] = eraEpb(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyb);
    return (PyObject *)pyb;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyb);
    return NULL;
}

PyDoc_STRVAR(_erfa_epb_doc,
"\nepb(d1, d2) -> b\n\n"
"Julian Date to Besselian Epoch\n"
"Given:\n"
"    d1,d2      2-part Julian Date\n"
"Returned:\n"
"    b          Besselian Epoch.");

static PyObject *
_erfa_epb2jd(PyObject *self, PyObject *args)
{
    double *epb, *jd0, *jd1;
    PyObject *pyepb;
    PyObject *aepb;
    PyArrayObject *pyjd0 = NULL, *pyjd1 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyepb))
        return NULL;
    aepb = PyArray_FROM_OTF(pyepb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aepb == NULL) goto fail;
    ndim = PyArray_NDIM(aepb);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }            
    dims = PyArray_DIMS(aepb);
    pyjd0 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyjd1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyjd0 || NULL == pyjd1)  goto fail;
    epb = (double *)PyArray_DATA(aepb);
    jd0 = (double *)PyArray_DATA(pyjd0);
    jd1 = (double *)PyArray_DATA(pyjd1);
    for (i=0;i<dims[0];i++) {
        eraEpb2jd(epb[i], &jd0[i], &jd1[i]);
    }
    Py_DECREF(aepb);
    Py_INCREF(pyjd0);
    Py_INCREF(pyjd1);
    return Py_BuildValue("OO", pyjd0, pyjd1);

fail:
    Py_XDECREF(aepb);
    Py_XDECREF(pyjd0);
    Py_XDECREF(pyjd1);
    return NULL;
}

PyDoc_STRVAR(_erfa_epb2jd_doc,
"\nepb2jd(epb) -> 2400000.5, djm\n"
"Given:\n"
"    epb        Besselian Epoch,\n"
"Returned:\n"
"    djm        Modified Julian Date");

static PyObject *
_erfa_epj(PyObject *self, PyObject *args)
{
    double *d1, *d2, *j;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyj = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyj = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyj) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    j = (double *)PyArray_DATA(pyj);
    for (i=0;i<dims[0];i++) {
        j[i] = eraEpj(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyj);
    return (PyObject *)pyj;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyj);
    return NULL;
}

PyDoc_STRVAR(_erfa_epj_doc,
"\nepj(d1, d2) -> b\n\n"
"Julian Date to Julian Epoch.\n"
"Given:\n"
"    d1,d2      2-part Julian Date\n"
"Returned:\n"
"    b          Julian Epoch");

static PyObject *
_erfa_epj2jd(PyObject *self, PyObject *args)
{
    double *epj, *jd0, *jd1;
    PyObject *pyepj;
    PyObject *aepj;
    PyArrayObject *pyjd0 = NULL, *pyjd1 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyepj))
        return NULL;
    aepj = PyArray_FROM_OTF(pyepj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aepj == NULL) goto fail;
    ndim = PyArray_NDIM(aepj);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }            
    dims = PyArray_DIMS(aepj);
    pyjd0 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyjd1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyjd0 || NULL == pyjd1)  goto fail;
    epj = (double *)PyArray_DATA(aepj);
    jd0 = (double *)PyArray_DATA(pyjd0);
    jd1 = (double *)PyArray_DATA(pyjd1);
    for (i=0;i<dims[0];i++) {
        eraEpj2jd(epj[i], &jd0[i], &jd1[i]);
    }
    Py_DECREF(aepj);
    Py_INCREF(pyjd0);
    Py_INCREF(pyjd1);
    return Py_BuildValue("OO", pyjd0, pyjd1);

fail:
    Py_XDECREF(aepj);
    Py_XDECREF(pyjd0);
    Py_XDECREF(pyjd1);
    return NULL;
}

PyDoc_STRVAR(_erfa_epj2jd_doc,
"\nepj2jd(epj) -> 2400000.5, djm\n\n"
"Julian Epoch to Julian Date\n"
"Given:\n"
"    epj        Julian Epoch\n"
"Returned:\n"
"    djm        Modified Julian Date");

static PyObject *
_erfa_eqeq94(PyObject *self, PyObject *args)
{
    double *d1, *d2, *ee;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyee = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyee = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyee) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ee = (double *)PyArray_DATA(pyee);
    for (i=0;i<dims[0];i++) {
        ee[i] = eraEqeq94(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyee);
    return (PyObject *)pyee;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyee);
    return NULL;
}

PyDoc_STRVAR(_erfa_eqeq94_doc,
"\neqeq94(d1,d2) -> ee\n\n"
"Equation of the equinoxes, IAU 1994 model.\n"
"Given:\n"
"    d1,d2      TDB as 2-part Julian Date\n"
"Returned:\n"
"    ee         equation of the equinoxes");

static PyObject *
_erfa_era00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *era;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    era = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        era[i] = eraEra00(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyout);
    return NULL;    
}

PyDoc_STRVAR(_erfa_era00_doc,
"\nera00(d1,d2) -> era\n\n"
"Earth rotation angle (IAU 2000 model).\n"
"Given:\n"
"    d1,d2      UT1 as 2-part Julian Date (d1,d2)\n"
"Returned:\n"
"    era         Earth rotation angle (radians), range 0-2pi");

static PyObject *
_erfa_gmst00(PyObject *self, PyObject *args)
{
    double *uta, *utb, *tta, *ttb, *g;
    PyObject *pyuta, *pyutb, *pytta, *pyttb;
    PyObject *auta, *autb, *atta, *attb;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb))
        return NULL;

    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL || atta == NULL || attb == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(auta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(auta);
    if (dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(atta)[0] ||
        dims[0] != PyArray_DIMS(attb)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyout = (PyArrayObject *) PyArray_Zeros(1, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    g = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        g[i] = eraGmst00(uta[i], utb[i], tta[i], ttb[i]);
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_gmst00_doc,
"\ngmst00(uta, utb, tta, ttb) -> gmst\n\n"
"Greenwich mean sidereal time\n"
"(model consistent with IAU 2000 resolutions).\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"    tta,ttb    TT as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich mean sidereal time (radians)");

static PyObject *
_erfa_gmst82(PyObject *self, PyObject *args)
{
    double *d1, *d2, *g;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    g = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        g[i] = eraGmst82(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_gmst82_doc,
"\ngmst82(d1, d2) -> gmst\n\n"
"Universal Time to Greenwich mean sidereal time (IAU 1982 model)\n"
"Given:\n"
"    d1,d2      UT1 as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich mean sidereal time (radians)");

static PyObject *
_erfa_numat(PyObject *self, PyObject *args)
{
    double *epsa, *dpsi, *deps, rmatn[3][3];
    PyObject *pyepsa, *pydpsi, *pydeps;
    PyObject *aepsa, *adpsi, *adeps;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                                 &PyArray_Type, &pyepsa,
                                 &PyArray_Type, &pydpsi,
                                 &PyArray_Type, &pydeps)) {
        return NULL;
    }
    aepsa = PyArray_FROM_OTF(pyepsa, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adpsi = PyArray_FROM_OTF(pydpsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adeps = PyArray_FROM_OTF(pydeps, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aepsa == NULL || adpsi == NULL || adeps == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aepsa);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aepsa);
    if (dims[0] != PyArray_DIMS(adpsi)[0] || dims[0] != PyArray_DIMS(adeps)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    epsa = (double *)PyArray_DATA(aepsa);
    dpsi = (double *)PyArray_DATA(adpsi);
    deps = (double *)PyArray_DATA(adeps);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *)PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout)  goto fail;
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraNumat(epsa[i], dpsi[i], deps[i], rmatn);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rmatn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(aepsa);
    Py_DECREF(adpsi);
    Py_DECREF(adeps);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(aepsa);
    Py_XDECREF(adpsi);
    Py_XDECREF(adeps);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_numat_doc,
"\nnumat(epsa, dpsi, deps) -> rmatn\n\n"
"Form the matrix of nutation.\n"
"Given:\n"
"     epsa          mean obliquity of date\n"
"     dpsi,deps     nutation\n"
"Returned:\n"
"     rmatn         nutation matrix");

static PyObject *
_erfa_nut00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, dpsi, deps;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dim_out[0] = 2;
    dim_out[1] = dims[0];
    pyout = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;
    for (i=0;i<dims[0];i++) {
        eraNut00a(d1[i], d2[i], &dpsi, &deps);
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(dpsi))) {
            PyErr_SetString(_erfaError, "unable to set dpsi");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(deps))) {
            PyErr_SetString(_erfaError, "unable to set deps");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_nut00a_doc,
"\nnut00a(d1, d2) -> dpsi, deps\n\n"
"Nutation, IAU 2000A model (MHB2000 luni-solar and planetary nutation\n"
"with free core nutation omitted).\n"
"Given:\n"
"    d1,d2      TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsi,deps   nutation, luni-solar + planetary\n");

static PyObject *
_erfa_nut80(PyObject *self, PyObject *args)
{
    double *d1, *d2, dpsi, deps;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dim_out[0] = 2;
    dim_out[1] = dims[0];
    pyout = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;
    for (i=0;i<dims[0];i++) {
        eraNut80(d1[i], d2[i], &dpsi, &deps);
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(dpsi))) {
            PyErr_SetString(_erfaError, "unable to set dpsi");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(deps))) {
            PyErr_SetString(_erfaError, "unable to set deps");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_nut80_doc,
"\nnut80(d1, d2) -> dpsi, deps\n\n"
"Nutation, IAU 1980 model.\n"
"Given:\n"
"    d1,d2      TT as a 2-part Julian Date\n"
"Returned:\n"
"    dpsi        nutation in longitude (radians)\n"
"    deps        nutation in obliquity (radians)\n");

static PyObject *
_erfa_obl80(PyObject *self, PyObject *args)
{
    double *d1, *d2, obl;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dim_out[0] = dims[0];
    pyout = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;
    for (i=0;i<dims[0];i++) {
        obl = eraObl80(d1[i], d2[i]);
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(obl))) {
            PyErr_SetString(_erfaError, "unable to set obl");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;
    
fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_obl80_doc,
"\nobl80(d1, d2) -> obl\n\n"
"Mean obliquity of the ecliptic, IAU 1980 model.\n"
"Given:\n"
"    d1,d2      TT as a 2-part Julian Date\n"
"Returned:\n"
"    obl        obliquity of the ecliptic (radians)");

static PyObject *
_erfa_plan94(PyObject *self, PyObject *args)
{
    double *d1, *d2, pv[2][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    int np, status;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!i",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &np)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dim_out[0] = dims[0];
    dim_out[1] = 2;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) {
        goto fail;
    }
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        status = eraPlan94(d1[i], d2[i], np, pv);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal np,  not in range(1,8) for planet");
            goto fail;
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
        int j,k;
        for (j=0;j<2;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(pv[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set pv");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
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
_erfa_pmat76(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatp[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2))
        return NULL;
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (NULL == ad1 || NULL == ad2) goto fail;
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                                                   dsc,
                                                   3,
                                                   dim_out,
                                                   NULL,
                                                   NULL,
                                                   0,
                                                   NULL);
    //pyout = (PyArrayObject *)PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout)  goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);

    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraPmat76(d1[i], d2[i], rmatp);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rmatp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_pmat76_doc,
"\npmat76(d1, d2) -> rmatp\n\n"
"Precession matrix from J2000.0 to a specified date, IAU 1976 model.\n"
"Given:\n"
"   d1,d2       TT ending date as a 2-part Julian Date\n"
"Returned:\n"
"   rmatp       precession matrix, J2000.0 -> d1+d2");

static PyObject *
_erfa_pn00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *dpsi, *deps;
    double epsa, rb[3][3],rp[3][3],rbp[3][3],rn[3][3],rbpn[3][3];
    PyObject *pyd1, *pyd2, *pydpsi, *pydeps;
    PyObject *ad1, *ad2, *adpsi, *adeps;
    PyArrayObject *pyepsa = NULL, *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL, *pyrn = NULL, *pyrbpn = NULL;
    PyObject *epsa_iter = NULL, *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL, *rn_iter = NULL, *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3], dim_epsa[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pydpsi,
                                 &PyArray_Type, &pydeps)) {
        return NULL;
    }
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adpsi = PyArray_FROM_OTF(pydpsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adeps = PyArray_FROM_OTF(pydeps, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || adpsi == NULL || adeps == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dpsi = (double *)PyArray_DATA(adpsi);
    deps = (double *)PyArray_DATA(adeps);
    dim_epsa[0] = dims[0];
    pyepsa = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    if (NULL == pyepsa) {
        goto fail;
    }
    epsa_iter = PyArray_IterNew((PyObject*)pyepsa);
    if (epsa_iter == NULL) goto fail;
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrb = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrbp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrbpn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrb || NULL == pyrp ||
        NULL == pyrbp || NULL == pyrn ||
        NULL == pyrbpn) {
        goto fail;
    }
    rb_iter = PyArray_IterNew((PyObject*)pyrb);
    if (rb_iter == NULL) goto fail;
    rp_iter = PyArray_IterNew((PyObject*)pyrp);
    if (rp_iter == NULL) goto fail;
    rbp_iter = PyArray_IterNew((PyObject*)pyrbp);
    if (rbp_iter == NULL) goto fail;
    rn_iter = PyArray_IterNew((PyObject*)pyrn);
    if (rn_iter == NULL) goto fail;
    rbpn_iter = PyArray_IterNew((PyObject*)pyrbpn);
    if (rbpn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraPn00(d1[i], d2[i], dpsi[i], deps[i],
                &epsa, rb, rp, rbp, rn, rbpn);
        if (PyArray_SETITEM(pyepsa, PyArray_ITER_DATA(epsa_iter), PyFloat_FromDouble(epsa))) {
            PyErr_SetString(_erfaError, "unable to set epsa");
            goto fail;
        }
        PyArray_ITER_NEXT(epsa_iter);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrb, PyArray_ITER_DATA(rb_iter), PyFloat_FromDouble(rb[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rb");
                    goto fail;
                }
                PyArray_ITER_NEXT(rb_iter);
                if (PyArray_SETITEM(pyrp, PyArray_ITER_DATA(rp_iter), PyFloat_FromDouble(rp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rp");
                    goto fail;
                }
                PyArray_ITER_NEXT(rp_iter);
                if (PyArray_SETITEM(pyrbp, PyArray_ITER_DATA(rbp_iter), PyFloat_FromDouble(rbp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rbp");
                    goto fail;
                }
                PyArray_ITER_NEXT(rbp_iter);
                if (PyArray_SETITEM(pyrn, PyArray_ITER_DATA(rn_iter), PyFloat_FromDouble(rn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rn_iter);
                if (PyArray_SETITEM(pyrbpn, PyArray_ITER_DATA(rbpn_iter), PyFloat_FromDouble(rbpn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rbpn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rbpn_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(adpsi);
    Py_DECREF(adeps);
    Py_DECREF(epsa_iter);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_DECREF(rn_iter);
    Py_DECREF(rbpn_iter);
    return Py_BuildValue("OOOOOO", pyepsa, pyrb, pyrp, pyrbp, pyrn, pyrbpn);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(adpsi);
    Py_XDECREF(adeps);
    Py_XDECREF(epsa_iter);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    Py_XDECREF(rn_iter);
    Py_XDECREF(rbpn_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_pn00_doc,
"\npn00(d1,d2,dpsi,deps) -> epsa,rb,rp,rbp,rn,rbpn\n\n"
"Precession-nutation, IAU 2000 model:  a multi-purpose function,\n"
"supporting classical (equinox-based) use directly and CIO-based\n"
"use indirectly.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"   dpsi,deps   nutation\n"
"Returned:\n"
"   epsa        mean obliquity\n"
"   rb          frame bias matrix\n"
"   rp          precession matrix\n"
"   rbp         bias-precession matrix\n"
"   rn          nutation matrix\n"
"   rbpn        GCRS-to-true matrix");

static PyObject *
_erfa_pom00(PyObject *self, PyObject *args)
{    
    double *xp, *yp, *sp, rpom[3][3];
    PyObject *pyxp, *pyyp, *pysp;
    PyObject *axp, *ayp, *asp;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp,
                                 &PyArray_Type, &pysp))
        return NULL;

    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    asp = PyArray_FROM_OTF(pysp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (asp == NULL || axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(axp);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(axp);
    if (dims[0] != PyArray_DIMS(asp)[0] || dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    sp = (double *)PyArray_DATA(asp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraPom00(xp[i], yp[i], sp[i], rpom);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rpom[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(asp);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(asp);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_pom00_doc,
"\npom00(xp, yp, sp) -> rpom\n\n"
"Form the matrix of polar motion for a given date, IAU 2000.\n"
"Given:\n"
"   xp,yp       coordinates of the pole (radians)\n"
"   sp          the TIO locator s' (radians)\n"
"Returned:\n"
"   rpom        polar-motion matrix");

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

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || ax == NULL || ay == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0] ||
        dims[0] != PyArray_DIMS(ax)[0] || dims[0] != PyArray_DIMS(ay)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
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
    Py_INCREF(pyout);
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

static PyObject *
_erfa_s06(PyObject *self, PyObject *args)
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

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || ax == NULL || ay == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0] ||
        dims[0] != PyArray_DIMS(ax)[0] || dims[0] != PyArray_DIMS(ay)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    s = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        s[i] = eraS06(d1[i], d2[i], x[i], y[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_s06_doc,
"\ns06(d1, d2, x, y) -> s\n\n"
"The CIO locator s, positioning the Celestial Intermediate Origin on\n"
"the equator of the Celestial Intermediate Pole, given the CIP's X,Y\n"
"coordinates.  Compatible with IAU 2006/2000A precession-nutation.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"   x,y     CIP coordinates\n"
"Returned:\n"
"   s       the CIO locator s in radians");

static PyObject *
_erfa_sp00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    s = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        s[i] = eraSp00(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_sp00_doc,
"\nsp00(d1, d2) -> s\n\n"
"The TIO locator s', positioning the Terrestrial Intermediate Origin\n"
"on the equator of the Celestial Intermediate Pole.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   s       the TIO locator s' in radians");

static PyObject *
_erfa_taitt(PyObject *self, PyObject *args)
{
    double *tai1, *tai2, *tt1, *tt2;
    int status;
    PyObject *pytai1, *pytai2;
    PyObject *atai1, *atai2;
    PyArrayObject *pytt1 = NULL, *pytt2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pytai1,
                                 &PyArray_Type, &pytai2))
        return NULL;

    atai1 = PyArray_FROM_OTF(pytai1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atai2 = PyArray_FROM_OTF(pytai2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atai1 == NULL || atai2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atai1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atai1);
    if (dims[0] != PyArray_DIMS(atai2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytt1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytt2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytt1 || NULL == pytt2) goto fail;
    tai1 = (double *)PyArray_DATA(atai1);
    tai2 = (double *)PyArray_DATA(atai2);
    tt1 = (double *)PyArray_DATA(pytt1);
    tt2 = (double *)PyArray_DATA(pytt2);
    for (i=0;i<dims[0];i++) {
        status = eraTaitt(tai1[i], tai2[i], &tt1[i], &tt2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error...");
            goto fail;
        }
    }
    Py_DECREF(atai1);
    Py_DECREF(atai2);
    Py_INCREF(pytt1); Py_INCREF(pytt2);
    return Py_BuildValue("OO", pytt1, pytt2);

fail:
    Py_XDECREF(atai1);
    Py_XDECREF(atai2);
    Py_XDECREF(pytt1); Py_XDECREF(pytt2);
    return NULL;
}

PyDoc_STRVAR(_erfa_taitt_doc,
"\ntaitt(tai1, tai2) -> tt1, tt2\n\n"
"Time scale transformation:  International Atomic Time, TAI, to\n"
"Terrestrial Time, TT.\n"
"Given:\n"
"   tai1,tai2   TAI as a 2-part Julian Date\n"
"Returned:\n"
"   tt1,tt2     TT as a 2-part Julian Date");

static PyObject *
_erfa_tcbtdb(PyObject *self, PyObject *args)
{
    double *tcb1, *tcb2, *tdb1, *tdb2;
    int status;
    PyObject *pytcb1, *pytcb2;
    PyObject *atcb1, *atcb2;
    PyArrayObject *pytdb1 = NULL, *pytdb2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pytcb1,
                                 &PyArray_Type, &pytcb2))
        return NULL;

    atcb1 = PyArray_FROM_OTF(pytcb1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atcb2 = PyArray_FROM_OTF(pytcb2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atcb1 == NULL || atcb2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atcb1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atcb1);
    if (dims[0] != PyArray_DIMS(atcb2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytdb1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytdb2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytdb1 || NULL == pytdb2) goto fail;
    tcb1 = (double *)PyArray_DATA(atcb1);
    tcb2 = (double *)PyArray_DATA(atcb2);
    tdb1 = (double *)PyArray_DATA(pytdb1);
    tdb2 = (double *)PyArray_DATA(pytdb2);
    for (i=0;i<dims[0];i++) {
        status = eraTcbtdb(tcb1[i], tcb2[i], &tdb1[i], &tdb2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error...");
            goto fail;
        }
    }
    Py_DECREF(atcb1);
    Py_DECREF(atcb2);
    Py_INCREF(pytdb1); Py_INCREF(pytdb2);
    return Py_BuildValue("OO", pytdb1, pytdb2);

fail:
    Py_XDECREF(atcb1);
    Py_XDECREF(atcb2);
    Py_XDECREF(pytdb1); Py_XDECREF(pytdb2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tcbtdb_doc,
"\ntcbtdb(tcb1, tcb2) -> tdb1, tdb2\n\n"
"Time scale transformation:  Barycentric Coordinate Time, TCB, to\n"
"Barycentric Dynamical Time, TDB.\n"
"Given:\n"
"   tcb1,tcb2   TCB as a 2-part Julian Date\n"
"Returned:\n"
"   tdb1,tdb2   TDB as a 2-part Julian Date");

static PyObject *
_erfa_tcgtt(PyObject *self, PyObject *args)
{
    double *tcg1, *tcg2, *tt1, *tt2;
    int status;
    PyObject *pytcg1, *pytcg2;
    PyObject *atcg1, *atcg2;
    PyArrayObject *pytt1 = NULL, *pytt2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pytcg1,
                                 &PyArray_Type, &pytcg2))
        return NULL;

    atcg1 = PyArray_FROM_OTF(pytcg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atcg2 = PyArray_FROM_OTF(pytcg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atcg1 == NULL || atcg2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atcg1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atcg1);
    if (dims[0] != PyArray_DIMS(atcg2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }

    pytt1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytt2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytt1 || NULL == pytt2) goto fail;
    tcg1 = (double *)PyArray_DATA(atcg1);
    tcg2 = (double *)PyArray_DATA(atcg2);
    tt1 = (double *)PyArray_DATA(pytt1);
    tt2 = (double *)PyArray_DATA(pytt2);
    for (i=0;i<dims[0];i++) {
        status = eraTcgtt(tcg1[i], tcg2[i], &tt1[i], &tt2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error...");
            goto fail;
        }
    }
    Py_DECREF(atcg1);
    Py_DECREF(atcg2);
    Py_INCREF(pytt1); Py_INCREF(pytt2);
    return Py_BuildValue("OO", pytt1, pytt2);

fail:
    Py_XDECREF(atcg1);
    Py_XDECREF(atcg2);
    Py_XDECREF(pytt1); Py_XDECREF(pytt2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tcgtt_doc,
"\ntcgtt(tcg1, tcg2) -> tt1, tt2\n\n"
"Time scale transformation:  Geocentric Coordinate Time, TCG, to\n"
"Terrestrial Time, TT.\n"
"Given:\n"
"   tcg1,tcg2   TCG as a 2-part Julian Date\n"
"Returned:\n"
"   tt1,tt2   TT as a 2-part Julian Date");

static PyObject *
_erfa_tdbtcb(PyObject *self, PyObject *args)
{
    double *tdb1, *tdb2, *tcb1, *tcb2;
    int status;
    PyObject *pytdb1, *pytdb2;
    PyObject *atdb1, *atdb2;
    PyArrayObject *pytcb1 = NULL, *pytcb2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pytdb1,
                                 &PyArray_Type, &pytdb2))
        return NULL;

    atdb1 = PyArray_FROM_OTF(pytdb1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atdb2 = PyArray_FROM_OTF(pytdb2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atdb1 == NULL || atdb2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atdb1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atdb1);
    if (dims[0] != PyArray_DIMS(atdb2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }

    pytcb1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytcb2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytcb1 || NULL == pytcb2) goto fail;
    tdb1 = (double *)PyArray_DATA(atdb1);
    tdb2 = (double *)PyArray_DATA(atdb2);
    tcb1 = (double *)PyArray_DATA(pytcb1);
    tcb2 = (double *)PyArray_DATA(pytcb2);
    for (i=0;i<dims[0];i++) {
        status = eraTdbtcb(tdb1[i], tdb2[i], &tcb1[i], &tcb2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "should NOT occur...");
            goto fail;
        }
    }
    Py_DECREF(atdb1);
    Py_DECREF(atdb2);
    Py_INCREF(pytcb1); Py_INCREF(pytcb2);
    return Py_BuildValue("OO", pytcb1, pytcb2);

fail:
    Py_XDECREF(atdb1);
    Py_XDECREF(atdb2);
    Py_XDECREF(pytcb1); Py_XDECREF(pytcb2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tdbtcb_doc,
"\ntdbtcb(tdb1, tdb2) -> tcb1, tcb2\n\n"
"Time scale transformation:  Barycentric Dynamical Time, TDB, to\n"
"Barycentric Coordinate Time, TCB.\n"
"Given:\n"
"   tdb1,tdb2   TDB as a 2-part Julian Date\n"
"Returned:\n"
"   tcb1,tcb2   TCB as a 2-part Julian Date");

static PyObject *
_erfa_tttai(PyObject *self, PyObject *args)
{
    double *tai1, *tai2, *tt1, *tt2;
    int status;
    PyObject *pytai1, *pytai2;
    PyObject *atai1, *atai2;
    PyArrayObject *pytt1 = NULL, *pytt2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pytai1,
                                 &PyArray_Type, &pytai2))
        return NULL;

    atai1 = PyArray_FROM_OTF(pytai1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atai2 = PyArray_FROM_OTF(pytai2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atai1 == NULL || atai2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atai1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atai1);
    if (dims[0] != PyArray_DIMS(atai2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }

    pytt1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytt2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytt1 || NULL == pytt2) goto fail;
    tai1 = (double *)PyArray_DATA(atai1);
    tai2 = (double *)PyArray_DATA(atai2);
    tt1 = (double *)PyArray_DATA(pytt1);
    tt2 = (double *)PyArray_DATA(pytt2);
    for (i=0;i<dims[0];i++) {
        status = eraTttai(tai1[i], tai2[i], &tt1[i], &tt2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "should NOT occur...");
            goto fail;
        }
    }
    Py_DECREF(atai1);
    Py_DECREF(atai2);
    Py_INCREF(pytt1); Py_INCREF(pytt2);
    return Py_BuildValue("OO", pytt1, pytt2);

fail:
    Py_XDECREF(atai1);
    Py_XDECREF(atai2);
    Py_XDECREF(pytt1); Py_XDECREF(pytt2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tttai_doc,
"\ntttai(tt1, tt2) -> tai1, tai2\n\n"
"Time scale transformation: Terrestrial Time, TT, to\n"
"International Atomic Time, TAI.\n"
"Given:\n"
"   tt1,tt2     TT as a 2-part Julian Date\n"
"Returned:\n"
"   tai1,tai2   TAI as a 2-part Julian Date");

static PyObject *
_erfa_tttcg(PyObject *self, PyObject *args)
{
    double *tt1, *tt2, *tcg1, *tcg2;
    int status;
    PyObject *pytt1, *pytt2;
    PyObject *att1, *att2;
    PyArrayObject *pytcg1 = NULL, *pytcg2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pytt1,
                                 &PyArray_Type, &pytt2))
        return NULL;

    att1 = PyArray_FROM_OTF(pytt1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    att2 = PyArray_FROM_OTF(pytt2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (att1 == NULL || att2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(att1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(att1);
    if (dims[0] != PyArray_DIMS(att2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }

    pytcg1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytcg2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytcg1 || NULL == pytcg2) goto fail;
    tt1 = (double *)PyArray_DATA(att1);
    tt2 = (double *)PyArray_DATA(att2);
    tcg1 = (double *)PyArray_DATA(pytcg1);
    tcg2 = (double *)PyArray_DATA(pytcg2);
    for (i=0;i<dims[0];i++) {
        status = eraTttcg(tt1[i], tt2[i], &tcg1[i], &tcg2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "should NOT occur...");
            goto fail;
        }
    }
    Py_DECREF(att1);
    Py_DECREF(att2);
    Py_INCREF(pytcg1); Py_INCREF(pytcg2);
    return Py_BuildValue("OO", pytcg1, pytcg2);

fail:
    Py_XDECREF(att1);
    Py_XDECREF(att2);
    Py_XDECREF(pytcg1); Py_XDECREF(pytcg2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tttcg_doc,
"\ntttcg(tt1, tt2) -> tcg1, tcg2\n\n"
"Time scale transformation: Terrestrial Time, TT, to Geocentric\n"
"Coordinate Time, TCG.\n"
"Given:\n"
"   tt1,tt2     TT as a 2-part Julian Date\n"
"Returned:\n"
"   tcg1,tcg2   TCG as a 2-part Julian Date");

static PyObject *
_erfa_xy06(PyObject *self, PyObject *args)
{
    double *d1, *d2, *x, *y;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyx = NULL, *pyy = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyx = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyy = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyx || NULL == pyy) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(pyx);
    y = (double *)PyArray_DATA(pyy);
    for (i=0;i<dims[0];i++) {
        eraXy06(d1[i], d2[i], &x[i], &y[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyx);Py_INCREF(pyy);
    return Py_BuildValue("OO", pyx, pyy);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyx);Py_XDECREF(pyy);
    return NULL;
}

PyDoc_STRVAR(_erfa_xy06_doc,
"\nxy06(d1, d2) -> x, y\n\n"
"X,Y coordinates of celestial intermediate pole from series based\n"
"on IAU 2006 precession and IAU 2000A nutation.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   x,y     CIP X,Y coordinates");

static PyObject *
_erfa_xys00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, *x, *y, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyx = NULL, *pyy = NULL, *pys = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyx = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyy = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pys = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyx || NULL == pyy || NULL == pys) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(pyx);
    y = (double *)PyArray_DATA(pyy);
    s = (double *)PyArray_DATA(pys);
    for (i=0;i<dims[0];i++) {
        eraXys00a(d1[i], d2[i], &x[i], &y[i], &s[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyx);Py_INCREF(pyy);Py_INCREF(pys);
    return Py_BuildValue("OOO", pyx, pyy, pys);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyx);Py_XDECREF(pyy);Py_XDECREF(pys);
    return NULL;
}

PyDoc_STRVAR(_erfa_xys00a_doc,
"\nxys00a(d1, d2) -> x, y, s\n\n"
"For a given TT date, compute the X,Y coordinates of the Celestial\n"
"Intermediate Pole and the CIO locator s, using the IAU 2000A\n"
"precession-nutation model.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   x,y     Celestial Intermediate Pole\n"
"   s       the CIO locator s");

static PyObject *
_erfa_xys06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, *x, *y, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyx = NULL, *pyy = NULL, *pys = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!", 
                                 &PyArray_Type, &pyd1, &PyArray_Type, &pyd2))
        return NULL;

    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyx = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyy = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pys = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyx || NULL == pyy || NULL == pys) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(pyx);
    y = (double *)PyArray_DATA(pyy);
    s = (double *)PyArray_DATA(pys);
    for (i=0;i<dims[0];i++) {
        eraXys06a(d1[i], d2[i], &x[i], &y[i], &s[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyx);Py_INCREF(pyy);Py_INCREF(pys);
    return Py_BuildValue("OOO", pyx, pyy, pys);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyx);Py_XDECREF(pyy);Py_XDECREF(pys);
    return NULL;
}

PyDoc_STRVAR(_erfa_xys06a_doc,
"\nxys06a(d1, d2) -> x, y, s\n\n"
"For a given TT date, compute the X,Y coordinates of the Celestial\n"
"Intermediate Pole and the CIO locator s, using the IAU 2006\n"
"precession and IAU 2000A nutation model.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   x,y     Celestial Intermediate Pole\n"
"   s       the CIO locator s");

static PyObject *
_erfa_a2af(PyObject *self, PyObject *args)
{
    int ndp, idmsf[4];
    char sign;
    double *a;
    PyObject *pya;
    PyObject *aa = NULL, *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "iO!", &ndp, &PyArray_Type, &pya))      
        return NULL;
    ndim = PyArray_NDIM(pya);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pya);
    aa = PyArray_FROM_OTF(pya, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aa == NULL) goto fail;
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    a = (double *)PyArray_DATA(aa);
    for (i=0;i<dims[0];i++) {
        eraA2af(ndp, a[i], &sign, idmsf);
        if (PyList_SetItem(pyout, i, _a2xf_object(sign, idmsf))) {
            PyErr_SetString(_erfaError, "cannot set a2af into list");
            goto fail;
        }
    }
    Py_DECREF(aa);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(aa);
    Py_XDECREF(pyout);
    return NULL;  
}

PyDoc_STRVAR(_erfa_a2af_doc,
"\na2af(n, a) -> +/-, d, m, s, f\n\n"
"Decompose radians into degrees, arcminutes, arcseconds, fraction.\n"
"Given:\n"
"   n           resolution\n"
"   a           angle in radians\n"
"Returned:\n"
"   sign        '+' or '-'\n"
"   d           degrees\n"
"   m           arcminutes\n"
"   s           arcseconds\n"
"   f           fraction");

static PyObject *
_erfa_a2tf(PyObject *self, PyObject *args)
{
    int ndp, ihmsf[4];
    char sign;
    double *a;
    PyObject *pya;
    PyObject *aa = NULL, *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "iO!", &ndp, &PyArray_Type, &pya))      
        return NULL;
    ndim = PyArray_NDIM(pya);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pya);
    aa = PyArray_FROM_OTF(pya, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aa == NULL) goto fail;
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    a = (double *)PyArray_DATA(aa);
    for (i=0;i<dims[0];i++) {
        eraA2tf(ndp, a[i], &sign, ihmsf);
        if (PyList_SetItem(pyout, i, _a2xf_object(sign, ihmsf))) {
            PyErr_SetString(_erfaError, "cannot set a2af into list");
            goto fail;
        }
    }
    Py_DECREF(aa);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(aa);
    Py_XDECREF(pyout);
    return NULL;  
}

PyDoc_STRVAR(_erfa_a2tf_doc,
"\na2tf(n, a) -> +/-, h, m, s, f\n\n"
"Decompose radians into hours, minutes, seconds, fraction.\n"
"Given:\n"
"   n           resolution\n"
"   a           angle in radians\n"
"Returned:\n"
"   sign        '+' or '-'\n"
"   h           hours\n"
"   m           minutes\n"
"   s           seconds\n"
"   f           fraction");

static PyObject *
_erfa_anp(PyObject *self, PyObject *args)
{
    double *a, *out;
    PyObject *pya;
    PyObject *aa = NULL;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pya))      
        return NULL;
    ndim = PyArray_NDIM(pya);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pya);
    aa = PyArray_FROM_OTF(pya, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aa == NULL) goto fail;
    
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    a = (double *)PyArray_DATA(aa);
    out = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        out[i] = eraAnp(a[i]);
    }
    Py_DECREF(aa);
    Py_INCREF(pyout);
    return (PyObject *)pyout;    

fail:
    Py_XDECREF(aa);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_anp_doc,
"\nanp(a) -> 0 <= a < 2pi\n\n"
"Normalize angle into the range 0 <= a < 2pi.\n"
"Given:\n"
"    a          angle (radians)\n"
"Returned:\n"
"    a          angle in range 0-2pi");

static PyObject *
_erfa_cr(PyObject *self, PyObject *args)
{
    PyArrayObject *r, *c;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &r)) {
        return NULL;
    }
    c = (PyArrayObject *)PyArray_NewCopy(r, NPY_CORDER);
    if (c == NULL) return NULL;
    return (PyObject *)c;
}

PyDoc_STRVAR(_erfa_cr_doc,
"\ncr(r) -> c\n\n"
"Copy an r-matrix.\n"
"Given:\n"
"   r           r-matrix to be copied\n"
"  Returned:\n"
"   c           copy");

static PyObject *
_erfa_rxp(PyObject *self, PyObject *args)
{
    double r[3][3], p[3], rp[3];
    PyObject *pyr, *pyp;
    PyObject *ar = NULL, *ap = NULL;
    PyArrayObject *pyrp = NULL;
    PyObject *r_iter = NULL, *p_iter = NULL, *rp_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyr,
                                 &PyArray_Type, &pyp))      
        return NULL;
    ndim = PyArray_NDIM(pyr);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyr);
    if (dims[0] != PyArray_DIMS(pyp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    pyrp = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyrp) goto fail;
    r_iter = PyArray_IterNew((PyObject *)pyr);
    p_iter = PyArray_IterNew((PyObject *)pyp);
    rp_iter = PyArray_IterNew((PyObject *)pyrp);
    if (r_iter == NULL || p_iter == NULL || rp_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k;
        double vr, vp;
        for (j=0;j<3;j++) {
            ap = PyArray_GETITEM(pyp, PyArray_ITER_DATA(p_iter));
            if (ap == NULL) {
                PyErr_SetString(_erfaError, "cannot retrieve data from args");
                goto fail;
            }
            Py_INCREF(ap);
            vp = (double)PyFloat_AsDouble(ap);
            if (vp == -1 && PyErr_Occurred()) goto fail;
            p[j] = vp;
            Py_DECREF(ap);
            PyArray_ITER_NEXT(p_iter);
            for (k=0;k<3;k++) {
                ar = PyArray_GETITEM(pyr, PyArray_ITER_DATA(r_iter));
                if (ar == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(ar);
                vr = (double)PyFloat_AsDouble(ar);
                if (vr == -1 && PyErr_Occurred()) goto fail;
                r[j][k] = vr;
                Py_DECREF(ar);                
                PyArray_ITER_NEXT(r_iter); 
            }
        }
        eraRxp(r, p, rp);
        for (j=0;j<3;j++) {
            if (PyArray_SETITEM(pyrp, PyArray_ITER_DATA(rp_iter), PyFloat_FromDouble(rp[j]))) {
                PyErr_SetString(_erfaError, "unable to set output rp");
                goto fail;
            }
            PyArray_ITER_NEXT(rp_iter);
        }
    }
    Py_DECREF(ar);
    Py_DECREF(ap);
    Py_DECREF(r_iter);
    Py_DECREF(p_iter);
    Py_DECREF(rp_iter);
    Py_INCREF(pyrp);
    return (PyObject *)pyrp;    

fail:
    Py_XDECREF(ar);
    Py_XDECREF(ap);
    Py_XDECREF(r_iter);
    Py_XDECREF(p_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(pyrp);
    return NULL;
}

PyDoc_STRVAR(_erfa_rxp_doc,
"\nrxp(r, p) -> rp\n\n"
"Multiply a p-vector by an r-matrix.\n"
"Given:\n"
"   r           r-matrix\n"
"   p           p-vector\n"
"Returned:\n"
"   rp          r * p");

static PyObject *
_erfa_rxr(PyObject *self, PyObject *args)
{
    double a[3][3], b[3][3], atb[3][3];
    PyObject *pya, *pyb;
    PyObject *aa = NULL, *ab = NULL;
    PyArrayObject *pyatb = NULL;
    PyObject *a_iter = NULL, *b_iter = NULL, *atb_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pya,
                                 &PyArray_Type, &pyb))      
        return NULL;
    ndim = PyArray_NDIM(pya);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pya);
    if (dims[0] != PyArray_DIMS(pyb)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyatb = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyatb) goto fail;
    a_iter = PyArray_IterNew((PyObject *)pya);
    b_iter = PyArray_IterNew((PyObject *)pyb);
    atb_iter = PyArray_IterNew((PyObject *)pyatb);
    if (a_iter == NULL || b_iter == NULL || atb_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k;
        double va, vb;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                aa = PyArray_GETITEM(pya, PyArray_ITER_DATA(a_iter));
                ab = PyArray_GETITEM(pyb, PyArray_ITER_DATA(b_iter));
                if (aa == NULL || ab == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(aa);Py_INCREF(ab);
                va = (double)PyFloat_AsDouble(aa);
                if (va == -1 && PyErr_Occurred()) goto fail;
                a[j][k] = va;
                vb = (double)PyFloat_AsDouble(ab);
                if (vb == -1 && PyErr_Occurred()) goto fail;
                b[j][k] = vb;
                Py_DECREF(aa);Py_DECREF(ab);
                PyArray_ITER_NEXT(a_iter); 
                PyArray_ITER_NEXT(b_iter); 
            }
        }
        eraRxr(a, b, atb);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyatb, PyArray_ITER_DATA(atb_iter), PyFloat_FromDouble(atb[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set output atb");
                    goto fail;
                }
                PyArray_ITER_NEXT(atb_iter);
            }
        }
    }
    Py_DECREF(aa);
    Py_DECREF(ab);
    Py_DECREF(a_iter);
    Py_DECREF(b_iter);
    Py_DECREF(atb_iter);
    Py_INCREF(pyatb);
    return (PyObject *)pyatb;    

fail:
    Py_XDECREF(aa);
    Py_XDECREF(ab);
    Py_XDECREF(a_iter);
    Py_XDECREF(b_iter);
    Py_XDECREF(atb_iter);
    Py_XDECREF(pyatb);
    return NULL;
}

PyDoc_STRVAR(_erfa_rxr_doc,
"\nrxr(a, b -> atb\n\n"
"Multiply two r-matrices.\n"
"Given:\n"
"   a           first r-matrix\n"
"   b           second r-matrix\n"
"Returned:\n"
"   atb         a * b");

static PyObject *
_erfa_rx(PyObject *self, PyObject *args)
{
    double r[3][3], *phi;
    PyObject *pyphi, *pyr;
    PyObject *aphi, *ar = NULL;
    PyArrayObject *pyout = NULL;
    PyObject *r_iter = NULL, *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyphi,
                                 &PyArray_Type, &pyr))
        return NULL;

    aphi = PyArray_FROM_OTF(pyphi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aphi == NULL) goto fail;
    ndim = PyArray_NDIM(aphi);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aphi);
    if (dims[0] != PyArray_DIMS(pyr)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    phi = (double *)PyArray_DATA(aphi);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    r_iter = PyArray_IterNew((PyObject*)pyr);
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL || r_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vr;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                ar = PyArray_GETITEM(pyr, PyArray_ITER_DATA(r_iter));
                if (ar == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(ar);
                vr = (double)PyFloat_AsDouble(ar);
                if (vr == -1 && PyErr_Occurred()) goto fail;
                r[j][k] = vr;
                Py_DECREF(ar);
                PyArray_ITER_NEXT(r_iter); 
            }
        }
        eraRx(phi[i], r);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(r[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(aphi);
    Py_DECREF(ar);
    Py_DECREF(r_iter);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(aphi);
    Py_XDECREF(ar);
    Py_XDECREF(r_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_rx_doc,
"\nrx(phi, r) -> r\n\n"
"Rotate an r-matrix about the y-axis.\n"
"Given:\n"
"   phi         angle (radians)\n"
"Given and returned:\n"
"   r           r-matrix, rotated");

static PyObject *
_erfa_ry(PyObject *self, PyObject *args)
{
    double r[3][3], *theta;
    PyObject *pytheta, *pyr;
    PyObject *atheta, *ar = NULL;
    PyArrayObject *pyout = NULL;
    PyObject *r_iter = NULL, *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pytheta,
                                 &PyArray_Type, &pyr))
        return NULL;

    atheta = PyArray_FROM_OTF(pytheta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atheta == NULL) goto fail;
    ndim = PyArray_NDIM(atheta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atheta);
    if (dims[0] != PyArray_DIMS(pyr)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    theta = (double *)PyArray_DATA(atheta);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    r_iter = PyArray_IterNew((PyObject*)pyr);
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL || r_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vr;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                ar = PyArray_GETITEM(pyr, PyArray_ITER_DATA(r_iter));
                if (ar == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(ar);
                vr = (double)PyFloat_AsDouble(ar);
                if (vr == -1 && PyErr_Occurred()) goto fail;
                r[j][k] = vr;
                Py_DECREF(ar);
                PyArray_ITER_NEXT(r_iter); 
            }
        }
        eraRy(theta[i], r);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(r[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(atheta);
    Py_DECREF(ar);
    Py_DECREF(r_iter);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(atheta);
    Py_XDECREF(ar);
    Py_XDECREF(r_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_ry_doc,
"\nry(theta, r) -> r\n\n"
"Rotate an r-matrix about the y-axis.\n"
"Given:\n"
"   theta         angle (radians)\n"
"Given and returned:\n"
"   r           r-matrix, rotated");

static PyObject *
_erfa_rz(PyObject *self, PyObject *args)
{
    double r[3][3], *psi;
    PyObject *pypsi, *pyr;
    PyObject *apsi, *ar = NULL;
    PyArrayObject *pyout = NULL;
    PyObject *r_iter = NULL, *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pypsi,
                                 &PyArray_Type, &pyr))
        return NULL;

    apsi = PyArray_FROM_OTF(pypsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (apsi == NULL) goto fail;
    ndim = PyArray_NDIM(apsi);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(apsi);
    if (dims[0] != PyArray_DIMS(pyr)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    psi = (double *)PyArray_DATA(apsi);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyout = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    r_iter = PyArray_IterNew((PyObject*)pyr);
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL || r_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vr;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                ar = PyArray_GETITEM(pyr, PyArray_ITER_DATA(r_iter));
                if (ar == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(ar);
                vr = (double)PyFloat_AsDouble(ar);
                if (vr == -1 && PyErr_Occurred()) goto fail;
                r[j][k] = vr;
                Py_DECREF(ar);
                PyArray_ITER_NEXT(r_iter); 
            }
        }
        eraRz(psi[i], r);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(r[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(apsi);
    Py_DECREF(ar);
    Py_DECREF(r_iter);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(apsi);
    Py_XDECREF(ar);
    Py_XDECREF(r_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_rz_doc,
"\nrz(psi, r) -> r\n\n"
"Rotate an r-matrix about the z-axis.\n"
"Given:\n"
"   psi         angle (radians)\n"
"Given and returned:\n"
"   r           r-matrix, rotated");


static PyMethodDef _erfa_methods[] = {
    {"ab", _erfa_ab, METH_VARARGS, _erfa_ab_doc},
    {"apcs", _erfa_apcs, METH_VARARGS, _erfa_apcs_doc},
    {"ld", _erfa_ld, METH_VARARGS, _erfa_ld_doc},
    {"pmsafe", _erfa_pmsafe, METH_VARARGS, _erfa_pmsafe_doc},
    {"c2ixys", _erfa_c2ixys, METH_VARARGS, _erfa_c2ixys_doc},
    {"cal2jd", _erfa_cal2jd, METH_VARARGS, _erfa_cal2jd_doc},
    {"d2dtf", _erfa_d2dtf, METH_VARARGS, _erfa_d2dtf_doc},
    {"dat", _erfa_dat, METH_VARARGS, _erfa_dat_doc},
    {"dtf2d", _erfa_dtf2d, METH_VARARGS, _erfa_dtf2d_doc},
    {"ee00", _erfa_ee00, METH_VARARGS, _erfa_ee00_doc},
    {"epb", _erfa_epb, METH_VARARGS, _erfa_epb_doc},
    {"epb2jd", _erfa_epb2jd, METH_VARARGS, _erfa_epb2jd_doc},
    {"epj", _erfa_epj, METH_VARARGS, _erfa_epj_doc},
    {"epj2jd", _erfa_epj2jd, METH_VARARGS, _erfa_epj2jd_doc},
    {"eqeq94", _erfa_eqeq94, METH_VARARGS, _erfa_eqeq94_doc},
    {"era00", _erfa_era00, METH_VARARGS, _erfa_era00_doc},
    {"gmst00", _erfa_gmst00, METH_VARARGS, _erfa_gmst00_doc},
    {"gmst82", _erfa_gmst82, METH_VARARGS, _erfa_gmst82_doc},
    {"numat", _erfa_numat, METH_VARARGS, _erfa_numat_doc},
    {"nut00a", _erfa_nut00a, METH_VARARGS, _erfa_nut00a_doc},
    {"nut80", _erfa_nut80, METH_VARARGS, _erfa_nut80_doc},
    {"obl80", _erfa_obl80, METH_VARARGS, _erfa_obl80_doc},
    {"plan94", _erfa_plan94, METH_VARARGS, _erfa_plan94_doc},
    {"pmat76", _erfa_pmat76, METH_VARARGS, _erfa_pmat76_doc},
    {"pn00", _erfa_pn00, METH_VARARGS, _erfa_pn00_doc},
    {"pom00", _erfa_pom00, METH_VARARGS, _erfa_pom00_doc},
    {"s00", _erfa_s00, METH_VARARGS, _erfa_s00_doc},
    {"s06", _erfa_s06, METH_VARARGS, _erfa_s06_doc},
    {"sp00", _erfa_sp00, METH_VARARGS, _erfa_sp00_doc},
    {"taitt", _erfa_taitt, METH_VARARGS, _erfa_taitt_doc},
    {"tcbtdb", _erfa_tcbtdb, METH_VARARGS, _erfa_tcbtdb_doc},
    {"tcgtt", _erfa_tcgtt, METH_VARARGS, _erfa_tcgtt_doc},
    {"tdbtcb", _erfa_tdbtcb, METH_VARARGS, _erfa_tdbtcb_doc},
    {"tttai", _erfa_tttai, METH_VARARGS, _erfa_tttai_doc},
    {"tttcg", _erfa_tttcg, METH_VARARGS, _erfa_tttcg_doc},
    {"xy06", _erfa_xy06, METH_VARARGS, _erfa_xy06_doc},
    {"xys00a", _erfa_xys00a, METH_VARARGS, _erfa_xys00a_doc},
    {"xys06a", _erfa_xys06a, METH_VARARGS, _erfa_xys06a_doc},
    {"a2af", _erfa_a2af, METH_VARARGS, _erfa_a2af_doc},
    {"a2tf", _erfa_a2tf, METH_VARARGS, _erfa_a2tf_doc},
    {"anp", _erfa_anp, METH_VARARGS, _erfa_anp_doc},
    {"cr", _erfa_cr, METH_VARARGS, _erfa_cr_doc},
    {"rxp", _erfa_rxp, METH_VARARGS, _erfa_rxp_doc},
    {"rxr", _erfa_rxr, METH_VARARGS, _erfa_rxr_doc},
    {"rx", _erfa_rx, METH_VARARGS, _erfa_rx_doc},
    {"ry", _erfa_ry, METH_VARARGS, _erfa_ry_doc},
    {"rz", _erfa_rz, METH_VARARGS, _erfa_rz_doc},
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
