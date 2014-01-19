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
_erfa_apcg(PyObject *self, PyObject *args)
{
    double *date1, *date2, ebpv[2][3], ehp[3];
    PyObject *adate1, *adate2, *aebpv, *aehp;
    PyObject *pydate1, *pydate2, *pyebpv, *pyehp;
    PyObject *ebpv_iter = NULL, *ehp_iter = NULL;
    PyObject *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2,
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
    if (dims[0] != PyArray_DIMS(adate2)[0] ||
        dims[0] != PyArray_DIMS(pyebpv)[0] ||
        dims[0] != PyArray_DIMS(pyehp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    date1 = (double *)PyArray_DATA(adate1);
    date2 = (double *)PyArray_DATA(adate2);
    ebpv_iter = PyArray_IterNew((PyObject *)pyebpv);
    ehp_iter = PyArray_IterNew((PyObject *)pyehp);
    if (ebpv_iter == NULL || ehp_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k,l;
        double e, h;
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
                aebpv = PyArray_GETITEM(pyebpv, PyArray_ITER_DATA(ebpv_iter));                
                if (aebpv == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(aebpv);
                e = (double)PyFloat_AsDouble(aebpv);
                if (e == -1 && PyErr_Occurred()) goto fail;
                ebpv[j][k] = e;
                Py_DECREF(aebpv);
                PyArray_ITER_NEXT(ebpv_iter); 
            }
        }
        eraApcg(date1[i], date2[i], ebpv, ehp, &astrom);
        if (PyList_SetItem(pyout, i, _to_py_astrom(&astrom))) {
            PyErr_SetString(_erfaError, "cannot set astrom into list");
            goto fail;
        }       
    }
    Py_DECREF(adate1);
    Py_DECREF(adate2);
    Py_DECREF(ebpv_iter);
    Py_DECREF(ehp_iter);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(adate1);
    Py_XDECREF(adate2);
    Py_XDECREF(ebpv_iter);
    Py_XDECREF(ehp_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_apcg_doc,
"\napcg(date1, date2, ebpv[2][3], ehp[3]) -> astrom\n"
"For a geocentric observer, prepare star-independent astrometry\n"
"parameters for transformations between ICRS and GCRS coordinates.\n"
"The Earth ephemeris is supplied by the caller.\n"
"\n"
"The parameters produced by this function are required in the\n"
"parallax, light deflection and aberration parts of the astrometric\n"
"transformation chain.\n"
"Given:\n"
"    date1      TDB as a 2-part...\n"
"    date2      ...Julian Date\n"
"    ebpv       Earth barycentric pos/vel (au, au/day)\n"
"    ehp        Earth heliocentric position (au)\n"
"Returned:\n"
"    astrom     star-independent astrometry parameters");

static PyObject *
_erfa_apcg13(PyObject *self, PyObject *args)
{
    double *date1, *date2;
    PyObject *adate1, *adate2;
    PyObject *pydate1, *pydate2;
    PyObject *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2))      
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
    if (dims[0] != PyArray_DIMS(adate2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    date1 = (double *)PyArray_DATA(adate1);
    date2 = (double *)PyArray_DATA(adate2);
    for (i=0;i<dims[0];i++) {
        eraApcg13(date1[i], date2[i], &astrom);
        if (PyList_SetItem(pyout, i, _to_py_astrom(&astrom))) {
            PyErr_SetString(_erfaError, "cannot set astrom into list");
            goto fail;
        }       
    }
    Py_DECREF(adate1);
    Py_DECREF(adate2);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(adate1);
    Py_XDECREF(adate2);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_apcg13_doc,
"\napcg13(date1, date2) -> astrom\n"
"For a geocentric observer, prepare star-independent astrometry\n"
"parameters for transformations between ICRS and GCRS coordinates.\n"
"The caller supplies the date, and ERFA models are used to predict\n"
"the Earth ephemeris.\n"
"\n"
"The parameters produced by this function are required in the\n"
"parallax, light deflection and aberration parts of the astrometric\n"
"transformation chain.\n"
"Given:\n"
"    date1      TDB as a 2-part...\n"
"    date2      ...Julian Date\n"
"Returned:\n"
"    astrom     star-independent astrometry parameters");

static PyObject *
_erfa_apci(PyObject *self, PyObject *args)
{
    double *date1, *date2, ebpv[2][3], ehp[3], *x, *y, *s;
    PyObject *adate1, *adate2, *aebpv, *aehp, *ax, *ay, *as;
    PyObject *pydate1, *pydate2, *pyebpv, *pyehp, *pyx, *pyy, *pys;
    PyObject *ebpv_iter = NULL, *ehp_iter = NULL;
    PyObject *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2,
                                 &PyArray_Type, &pyebpv,
                                 &PyArray_Type, &pyehp,
                                 &PyArray_Type, &pyx,
                                 &PyArray_Type, &pyy,
                                 &PyArray_Type, &pys))      
        return NULL;
    adate1 = PyArray_FROM_OTF(pydate1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adate2 = PyArray_FROM_OTF(pydate2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (adate1 == NULL || adate2 == NULL || ax == NULL ||
        ay == NULL || as == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(adate1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(adate1);
    if (dims[0] != PyArray_DIMS(adate2)[0] ||
        dims[0] != PyArray_DIMS(pyebpv)[0] || dims[0] != PyArray_DIMS(pyehp)[0] ||
        dims[0] != PyArray_DIMS(pyx)[0] || dims[0] != PyArray_DIMS(pyy)[0] ||
        dims[0] != PyArray_DIMS(pys)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    date1 = (double *)PyArray_DATA(adate1);
    date2 = (double *)PyArray_DATA(adate2);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    s = (double *)PyArray_DATA(as);
    ebpv_iter = PyArray_IterNew((PyObject *)pyebpv);
    ehp_iter = PyArray_IterNew((PyObject *)pyehp);
    if (ebpv_iter == NULL || ehp_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k,l;
        double e, h;
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
                aebpv = PyArray_GETITEM(pyebpv, PyArray_ITER_DATA(ebpv_iter));                
                if (aebpv == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(aebpv);
                e = (double)PyFloat_AsDouble(aebpv);
                if (e == -1 && PyErr_Occurred()) goto fail;
                ebpv[j][k] = e;
                Py_DECREF(aebpv);
                PyArray_ITER_NEXT(ebpv_iter); 
            }
        }
        eraApci(date1[i], date2[i], ebpv, ehp, x[i], y[i], s[i], &astrom);
        if (PyList_SetItem(pyout, i, _to_py_astrom(&astrom))) {
            PyErr_SetString(_erfaError, "cannot set astrom into list");
            goto fail;
        }       
    }
    Py_DECREF(adate1);
    Py_DECREF(adate2);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(as);
    Py_DECREF(ebpv_iter);
    Py_DECREF(ehp_iter);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(adate1);
    Py_XDECREF(adate2);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(as);
    Py_XDECREF(ebpv_iter);
    Py_XDECREF(ehp_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_apci_doc,
"\napci(date1, date2, ebpv[2][3], ehp[3], x, y, s) -> astrom\n"
"For a terrestrial observer, prepare star-independent astrometry\n"
"parameters for transformations between ICRS and geocentric CIRS\n"
"coordinates.  The Earth ephemeris and CIP/CIO are supplied by the caller.\n"
"\n"
"The parameters produced by this function are required in the\n"
"parallax, light deflection, aberration, and bias-precession-nutation\n"
"parts of the astrometric transformation chain.\n"
"Given:\n"
"    date1      TDB as a 2-part...\n"
"    date2      ...Julian Date\n"
"    ebpv       Earth barycentric pos/vel (au, au/day)\n"
"    ehp        Earth heliocentric position (au)\n"
"    x,y        CIP X,Y (components of unit vector)\n"
"    s          the CIO locator s (radians)\n"
"Returned:\n"
"    astrom     star-independent astrometry parameters");

static PyObject *
_erfa_apco(PyObject *self, PyObject *args)
{
    double *date1, *date2, ebpv[2][3], ehp[3], *x, *y, *s;
    double *theta, *elong, *phi, *hm, *xp, *yp, *sp, *refa, *refb;
    PyObject *pydate1, *pydate2, *pyebpv, *pyehp, *pyx, *pyy, *pys;
    PyObject *pytheta, *pyelong, *pyphi, *pyhm, *pyxp, *pyyp, *pysp, *pyrefa, *pyrefb;
    PyObject *adate1, *adate2, *aebpv, *aehp, *ax, *ay, *as;
    PyObject *atheta, *aelong, *aphi, *ahm, *axp, *ayp, *asp, *arefa, *arefb;
    PyObject *ebpv_iter = NULL, *ehp_iter = NULL;
    PyObject *pyout = NULL;
    npy_intp *dims;
    int ndim, i;
    eraASTROM astrom;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!",
                                 &PyArray_Type, &pydate1,
                                 &PyArray_Type, &pydate2,
                                 &PyArray_Type, &pyebpv,
                                 &PyArray_Type, &pyehp,
                                 &PyArray_Type, &pyx,
                                 &PyArray_Type, &pyy,
                                 &PyArray_Type, &pys,
                                 &PyArray_Type, &pytheta,
                                 &PyArray_Type, &pyelong,
                                 &PyArray_Type, &pyphi,
                                 &PyArray_Type, &pyhm,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp,
                                 &PyArray_Type, &pysp,
                                 &PyArray_Type, &pyrefa,
                                 &PyArray_Type, &pyrefb))      
        return NULL;
    adate1 = PyArray_FROM_OTF(pydate1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adate2 = PyArray_FROM_OTF(pydate2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atheta = PyArray_FROM_OTF(pytheta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aelong = PyArray_FROM_OTF(pyelong, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphi = PyArray_FROM_OTF(pyphi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ahm = PyArray_FROM_OTF(pyhm, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    asp = PyArray_FROM_OTF(pysp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arefa = PyArray_FROM_OTF(pyrefa, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arefb = PyArray_FROM_OTF(pyrefb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (adate1 == NULL || adate2 == NULL || ax == NULL || ay == NULL || as == NULL ||
        atheta == NULL || aelong == NULL || aphi == NULL || ahm == NULL || axp == NULL ||
        ayp == NULL || asp == NULL || arefa == NULL || arefb == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(adate1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(adate1);
    if (dims[0] != PyArray_DIMS(adate2)[0] ||
        dims[0] != PyArray_DIMS(pyebpv)[0] || dims[0] != PyArray_DIMS(pyehp)[0] ||
        dims[0] != PyArray_DIMS(pyx)[0] || dims[0] != PyArray_DIMS(pyy)[0] || 
        dims[0] != PyArray_DIMS(pys)[0] || dims[0] != PyArray_DIMS(pytheta)[0] || 
        dims[0] != PyArray_DIMS(pyelong)[0] || dims[0] != PyArray_DIMS(pyphi)[0] || 
        dims[0] != PyArray_DIMS(pyhm)[0] || dims[0] != PyArray_DIMS(pyxp)[0] || 
        dims[0] != PyArray_DIMS(pyyp)[0] || dims[0] != PyArray_DIMS(pysp)[0] || 
         dims[0] != PyArray_DIMS(pyrefa)[0] || dims[0] != PyArray_DIMS(pyrefb)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
    pyout = PyList_New(dims[0]);
    if (NULL == pyout)  goto fail;
    date1 = (double *)PyArray_DATA(adate1);
    date2 = (double *)PyArray_DATA(adate2);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    s = (double *)PyArray_DATA(as);
    theta = (double *)PyArray_DATA(atheta);
    elong = (double *)PyArray_DATA(aelong);
    phi = (double *)PyArray_DATA(aphi);
    hm = (double *)PyArray_DATA(ahm);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    sp = (double *)PyArray_DATA(asp);
    refa = (double *)PyArray_DATA(arefa);
    refb = (double *)PyArray_DATA(arefb);
    ebpv_iter = PyArray_IterNew((PyObject *)pyebpv);
    ehp_iter = PyArray_IterNew((PyObject *)pyehp);
    if (ebpv_iter == NULL || ehp_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k,l;
        double e, h;
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
                aebpv = PyArray_GETITEM(pyebpv, PyArray_ITER_DATA(ebpv_iter));                
                if (aebpv == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(aebpv);
                e = (double)PyFloat_AsDouble(aebpv);
                if (e == -1 && PyErr_Occurred()) goto fail;
                ebpv[j][k] = e;
                Py_DECREF(aebpv);
                PyArray_ITER_NEXT(ebpv_iter); 
            }
        }
        eraApco(date1[i], date2[i], ebpv, ehp, x[i], y[i], s[i],
                theta[i], elong[i], phi[i], hm[i], xp[i], yp[i], sp[i], refa[i], refb[i],
                &astrom);
        if (PyList_SetItem(pyout, i, _to_py_astrom(&astrom))) {
            PyErr_SetString(_erfaError, "cannot set astrom into list");
            goto fail;
        }       
    }
    Py_DECREF(adate1);
    Py_DECREF(adate2);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(as);
    Py_DECREF(atheta);
    Py_DECREF(aelong);
    Py_DECREF(aphi);
    Py_DECREF(ahm);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(asp);
    Py_DECREF(arefa);
    Py_DECREF(arefb);
    Py_DECREF(ebpv_iter);
    Py_DECREF(ehp_iter);
    Py_INCREF(pyout);
    return (PyObject*)pyout;

fail:
    Py_XDECREF(adate1);
    Py_XDECREF(adate2);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(as);
    Py_XDECREF(atheta);
    Py_XDECREF(aelong);
    Py_XDECREF(aphi);
    Py_XDECREF(ahm);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(asp);
    Py_XDECREF(arefa);
    Py_XDECREF(arefb);
    Py_XDECREF(ebpv_iter);
    Py_XDECREF(ehp_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_apco_doc,
"\napco(date1, date2, ebpv[2][3], ehp[3], x, y, s,theta,,elong, phi, hm,xp, yp, sp, refa, refb) -> astrom\n"
"For a terrestrial observer, prepare star-independent astrometry\n"
"parameters for transformations between ICRS and observed\n"
"coordinates. The caller supplies the Earth ephemeris, the Earth\n"
"rotation information and the refraction constants as well as the\n"
"site coordinates.\n"
"Given:\n"
"    date1      TDB as a 2-part...\n"
"    date2      ...Julian Date\n"
"    ebpv       Earth barycentric pos/vel (au, au/day)\n"
"    ehp        Earth heliocentric position (au)\n"
"    x,y        CIP X,Y (components of unit vector)\n"
"    s          the CIO locator s (radians)\n"
"    theta      Earth rotation angle (radians)\n"
"    elong      longitude (radians, east +ve)\n"
"    phi        latitude (geodetic, radians)\n"
"    hm         height above ellipsoid (m, geodetic3)\n"
"    xp,yp      polar motion coordinates (radians)\n"
"    sp         the TIO locator s' (radians)\n"
"    refa       refraction constant A (radians)\n"
"    refb       refraction constant B (radians)\n"
"Returned:\n"
"    astrom     star-independent astrometry parameters");

static PyObject *
_erfa_apcs(PyObject *self, PyObject *args)
{
    double *date1, *date2, pv[2][3], ebpv[2][3], ehp[3];
    PyObject *adate1, *adate2, *apv, *aebpv, *aehp;
    PyObject *pydate1, *pydate2, *pypv, *pyebpv, *pyehp;
    PyObject *pv_iter = NULL, *ebpv_iter = NULL, *ehp_iter = NULL;
    PyObject *pyout = NULL;
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
    if (dims[0] != PyArray_DIMS(adate2)[0] ||
        dims[0] != PyArray_DIMS(pypv)[0] || dims[0] != PyArray_DIMS(pyebpv)[0] ||
        dims[0] != PyArray_DIMS(pyehp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
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
    return (PyObject*)pyout;

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
_erfa_pvstar(PyObject *self, PyObject *args)
{
    double pv[2][3], *ra, *dec, *pmr, *pmd, *px, *rv;
    int status;
    PyObject *pypv, *apv;
    PyObject *pv_iter = NULL;
    PyArrayObject *pyra = NULL, *pydec = NULL, *pypmr = NULL, *pypmd = NULL, *pypx = NULL, *pyrv = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pypv)) {
        return NULL;
    }
    pv_iter = PyArray_IterNew((PyObject *)pypv);
    if (pv_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    ndim = PyArray_NDIM(pypv);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pypv);
    dim_out[0] = dims[0];
    pyra = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pydec = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pypmr = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pypmd = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pypx = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyrv = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyra || NULL == pydec || NULL == pypmr ||
        NULL == pypmd || NULL == pypx || NULL == pyrv)
        goto fail;
    ra = (double *)PyArray_DATA(pyra);
    dec = (double *)PyArray_DATA(pydec);
    pmr = (double *)PyArray_DATA(pypmr);
    pmd = (double *)PyArray_DATA(pypmd);
    px = (double *)PyArray_DATA(pypx);
    rv = (double *)PyArray_DATA(pyrv);

    for (i=0;i<dims[0];i++) {
        int j,k;
        double p;
        for (j=0;j<2;j++) {
            for (k=0;k<3;k++) {
                apv = PyArray_GETITEM(pypv, PyArray_ITER_DATA(pv_iter));
                if (apv == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(apv);
                p = (double)PyFloat_AsDouble(apv);
                if (p == -1 && PyErr_Occurred()) goto fail;
                pv[j][k] = p;
                Py_DECREF(apv);
                PyArray_ITER_NEXT(pv_iter);
            }
        }
        status = eraPvstar(pv, &ra[i], &dec[i], &pmr[i], &pmd[i], &px[i], &rv[i]);
        if (status == -1) {
            PyErr_SetString(_erfaError, "superluminal speed");
            goto fail;
        }
        if (status == -2) {
            PyErr_SetString(_erfaError, "null position vector");
            goto fail;
        }
    }
    Py_DECREF(pv_iter);
    Py_INCREF(pyra);
    Py_INCREF(pydec);
    Py_INCREF(pypmr);
    Py_INCREF(pypmd);
    Py_INCREF(pypx);
    Py_INCREF(pyrv);
    return Py_BuildValue("OOOOOO", pyra, pydec, pypmr, pypmd, pypx, pyrv);

fail:
    Py_XDECREF(pv_iter);
    Py_XDECREF(pyra);
    Py_XDECREF(pydec);
    Py_XDECREF(pypmr);
    Py_XDECREF(pypmd);
    Py_XDECREF(pypx);
    Py_XDECREF(pyrv);
    return NULL;
}

PyDoc_STRVAR(_erfa_pvstar_doc,
"\npvstar(pv) -> ra, dec, pmr, pmd, px, rv\n\n"
"Convert star position+velocity vector to catalog coordinates.\n"
"Given:\n"
"   pv          pv-vector (AU, AU/day)\n"
"Returned:\n"
"   ra          right ascension (radians)\n"
"   dec         declination (radians)\n"
"   pmr         RA proper motion (radians/year)\n"
"   pmd         Dec proper motion (radians/year)\n"
"   px          parallax (arcsec)\n"
"   rv          radial velocity (km/s, positive = receding)");

static PyObject *
_erfa_starpv(PyObject *self, PyObject *args)
{
   double *ra, *dec, *pmr, *pmd, *px, *rv, pv[2][3];
   int status;
    PyObject *pyra, *pydec, *pypmr, *pypmd, *pypx, *pyrv;
    PyObject *ara, *adec, *apmr, *apmd, *apx, *arv;
    PyArrayObject *pypv = NULL;
    PyObject *pv_iter = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                                 &PyArray_Type, &pyra,
                                 &PyArray_Type, &pydec,
                                 &PyArray_Type, &pypmr,
                                 &PyArray_Type, &pypmd,
                                 &PyArray_Type, &pypx,
                                 &PyArray_Type, &pyrv)) {
        return NULL;
    }
    ara = PyArray_FROM_OTF(pyra, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adec = PyArray_FROM_OTF(pydec, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apmr = PyArray_FROM_OTF(pypmr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apmd = PyArray_FROM_OTF(pypmd, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apx = PyArray_FROM_OTF(pypx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arv = PyArray_FROM_OTF(pyrv, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ara == NULL || adec == NULL || apmr == NULL ||
        apmd == NULL || apx == NULL || arv == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ara);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ara);
    if (dims[0] != PyArray_DIMS(adec)[0] ||
        dims[0] != PyArray_DIMS(apmr)[0] || dims[0] != PyArray_DIMS(apmd)[0] ||
        dims[0] != PyArray_DIMS(apx)[0] || dims[0] != PyArray_DIMS(arv)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    ra = (double *)PyArray_DATA(ara);
    dec = (double *)PyArray_DATA(adec);
    pmr = (double *)PyArray_DATA(apmr);
    pmd = (double *)PyArray_DATA(apmd);
    px = (double *)PyArray_DATA(apx);
    rv = (double *)PyArray_DATA(arv);
    dim_out[0] = dims[0];
    dim_out[1] = 2;
    dim_out[2] = 3;
    pypv = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pypv) {
        goto fail;
    }
    pv_iter = PyArray_IterNew((PyObject*)pypv);
    if (pv_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        status = eraStarpv(ra[i], dec[i], pmr[i], pmd[i], px[i], rv[i], pv);
        if (status) {
            if (status == 1) {
                PyErr_SetString(_erfaError, "distance overriden, extremely small (or zero or negative) parallax");
                goto fail;
            }
            else if (status == 2) {
                PyErr_SetString(_erfaError, "excessive velocity");
                goto fail;
            }
            else if (status == 4) {
                PyErr_SetString(_erfaError, "solution didn't converge");
                goto fail;
            }
            else {
                PyErr_SetString(_erfaError, "binary logical OR of other error above");
                goto fail;
            }
        }
        else {
            int j,k;
            for (j=0;j<2;j++) {
                for (k=0;k<3;k++) {
                    if (PyArray_SETITEM(pypv, PyArray_ITER_DATA(pv_iter), PyFloat_FromDouble(pv[j][k]))) {
                        PyErr_SetString(_erfaError, "unable to set pv");
                        goto fail;
                    }
                    PyArray_ITER_NEXT(pv_iter);
                }
            }
        }
    }
    Py_DECREF(ara);
    Py_DECREF(adec);
    Py_DECREF(apmr);
    Py_DECREF(apmd);
    Py_DECREF(apx);
    Py_DECREF(arv);
    Py_DECREF(pv_iter);
    Py_INCREF(pypv);
    return PyArray_Return(pypv);

fail:
    Py_XDECREF(ara);
    Py_XDECREF(adec);
    Py_XDECREF(apmr);
    Py_XDECREF(apmd);
    Py_XDECREF(apx);
    Py_XDECREF(arv);
    Py_XDECREF(pypv);
    Py_XDECREF(pv_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_starpv_doc,
"\nstarpv(ra1, dec1, pmr1, pmd1, px1, rv1) -> pv\n\n"
"Convert star catalog coordinates to position+velocity vector.\n"
"Given:\n"
"   ra      right ascension (radians)\n"
"   dec     declination (radians)\n"
"   pmr     RA proper motion (radians/year)\n"
"   pmd     Dec proper motion (radians/year)\n"
"   px      parallax (arcseconds)\n"
"   rv  radial velocity (km/s, positive = receding)\n"
"Returned:\n"
"   pv      pv-vector (AU, AU/day)");

static PyObject *
_erfa_starpm(PyObject *self, PyObject *args)
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
        j = eraStarpm(ra1[i], dec1[i], pmr1[i], pmd1[i], px1[i],
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

PyDoc_STRVAR(_erfa_starpm_doc,
"\nstarpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b) -> ra2, dec2, pmr2, pmd2, px2, rv2\n\n"
"Star proper motion:  update star catalog data for space motion.\n"
"Given:\n"
"   ra1     right ascension (radians), before\n"
"   dec1    declination (radians), before\n"
"   pmr1    RA proper motion (radians/year), before\n"
"   pmd1    Dec proper motion (radians/year), before\n"
"   px1     parallax (arcseconds), before\n"
"   rv1     radial velocity (km/s, +ve = receding), before\n"
"   ep1a    ''before'' epoch, part A \n"
"   ep1b    ''before'' epoch, part B\n"
"   ep2a    ''after'' epoch, part A\n"
"   ep2b    ''after'' epoch, part B\n"
"Returned:\n"
"   ra2     right ascension (radians), after\n"
"   dec2    declination (radians), after\n"
"   pmr2    RA proper motion (radians/year), after\n"
"   pmd2    Dec proper motion (radians/year), after\n"
"   px2     parallax (arcseconds), after\n"
"   rv2     radial velocity (km/s, +ve = receding), after");

static PyObject *
_erfa_bi00(PyObject *self)
{
    double dpsibi, depsbi, dra;
    eraBi00(&dpsibi, &depsbi, &dra);
    if (!PyErr_Occurred()) {
        return Py_BuildValue("ddd", dpsibi, depsbi, dra);
    }
    else {
        return NULL;
    }
}

PyDoc_STRVAR(_erfa_bi00_doc,
"\nbi00() -> dpsibi,depsbi,dra\n"
"Frame bias components of IAU 2000 precession-nutation models (part of MHB2000 with additions).\n"
"Returned:\n"
"    dpsibi,depsbi    obliquity and correction\n"
"    dra              the ICRS RA of the J2000.0 mean equinox");

static PyObject *
_erfa_bp00(PyObject *self, PyObject *args)
{
    double *d1, *d2, rb[3][3], rp[3][3], rbp[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL;
    PyObject *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrb = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrbp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrb || NULL == pyrp || NULL == pyrbp) {
        goto fail;
    }
    rb_iter = PyArray_IterNew((PyObject*)pyrb);
    if (rb_iter == NULL) goto fail;
    rp_iter = PyArray_IterNew((PyObject*)pyrp);
    if (rp_iter == NULL) goto fail;
    rbp_iter = PyArray_IterNew((PyObject*)pyrbp);
    if (rbp_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraBp00(d1[i], d2[i], rb, rp, rbp);
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
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    return Py_BuildValue("OOO", pyrb, pyrp, pyrbp);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_bp00_doc,
"\nbp00(d1, d2) -> rb, rp, rbp\n\n"
"Frame bias and precession, IAU 2000.\n"
"Given:\n"
"    d1, d2     TT as a 2-part Julian Date\n"
"Returned:\n"
"    rb         frame bias matrix\n"
"    rp         precession matrix\n"
"    rbp        bias-precession matrix");

static PyObject *
_erfa_bp06(PyObject *self, PyObject *args)
{
    double *d1, *d2, rb[3][3], rp[3][3], rbp[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL;
    PyObject *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrb = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pyrbp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrb || NULL == pyrp || NULL == pyrbp) {
        goto fail;
    }
    rb_iter = PyArray_IterNew((PyObject*)pyrb);
    if (rb_iter == NULL) goto fail;
    rp_iter = PyArray_IterNew((PyObject*)pyrp);
    if (rp_iter == NULL) goto fail;
    rbp_iter = PyArray_IterNew((PyObject*)pyrbp);
    if (rbp_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraBp06(d1[i], d2[i], rb, rp, rbp);
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
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    return Py_BuildValue("OOO", pyrb, pyrp, pyrbp);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_bp06_doc,
"\nbp06(d1, d2) -> rb, rp, rbp\n\n"
"Frame bias and precession, IAU 2006.\n"
"Given:\n"
"    d1, d2     TT as 2-part Julian Date\n"
"Returned:\n"
"    rb         frame bias matrix\n"
"    p          precession matrix)\n"
"    rbp        bias-precession matrix");

static PyObject *
_erfa_bpn2xy(PyObject *self, PyObject *args)
{
    double x, y, *ax, *ay, rbpn[3][3];
    PyObject *pyrbpn;
    PyArrayObject *pyx = NULL, *pyy = NULL;
    PyObject *arbpn = NULL;
    PyObject *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyrbpn))      
        return NULL;
    ndim = PyArray_NDIM(pyrbpn);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyrbpn);
    dim_out[0] = dims[0];
    pyx = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyy = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyx || NULL == pyy) goto fail;
    ax = (double *)PyArray_DATA(pyx);
    ay = (double *)PyArray_DATA(pyy);
    rbpn_iter = PyArray_IterNew((PyObject *)pyrbpn);
    if (rbpn_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k;
        double vrbpn;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arbpn = PyArray_GETITEM(pyrbpn, PyArray_ITER_DATA(rbpn_iter));
                if (arbpn == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arbpn);
                vrbpn = (double)PyFloat_AsDouble(arbpn);
                if (vrbpn == -1 && PyErr_Occurred()) goto fail;
                rbpn[j][k] = vrbpn;
                Py_DECREF(arbpn);                
                PyArray_ITER_NEXT(rbpn_iter); 
            }
        }
        eraBpn2xy(rbpn, &x, &y);
        ax[i] = x;
        ay[i] = y;
    }
    Py_DECREF(rbpn_iter);
    Py_INCREF(pyx);
    Py_INCREF(pyy);
    return Py_BuildValue("OO", pyx, pyy);

fail:
    Py_XDECREF(arbpn);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyx);
    Py_XDECREF(pyy);
    return NULL;
}

PyDoc_STRVAR(_erfa_bpn2xy_doc,
"\nbpn2xy(rbpn) -> x, y\n\n"
"Extract from the bias-precession-nutation matrix\n"
"the X,Y coordinates of the Celestial Intermediate Pole.\n"
"Given:\n"
"    rbpn       celestial-to-true matrix\n"
"Returned:\n"
"    x,y        celestial Intermediate Pole");

static PyObject *
_erfa_c2i00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rc2i[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrc2i = NULL;
    PyObject *rc2i_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2i = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2i) {
        goto fail;
    }
    rc2i_iter = PyArray_IterNew((PyObject*)pyrc2i);
    if (rc2i_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraC2i00a(d1[i], d2[i], rc2i);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2i_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rc2i_iter);
    Py_INCREF(pyrc2i);
    return (PyObject *)pyrc2i;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rc2i_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2i00a_doc,
"\nc2i00a(d1, d2) -> rc2i\n\n"
"Form the celestial-to-intermediate matrix\n"
"for a given date using the IAU 2000A precession-nutation model.\n"
"Given:\n"
"    d1, d2     TT as 2-part Julian date\n"
"Returned:\n"
"    rc2i       celestial-to-intermediate matrix\n");

static PyObject *
_erfa_c2i00b(PyObject *self, PyObject *args)
{
    double *d1, *d2, rc2i[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrc2i = NULL;
    PyObject *rc2i_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2i = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2i) {
        goto fail;
    }
    rc2i_iter = PyArray_IterNew((PyObject*)pyrc2i);
    if (rc2i_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraC2i00b(d1[i], d2[i], rc2i);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2i_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rc2i_iter);
    Py_INCREF(pyrc2i);
    return (PyObject *)pyrc2i;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rc2i_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2i00b_doc,
"\nc2i00b(d1, d2) -> rc2i\n\n"
"Form the celestial-to-intermediate matrix for a given\n"
"date using the IAU 2000B precession-nutation model.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian date\n"
"Returned:\n"
"    rc2i       celestial-to-intermediate matrix");

static PyObject *
_erfa_c2i06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rc2i[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrc2i = NULL;
    PyObject *rc2i_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2i = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2i) {
        goto fail;
    }
    rc2i_iter = PyArray_IterNew((PyObject*)pyrc2i);
    if (rc2i_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraC2i06a(d1[i], d2[i], rc2i);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2i_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rc2i_iter);
    Py_INCREF(pyrc2i);
    return (PyObject *)pyrc2i;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rc2i_iter);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2i06a_doc,
"\nc2i06a(d1, d2) -> rc2i\n\n"
"Form the celestial-to-intermediate matrix for a given date \n"
"using the IAU 2006 precession and IAU 200A nutation model.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian date\n"
"Returned:\n"
"    rc2i       celestial-to-intermediate matrix");

static PyObject *
_erfa_c2ibpn(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbpn[3][3], rc2i[3][3];
    PyObject *pyd1, *pyd2,  *pyrbpn;
    PyObject *ad1, *ad2, *arbpn = NULL;
    PyArrayObject *pyrc2i = NULL;
    PyObject *rbpn_iter = NULL;
    PyObject *rc2i_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pyrbpn))      
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
    rbpn_iter = PyArray_IterNew((PyObject *)pyrbpn);
    if (rbpn_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2i= (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2i) goto fail;
    rc2i_iter = PyArray_IterNew((PyObject*)pyrc2i);
    if (rc2i_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vrbpn;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arbpn = PyArray_GETITEM(pyrbpn, PyArray_ITER_DATA(rbpn_iter));
                if (arbpn == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arbpn);
                vrbpn = (double)PyFloat_AsDouble(arbpn);
                if (vrbpn == -1 && PyErr_Occurred()) goto fail;
                rbpn[j][k] = vrbpn;
                Py_DECREF(arbpn);                
                PyArray_ITER_NEXT(rbpn_iter); 
            }
        }
        eraC2ibpn(d1[i], d2[i], rbpn, rc2i);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2i_iter);
            }
        }
    }
    Py_DECREF(rc2i_iter);
    Py_DECREF(rbpn_iter);
    Py_INCREF(pyrc2i);
    return (PyObject *)pyrc2i;

fail:
    Py_XDECREF(arbpn);
    Py_XDECREF(rc2i_iter);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrc2i);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2ibpn_doc,
"\nc2ibpn(d1, d2, rbpn) -> rc2i\n\n"
"Form the celestial-to-intermediate matrix for a given date\n"
"and given the bias-precession-nutation matrix. IAU 2000.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian date\n"
"    rbpn       celestial-to-true matrix\n"
"Returned:\n"
"    rc2i       celestial-to-intermediate matrix");

static PyObject *
_erfa_c2ixy(PyObject *self, PyObject *args)
{
    double *d1, *d2, *x, *y, rc2i[3][3];
    PyObject *pyd1, *pyd2, *pyx, *pyy;
    PyObject *ad1, *ad2, *ax, *ay;
    PyArrayObject *pyrc2i = NULL;
    PyObject *rc2i_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pyx,
                                 &PyArray_Type, &pyy))      
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
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2i= (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2i) goto fail;
    rc2i_iter = PyArray_IterNew((PyObject*)pyrc2i);
    if (rc2i_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2ixy(d1[i] ,d2[i] ,x[i] ,y[i] ,rc2i);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter), PyFloat_FromDouble(rc2i[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2i");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2i_iter);
            }
        }
    }
    Py_DECREF(rc2i_iter);
    Py_INCREF(pyrc2i);
    return (PyObject *)pyrc2i;

fail:
    Py_XDECREF(rc2i_iter);
    Py_XDECREF(pyrc2i);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2ixy_doc,
"\nc2ixy(d1, d2, x, y) -> rc2i\n\n"
"Form the celestial to intermediate-frame-of-date matrix for a given\n"
"date when the CIP X,Y coordinates are known. IAU 2000.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"    x, y       Celestial Intermediate Pole\n"
"Returned:\n"
"    rc2i       celestial-to-intermediate matrix");

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
_erfa_c2t00a(PyObject *self, PyObject *args)
{
    double *tta, *ttb, *uta, *utb, *xp, *yp, rc2t[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pyxp, *pyyp;
    PyObject *auta, *autb, *atta, *attb, *axp, *ayp;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp))
        return NULL;
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL ||
        atta == NULL || attb == NULL ||
        axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atta);
    if (dims[0] != PyArray_DIMS(attb)[0] ||
        dims[0] != PyArray_DIMS(auta)[0] ||
        dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] ||
        dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2t00a(tta[i] ,ttb[i], uta[i], utb[i], xp[i], yp[i], rc2t);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2t00a_doc,
"\nc2t00a(tta,ttb,uta,utb,xp,yp) -> rc2t\n\n"
"Form the celestial to terrestrial matrix given the date, the UT1 and\n"
"the polar motion, using the IAU 2000A nutation model.\n"
"Given:\n"
"    tta,ttb    TT as 2-part Julian Date\n"
"    uta,utb    UT1 as 2-part Julian Date\n"
"    xp, yp     coordinates of the pole (radians)\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2t00b(PyObject *self, PyObject *args)
{
    double *tta, *ttb, *uta, *utb, *xp, *yp, rc2t[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pyxp, *pyyp;
    PyObject *auta, *autb, *atta, *attb, *axp, *ayp;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp))
        return NULL;
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL ||
        atta == NULL || attb == NULL ||
        axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atta);
    if (dims[0] != PyArray_DIMS(attb)[0] ||
        dims[0] != PyArray_DIMS(auta)[0] ||
        dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] ||
        dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2t00b(tta[i] ,ttb[i], uta[i], utb[i], xp[i], yp[i], rc2t);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2t00b_doc,
"\nc2t00b(tta,ttb,uta,utb,xp,yp) -> rc2t\n\n"
"Form the celestial to terrestrial matrix given the date, the UT1 and\n"
"the polar motion, using the IAU 2000B nutation model.\n"
"Given:\n"
"    tta,ttb    TT as 2-part Julian Date\n"
"    uta,utb    UT1 as 2-part Julian Date\n"
"    xp, yp     coordinates of the pole (radians)\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2t06a(PyObject *self, PyObject *args)
{
    double *tta, *ttb, *uta, *utb, *xp, *yp, rc2t[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pyxp, *pyyp;
    PyObject *auta, *autb, *atta, *attb, *axp, *ayp;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp))
        return NULL;
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL ||
        atta == NULL || attb == NULL ||
        axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atta);
    if (dims[0] != PyArray_DIMS(attb)[0] ||
        dims[0] != PyArray_DIMS(auta)[0] ||
        dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] ||
        dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2t06a(tta[i] ,ttb[i], uta[i], utb[i], xp[i], yp[i], rc2t);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2t06a_doc,
"\nc2t06a(tta,ttb,uta,utb,xp,yp) -> rc2t\n\n"
"Form the celestial to terrestrial matrix given the date, the UT1 and\n"
"the polar motion, using the IAU 2006 precession and IAU 2000A nutation models.\n"
"Given:\n"
"    tta,ttb    TT as 2-part Julian Date\n"
"    uta,utb    UT1 as 2-part Julian Date\n"
"    xp, yp     coordinates of the pole (radians)\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2tcio(PyObject *self, PyObject *args)
{
    double rc2i[3][3], *era, rpom[3][3], rc2t[3][3];
    PyObject *pyrc2i, *pyera, *pyrpom;
    PyObject *aera, *arc2i = NULL, *arpom = NULL;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2i_iter = NULL, *rpom_iter = NULL, *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                                &PyArray_Type, &pyrc2i,
                                &PyArray_Type, &pyera,
                                &PyArray_Type, &pyrpom))
        return NULL;
    ndim = PyArray_NDIM(pyrc2i);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyrc2i);
    if (dims[0] != PyArray_DIMS(pyera)[0] || dims[0] != PyArray_DIMS(pyrpom)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    aera = PyArray_FROM_OTF(pyera, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aera == NULL) goto fail;
    era = (double *)PyArray_DATA(aera);
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2i_iter = PyArray_IterNew((PyObject *)pyrc2i);
    rpom_iter = PyArray_IterNew((PyObject *)pyrpom);
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2i_iter == NULL || rpom_iter == NULL || rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vrc2i, vrpom;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arc2i = PyArray_GETITEM(pyrc2i, PyArray_ITER_DATA(rc2i_iter));
                if (arc2i == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arc2i);
                vrc2i = (double)PyFloat_AsDouble(arc2i);
                if (vrc2i == -1 && PyErr_Occurred()) goto fail;
                rc2i[j][k] = vrc2i;
                Py_DECREF(arc2i);                
                PyArray_ITER_NEXT(rc2i_iter); 
                arpom = PyArray_GETITEM(pyrpom, PyArray_ITER_DATA(rpom_iter));
                if (arpom == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arpom);
                vrpom = (double)PyFloat_AsDouble(arpom);
                if (vrpom == -1 && PyErr_Occurred()) goto fail;
                rpom[j][k] = vrpom;
                Py_DECREF(arpom);                
                PyArray_ITER_NEXT(rpom_iter); 
            }
        }
        eraC2tcio(rc2i, era[i], rpom, rc2t);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(arc2i);
    Py_DECREF(arpom);
    Py_DECREF(rc2i_iter);
    Py_DECREF(rpom_iter);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(arc2i);
    Py_XDECREF(arpom);
    Py_XDECREF(rc2i_iter);
    Py_XDECREF(rpom_iter);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2tcio_doc,
"\nc2tcio(rc2i, era, rpom) -> rc2t\n\n"
"Assemble the celestial to terrestrial matrix from CIO-based components\n"
"(the celestial-to-intermediate matrix, the Earth Rotation Angle and the polar motion matrix)\n"
"Given:\n"
"    rc2i       celestial-to-intermediate matrix\n"
"    era        Earth rotation angle\n"
"    rpom       polar-motion matrix\n"
"Returned:t\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2teqx(PyObject *self, PyObject *args)
{
    double rbpn[3][3], *gst, rpom[3][3], rc2t[3][3];
    PyObject *pyrbpn, *pygst, *pyrpom;
    PyObject *agst, *arbpn = NULL, *arpom = NULL;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rbpn_iter = NULL, *rpom_iter = NULL, *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                                &PyArray_Type, &pyrbpn,
                                &PyArray_Type, &pygst,
                                &PyArray_Type, &pyrpom))
        return NULL;
    ndim = PyArray_NDIM(pyrbpn);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyrbpn);
    if (dims[0] != PyArray_DIMS(pygst)[0] || dims[0] != PyArray_DIMS(pyrpom)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    agst = PyArray_FROM_OTF(pygst, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (agst == NULL) goto fail;
    gst = (double *)PyArray_DATA(agst);
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rbpn_iter = PyArray_IterNew((PyObject *)pyrbpn);
    rpom_iter = PyArray_IterNew((PyObject *)pyrpom);
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rbpn_iter == NULL || rpom_iter == NULL || rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create itgsttors");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        int j,k;
        double vrbpn, vrpom;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arbpn = PyArray_GETITEM(pyrbpn, PyArray_ITER_DATA(rbpn_iter));
                if (arbpn == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arbpn);
                vrbpn = (double)PyFloat_AsDouble(arbpn);
                if (vrbpn == -1 && PyErr_Occurred()) goto fail;
                rbpn[j][k] = vrbpn;
                Py_DECREF(arbpn);
                PyArray_ITER_NEXT(rbpn_iter);
                arpom = PyArray_GETITEM(pyrpom, PyArray_ITER_DATA(rpom_iter));
                if (arpom == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arpom);
                vrpom = (double)PyFloat_AsDouble(arpom);
                if (vrpom == -1 && PyErr_Occurred()) goto fail;
                rpom[j][k] = vrpom;
                Py_DECREF(arpom);
                PyArray_ITER_NEXT(rpom_iter);
            }
        }
        eraC2teqx(rbpn, gst[i], rpom, rc2t);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(arbpn);
    Py_DECREF(arpom);
    Py_DECREF(rbpn_iter);
    Py_DECREF(rpom_iter);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(arbpn);
    Py_XDECREF(arpom);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(rpom_iter);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2teqx_doc,
"\nc2teqx(rbpn, gst, rpom -> rc2t\n\n"
"Assemble the celestial to terrestrial matrix from equinox-based\n"
"components (the celestial-to-true matrix, the Greenwich Apparent\n"
"Sidereal Time and the polar motion matrix).\n"
"Given:\n"
"    rbpn       celestial-to-true matrix\n"
"    gst        Greenwich (apparent) Sidereal Time\n"
"    rpom       polar-motion matrix\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2tpe(PyObject *self, PyObject *args)
{
    double *tta, *ttb, *uta, *utb, *dpsi, *deps, *xp, *yp, rc2t[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pydpsi, *pydeps, *pyxp, *pyyp;
    PyObject *auta, *autb, *atta, *attb, *adpsi, *adeps, *axp, *ayp;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!", 
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pydpsi,
                                 &PyArray_Type, &pydeps,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp))
        return NULL;
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adpsi = PyArray_FROM_OTF(pydpsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adeps = PyArray_FROM_OTF(pydeps, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL || atta == NULL || attb == NULL ||
        adpsi == NULL || adeps == NULL || axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atta);
    if (dims[0] != PyArray_DIMS(attb)[0] ||
        dims[0] != PyArray_DIMS(auta)[0] || dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(adpsi)[0] || dims[0] != PyArray_DIMS(adeps)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] || dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    dpsi = (double *)PyArray_DATA(adpsi);
    deps = (double *)PyArray_DATA(adeps);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2tpe(tta[i], ttb[i], uta[i], utb[i], dpsi[i], deps[i], xp[i], yp[i], rc2t);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(adpsi);
    Py_DECREF(adeps);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(adpsi);
    Py_XDECREF(adeps);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2tpe_doc,
"\nc2tpe(tta,ttb,uta,utb,dpsi,deps,xp,yp) -> rc2t\n\n"
"Form the celestial to terrestrial matrix given the date, the UT1,\n"
"the nutation and the polar motion.  IAU 2000.\n"
"Given:\n"
"    tta,ttb    TT as 2-part Julian Date\n"
"    uta,utb    UT1 as 2-part Julian Date\n"
"    dpsi,deps  nutation\n"
"    xp,yp      coordinates of the pole\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_c2txy(PyObject *self, PyObject *args)
{
    double *tta, *ttb, *uta, *utb, *x, *y, *xp, *yp, rc2t[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pyx, *pyy, *pyxp, *pyyp;
    PyObject *auta, *autb, *atta, *attb, *ax, *ay, *axp, *ayp;
    PyArrayObject *pyrc2t = NULL;
    PyObject *rc2t_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!", 
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pyx,
                                 &PyArray_Type, &pyy,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp))
        return NULL;
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ax = PyArray_FROM_OTF(pyx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ay = PyArray_FROM_OTF(pyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL || atta == NULL || attb == NULL ||
        ax == NULL || ay == NULL || axp == NULL || ayp == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atta);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atta);
    if (dims[0] != PyArray_DIMS(attb)[0] ||
        dims[0] != PyArray_DIMS(auta)[0] || dims[0] != PyArray_DIMS(autb)[0] ||
        dims[0] != PyArray_DIMS(ax)[0] || dims[0] != PyArray_DIMS(ay)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] || dims[0] != PyArray_DIMS(ayp)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    uta = (double *)PyArray_DATA(auta);
    utb = (double *)PyArray_DATA(autb);
    tta = (double *)PyArray_DATA(atta);
    ttb = (double *)PyArray_DATA(attb);
    x = (double *)PyArray_DATA(ax);
    y = (double *)PyArray_DATA(ay);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrc2t = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrc2t) goto fail;
    rc2t_iter = PyArray_IterNew((PyObject*)pyrc2t);
    if (rc2t_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }

    for (i=0;i<dims[0];i++) {
        eraC2txy(tta[i], ttb[i], uta[i], utb[i], x[i], y[i], xp[i], yp[i], rc2t);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyrc2t, PyArray_ITER_DATA(rc2t_iter), PyFloat_FromDouble(rc2t[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rc2t");
                    goto fail;
                }
                PyArray_ITER_NEXT(rc2t_iter);
            }
        }
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(rc2t_iter);
    Py_INCREF(pyrc2t);
    return (PyObject *)pyrc2t;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(rc2t_iter);
    Py_XDECREF(pyrc2t);
    return NULL;
}

PyDoc_STRVAR(_erfa_c2txy_doc,
"\nc2txy(tta,ttb,uta,utb,x,y,xp,yp) -> rc2t\n\n"
"Form the celestial to terrestrial matrix given the date, the UT1,\n"
"the CIP coordinates and the polar motion.  IAU 2000."
"Given:\n"
"    tta,ttb    TT as 2-part Julian Date\n"
"    uta,utb    UT1 as 2-part Julian Date\n"
"    x,y        Celestial Intermediate Pole\n"
"    xp,yp      coordinates of the pole\n"
"Returned:\n"
"    rc2t       celestial-to-terrestrial matrix");

static PyObject *
_erfa_eo06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, *eo;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyeo = NULL;
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
    pyeo = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyeo) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    eo = (double *)PyArray_DATA(pyeo);
    for (i=0;i<dims[0];i++) {
        eo[i] = eraEo06a(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyeo);
    return (PyObject *)pyeo;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyeo);
    return NULL;
}

PyDoc_STRVAR(_erfa_eo06a_doc,
"\neo06a(d1, d2) -> eo\n"
"equation of the origins, IAU 2006 precession and IAU 2000A nutation.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"Returned:\n"
"    eo         equation of the origins (radians)");

static PyObject *
_erfa_eors(PyObject *self, PyObject *args)
{
    double rnpb[3][3], *s, *eo;
    PyObject *pys, *pyrnpb;
    PyArrayObject *pyeo = NULL;
    PyObject *as = NULL, *arnpb = NULL;
    PyObject *rnpb_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyrnpb,
                                 &PyArray_Type, &pys))      
        return NULL;
    ndim = PyArray_NDIM(pyrnpb);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyrnpb);
    as = PyArray_FROM_OTF(pys, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (as == NULL) {
        goto fail;
    }
    if (dims[0] != PyArray_DIMS(as)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    dim_out[0] = dims[0];
    pyeo = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyeo) goto fail;
    eo = (double *)PyArray_DATA(pyeo);
    s = (double *)PyArray_DATA(as);
    rnpb_iter = PyArray_IterNew((PyObject *)pyrnpb);
    if (rnpb_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        int j,k;
        double vrnpb;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arnpb = PyArray_GETITEM(pyrnpb, PyArray_ITER_DATA(rnpb_iter));
                if (arnpb == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arnpb);
                vrnpb = (double)PyFloat_AsDouble(arnpb);
                if (vrnpb == -1 && PyErr_Occurred()) goto fail;
                rnpb[j][k] = vrnpb;
                Py_DECREF(arnpb);                
                PyArray_ITER_NEXT(rnpb_iter); 
            }
        }
        eo[i] = eraEors(rnpb, s[i]);
    }
    Py_DECREF(as);
    Py_DECREF(rnpb_iter);
    Py_INCREF(pyeo);
    return (PyObject *)pyeo;

fail:
    Py_XDECREF(as);
    Py_XDECREF(arnpb);
    Py_XDECREF(rnpb_iter);
    Py_XDECREF(pyeo);
    return NULL;
}

PyDoc_STRVAR(_erfa_eors_doc,
"\neors(rnpb, s) -> eo\n\n"
"Equation of the origins, given the classical NPB matrix and the quantity s\n"
"Given:\n"
"    rnpb       classical nutation x precession x bias matrix\n"
"    s          CIO locator\n"
"Returned:\n"
"    eo         equation of the origins in radians");

static PyObject *
_erfa_fw2m(PyObject *self, PyObject *args)
{
    double *gamb, *phib, *psi, *eps, r[3][3];
    PyObject *pygamb, *pyphib, *pypsi, *pyeps;
    PyObject *agamb = NULL, *aphib = NULL, *apsi = NULL, *aeps = NULL;
    PyArrayObject *pyr = NULL;
    PyObject *r_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pygamb,
                                 &PyArray_Type, &pyphib,
                                 &PyArray_Type, &pypsi,
                                 &PyArray_Type, &pyeps))      
        return NULL;
    agamb = PyArray_FROM_OTF(pygamb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphib = PyArray_FROM_OTF(pyphib, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apsi = PyArray_FROM_OTF(pypsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aeps = PyArray_FROM_OTF(pyeps, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (agamb == NULL || aphib == NULL || apsi == NULL || aeps == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(agamb);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(agamb);
    if (dims[0] != PyArray_DIMS(aphib)[0] ||
        dims[0] != PyArray_DIMS(apsi)[0] ||
        dims[0] != PyArray_DIMS(aeps)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyr = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyr) goto fail;
    r_iter = PyArray_IterNew((PyObject *)pyr);
    if (r_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    gamb = (double *)PyArray_DATA(agamb);
    phib = (double *)PyArray_DATA(aphib);
    psi = (double *)PyArray_DATA(apsi);
    eps = (double *)PyArray_DATA(aeps);

    for (i=0;i<dims[0];i++) {
        eraFw2m(gamb[i], phib[i], psi[i], eps[i], r);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pyr, PyArray_ITER_DATA(r_iter), PyFloat_FromDouble(r[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set output r");
                    goto fail;
                }
                PyArray_ITER_NEXT(r_iter);
            }
        }
    }
    Py_DECREF(agamb);
    Py_DECREF(aphib);
    Py_DECREF(apsi);
    Py_DECREF(aeps);
    Py_DECREF(r_iter);
    Py_INCREF(pyr);
    return (PyObject *)pyr;    

fail:
    Py_XDECREF(agamb);
    Py_XDECREF(aphib);
    Py_XDECREF(apsi);
    Py_XDECREF(aeps);
    Py_XDECREF(r_iter);
    Py_XDECREF(pyr);
    return NULL;
}

PyDoc_STRVAR(_erfa_fw2m_doc,
"\nfw2m(gamb, phib, psi, eps) -> r\n\n"
"Form rotation matrix given the Fukushima-Williams angles.\n"
"Given:\n"
"    gamb       F-W angle gamma_bar (radians)\n"
"    phib       F-W angle phi_bar (radians)\n"
"    si         F-W angle psi (radians)\n"
"    eps        F-W angle epsilon (radians)\n"
"Returned:\n"
"    r          rotation matrix");

static PyObject *
_erfa_fw2xy(PyObject *self, PyObject *args)
{
    double *gamb, *phib, *psi, *eps, *x, *y;
    PyObject *pygamb, *pyphib, *pypsi, *pyeps;
    PyObject *agamb = NULL, *aphib = NULL, *apsi = NULL, *aeps = NULL;
    PyArrayObject *pyx = NULL, *pyy = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pygamb,
                                 &PyArray_Type, &pyphib,
                                 &PyArray_Type, &pypsi,
                                 &PyArray_Type, &pyeps))      
        return NULL;
    agamb = PyArray_FROM_OTF(pygamb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphib = PyArray_FROM_OTF(pyphib, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apsi = PyArray_FROM_OTF(pypsi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aeps = PyArray_FROM_OTF(pyeps, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (agamb == NULL || aphib == NULL || apsi == NULL || aeps == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(agamb);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(agamb);
    if (dims[0] != PyArray_DIMS(aphib)[0] ||
        dims[0] != PyArray_DIMS(apsi)[0] ||
        dims[0] != PyArray_DIMS(aeps)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyx = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyy = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyx || NULL == pyy) {
        goto fail;
    }
    gamb = (double *)PyArray_DATA(agamb);
    phib = (double *)PyArray_DATA(aphib);
    psi = (double *)PyArray_DATA(apsi);
    eps = (double *)PyArray_DATA(aeps);
    x = (double *)PyArray_DATA(pyx);
    y = (double *)PyArray_DATA(pyy);

    for (i=0;i<dims[0];i++) {
        eraFw2xy(gamb[i], phib[i], psi[i], eps[i], &x[i], &y[i]);
    }
    Py_DECREF(agamb);
    Py_DECREF(aphib);
    Py_DECREF(apsi);
    Py_DECREF(aeps);
    Py_INCREF(pyx);
    Py_INCREF(pyy);
    return Py_BuildValue("OO", pyx, pyy);    

fail:
    Py_XDECREF(agamb);
    Py_XDECREF(aphib);
    Py_XDECREF(apsi);
    Py_XDECREF(aeps);
    Py_XDECREF(pyx);
    Py_XDECREF(pyy);
    return NULL;
}

PyDoc_STRVAR(_erfa_fw2xy_doc,
"\nfw2xy(gamb, phib, psi, eps) -> x, y\n\n"
"Form CIP X,Y given Fukushima-Williams bias-precession-nutation angles.\n"
"Given:\n"
"    gamb       F-W angle gamma_bar (radians)\n"
"    phib       F-W angle phi_bar (radians)\n"
"    psi        F-W angle psi (radians)\n"
"    eps        F-W angle epsilon (radians)\n"
"Returned:\n"
"    x,y        CIP X,Y (radians)");

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
_erfa_dtdb(PyObject *self, PyObject *args)
{
    double *tdbtt, *d1, *d2, *ut1, *elon, *u, *v;
    PyObject *pyd1, *pyd2, *pyut1, *pyelon, *pyu, *pyv;
    PyObject *ad1, *ad2, *aut1, *aelon, *au, *av;
    PyArrayObject *pytdbtt = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2,
                                 &PyArray_Type, &pyut1,
                                 &PyArray_Type, &pyelon,
                                 &PyArray_Type, &pyu,
                                 &PyArray_Type, &pyv))
        return NULL;
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aut1 = PyArray_FROM_OTF(pyut1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aelon = PyArray_FROM_OTF(pyelon, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    au = PyArray_FROM_OTF(pyu, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    av = PyArray_FROM_OTF(pyv, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ad1 == NULL || ad2 == NULL || aut1 == NULL ||
        aelon == NULL || au == NULL || av == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ad1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ad1);
    if (dims[0] != PyArray_DIMS(ad2)[0] ||
        dims[0] != PyArray_DIMS(aut1)[0] ||
        dims[0] != PyArray_DIMS(aelon)[0] ||
        dims[0] != PyArray_DIMS(au)[0] ||
        dims[0] != PyArray_DIMS(av)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pytdbtt = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytdbtt) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ut1 = (double *)PyArray_DATA(aut1);
    elon = (double *)PyArray_DATA(aelon);
    u = (double *)PyArray_DATA(au);
    v = (double *)PyArray_DATA(av);
    tdbtt = (double *)PyArray_DATA(pytdbtt);
    for (i=0;i<dims[0];i++) {
        tdbtt[i] = eraDtdb(d1[i], d2[i], ut1[i], elon[i], u[i], v[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(aut1);
    Py_DECREF(aelon);
    Py_DECREF(au);
    Py_DECREF(av);
    Py_INCREF(pytdbtt);
    return (PyObject *)pytdbtt;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(aut1);
    Py_XDECREF(aelon);
    Py_XDECREF(au);
    Py_XDECREF(av);
    Py_XDECREF(pytdbtt);
    return NULL;
}

PyDoc_STRVAR(_erfa_dtdb_doc,
"\ndtdb(d1, d2, ut1, elon, u, v) -> TDB-TT\n\n"
"An approximation to TDB-TT, the difference between barycentric\n"
"dynamical time and terrestrial time, for an observer on the Earth.\n"
"Given:\n"
"    d1,d2      TDB date as 2-part Julian Date\n"
"    ut1        UT1 universal time as fraction of one day\n"
"    elong      longitude (east positive, radians)\n"
"    u          distance from Earth spin axis (km)\n"
"    v          distance north of equatorial plane (km)\n"
"Returned:\n"
"    tdbtt      TDB-TT (seconds)");

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
_erfa_ee00a(PyObject *self, PyObject *args)
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
    pyee = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyee) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ee = (double *)PyArray_DATA(pyee);
    for (i=0;i<dims[0];i++) {
        ee[i] = eraEe00a(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyee);
    return PyArray_Return(pyee);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyee);
    return NULL;
}

PyDoc_STRVAR(_erfa_ee00a_doc,
"\nee00a(d1,d2) -> ee\n\n"
"equation of the equinoxes, compatible with IAU 2000 resolutions.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"Returned:\n"
"    ee         equation of the equinoxes");

static PyObject *
_erfa_ee00b(PyObject *self, PyObject *args)
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
    pyee = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyee) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ee = (double *)PyArray_DATA(pyee);
    for (i=0;i<dims[0];i++) {
        ee[i] = eraEe00b(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyee);
    return PyArray_Return(pyee);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyee);
    return NULL;
}

PyDoc_STRVAR(_erfa_ee00b_doc,
"\nee00b(d1,d2) -> ee\n\n"
"Equation of the equinoxes, compatible with IAU 2000 resolutions\n"
"but using the truncated nutation model IAU 2000B.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"Returned:\n"
"    ee         equation of the equinoxes");

static PyObject *
_erfa_ee06a(PyObject *self, PyObject *args)
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
    pyee = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyee) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ee = (double *)PyArray_DATA(pyee);
    for (i=0;i<dims[0];i++) {
        ee[i] = eraEe06a(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyee);
    return PyArray_Return(pyee);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyee);
    return NULL;
}

PyDoc_STRVAR(_erfa_ee06a_doc,
"\nee06a(d1,d2) -> ee\n\n"
"Equation of the equinoxes, compatible with IAU 2000 resolutions and \n"
"IAU 2006/2000A precession-nutation.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"Returned:\n"
"    ee         equation of the equinoxes");

static PyObject *
_erfa_eect00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *ct;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyct = NULL;
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
    pyct = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyct) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    ct = (double *)PyArray_DATA(pyct);
    for (i=0;i<dims[0];i++) {
        ct[i] = eraEect00(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyct);
    return PyArray_Return(pyct);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyct);
    return NULL;
}

PyDoc_STRVAR(_erfa_eect00_doc,
"\neect00(d1,d2) -> ct\n\n"
"Equation of the equinoxes complementary terms, consistent with IAU 2000 resolutions.\n"
"Given:\n"
"    d1,d2      TT as 2-part Julian Date\n"
"Returned:\n"
"    ct        complementary terms");

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
_erfa_epv00(PyObject *self, PyObject *args)
{
    double *d1, *d2, pvh[2][3], pvb[2][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pypvh = NULL, *pypvb = NULL;
    PyObject *pvh_iter = NULL, *pvb_iter = NULL;
    int status;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 2;
    dim_out[2] = 3;
    pypvh = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    pypvb = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pypvh || NULL == pypvb) {
        goto fail;
    }
    pvh_iter = PyArray_IterNew((PyObject*)pypvh);
    pvb_iter = PyArray_IterNew((PyObject*)pypvb);
    if (pvh_iter == NULL || pvb_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        status = eraEpv00(d1[i], d2[i], pvh, pvb);
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        if (status) {
            PyErr_WarnEx(PyExc_Warning, "date outside the range 1900-2100 AD", 1);
        }
        PyErr_Restore(exc, val, tb);
        int j,k;
        for (j=0;j<2;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pypvh, PyArray_ITER_DATA(pvh_iter), PyFloat_FromDouble(pvh[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set pvh");
                    goto fail;
                }
                PyArray_ITER_NEXT(pvh_iter);
                if (PyArray_SETITEM(pypvb, PyArray_ITER_DATA(pvb_iter), PyFloat_FromDouble(pvb[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set pvb");
                    goto fail;
                }
                PyArray_ITER_NEXT(pvb_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(pvh_iter);
    Py_DECREF(pvb_iter);
    Py_INCREF(pypvh);
    Py_INCREF(pypvb);
    return Py_BuildValue("OO",pypvh, pypvb);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pvh_iter);
    Py_XDECREF(pvb_iter);
    Py_XDECREF(pypvh);
    Py_XDECREF(pypvb);
    return NULL;
}

PyDoc_STRVAR(_erfa_epv00_doc,
"\nepv00(d1,d2) -> pvh, pvb\n\n"
"Earth position and velocity, heliocentric and barycentric,\n"
"with respect to the Barycentric Celestial Reference System.\n"
"Given:\n"
"    d1,d2      TDB as 2-part Julian Date\n"
"Returned:\n"
"    pvh        heliocentric Earth position/velocity\n"
"    pvb        barycentric Earth position/velocity");

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
_erfa_fad03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFad03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fad03_doc,
"\nfad03(t) -> d\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean elongation of the Moon from the Sun.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    d          mean elongation of the Moon from the Sun, radians.");

static PyObject *
_erfa_fae03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFae03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fae03_doc,
"\nfae03(t) -> e\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Earth.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    e          mean longitude of Earth, radians.");

static PyObject *
_erfa_faf03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFaf03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_faf03_doc,
"\nfaf03(t) -> f\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of the Moon minus mean longitude of the ascending node.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    f          mean longitude of the Moon minus mean longitude of the ascending node, radians.");

static PyObject *
_erfa_faju03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFaju03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_faju03_doc,
"\nfaju03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Jupiter.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Jupiter., in radians.");

static PyObject *
_erfa_fal03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFal03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fal03_doc,
"\nfal03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean anomaly of the Moon.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean anomaly of the Moon, in radians.");

static PyObject *
_erfa_falp03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFalp03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_falp03_doc,
"\nfal03(t) -> lp\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean anomaly of the Sun.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned\n"
"    lp         mean anomaly of the Sun, in radians.");

static PyObject *
_erfa_fama03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFama03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fama03_doc,
"\nfama03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Mars.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Mars, in radians.");

static PyObject *
_erfa_fame03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFame03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fame03_doc,
"\nfame03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Mercury.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Mercury, in radians.");

static PyObject *
_erfa_fane03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFane03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fane03_doc,
"\nfane03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Neptune.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Neptune, in radians.");

static PyObject *
_erfa_faom03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFaom03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_faom03_doc,
"\nfaom03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Moon's ascending node.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Moon's ascending node, in radians.");

static PyObject *
_erfa_fapa03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFapa03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fapa03_doc,
"\nfapa03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"general accumulated precession in longitude.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          general accumulated precession in longitude, in radians.");

static PyObject *
_erfa_fasa03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFasa03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fasa03_doc,
"\nfasa03(t) -> l\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Saturn.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Saturn, in radians.");

static PyObject *
_erfa_faur03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFaur03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_faur03_doc,
"\nfaur03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003):\n"
"mean longitude of Uranus.\n"
"Given:\n"
"    t          TDB as Julian centuries since J2000.,\n"
"Returned:\n"
"    l          mean longitude of Uranus, in radians.");

static PyObject *
_erfa_fave03(PyObject *self, PyObject *args)
{
    double *t, *d;
    PyObject *pyt;
    PyObject *at;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", 
                                 &PyArray_Type, &pyt))
        return NULL;

    at = PyArray_FROM_OTF(pyt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (at == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(at);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(at);
    pyout = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyout) goto fail;
    t = (double *)PyArray_DATA(at);
    d = (double *)PyArray_DATA(pyout);
    for (i=0;i<dims[0];i++) {
        d[i] = eraFave03(t[i]);
    }
    Py_DECREF(at);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(at);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_fave03_doc,
"\nfaver03(t) -> l\n\n"
"Fundamental argument, IERS Conventions (2003)\n"
"mean longitude of Venus."
"Given:\n"
"    t          TDB as Julian centuries since J2000.0\n"
"Returned:\n"
"    l          mean longitude of Venus, in radians.");

static PyObject *
_erfa_fk52h(PyObject *self, PyObject *args)
{
    double *r5, *d5, *dr5, *dd5, *px5, *rv5;
    PyObject *pyr5, *pyd5, *pydr5, *pydd5, *pypx5, *pyrv5;
    PyObject *ar5, *ad5, *adr5, *add5, *apx5, *arv5;
    double *rh, *dh, *drh, *ddh, *pxh, *rvh;
    PyArrayObject *pyrh = NULL, *pydh = NULL, *pydrh = NULL, *pyddh = NULL, *pypxh = NULL, *pyrvh = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pyr5,
                                 &PyArray_Type, &pyd5,
                                 &PyArray_Type, &pydr5,
                                 &PyArray_Type, &pydd5,
                                 &PyArray_Type, &pypx5,
                                 &PyArray_Type, &pyrv5))
        return NULL;
    ar5 = PyArray_FROM_OTF(pyr5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad5 = PyArray_FROM_OTF(pyd5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adr5 = PyArray_FROM_OTF(pydr5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    add5 = PyArray_FROM_OTF(pydd5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apx5 = PyArray_FROM_OTF(pypx5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arv5 = PyArray_FROM_OTF(pyrv5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ar5 == NULL || ad5 == NULL || adr5 == NULL ||
        apx5 == NULL || add5 == NULL || arv5 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ar5);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ar5);
    if (dims[0] != PyArray_DIMS(ad5)[0] ||
        dims[0] != PyArray_DIMS(adr5)[0] || dims[0] != PyArray_DIMS(apx5)[0] ||
        dims[0] != PyArray_DIMS(add5)[0] || dims[0] != PyArray_DIMS(arv5)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyrh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydrh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyddh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypxh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyrvh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyrh || NULL == pydh || NULL == pydrh ||
        NULL == pyddh || NULL == pypxh || NULL == pyrvh) goto fail;
    r5 = (double *)PyArray_DATA(ar5);
    d5 = (double *)PyArray_DATA(ad5);
    dr5 = (double *)PyArray_DATA(adr5);
    dd5 = (double *)PyArray_DATA(add5);
    px5 = (double *)PyArray_DATA(apx5);
    rv5 = (double *)PyArray_DATA(arv5);
    rh = (double *)PyArray_DATA(pyrh);
    dh = (double *)PyArray_DATA(pydh);
    drh = (double *)PyArray_DATA(pydrh);
    ddh = (double *)PyArray_DATA(pyddh);
    pxh = (double *)PyArray_DATA(pypxh);
    rvh = (double *)PyArray_DATA(pyrvh);

    for (i=0;i<dims[0];i++) {
        eraFk52h(r5[i], d5[i], dr5[i], dd5[i], px5[i], rv5[i],
                 &rh[i], &dh[i], &drh[i], &ddh[i], &pxh[i], &rvh[i]);
    }
    Py_DECREF(ar5);
    Py_DECREF(ad5);
    Py_DECREF(adr5);
    Py_DECREF(add5);
    Py_DECREF(apx5);
    Py_DECREF(arv5);
    Py_INCREF(pyrh);
    Py_INCREF(pydh);
    Py_INCREF(pydrh);
    Py_INCREF(pyddh);
    Py_INCREF(pypxh);
    Py_INCREF(pyrvh);
    return Py_BuildValue("OOOOOO", pyrh, pydh, pydrh, pyddh, pypxh, pyrvh);

fail:
    Py_XDECREF(ar5);
    Py_XDECREF(ad5);
    Py_XDECREF(adr5);
    Py_XDECREF(add5);
    Py_XDECREF(apx5);
    Py_XDECREF(arv5);
    Py_XDECREF(pyrh);
    Py_XDECREF(pydh);
    Py_XDECREF(pydrh);
    Py_XDECREF(pyddh);
    Py_XDECREF(pypxh);
    Py_XDECREF(pyrvh);
    return NULL;
}

PyDoc_STRVAR(_erfa_fk52h_doc,
"\nfk52h(r5, d5, dr5, dd5, px5,rv5) -> rh, dh, drh, ddh, pxh, rvh)\n\n"
"Transform FK5 (J2000.0) star data into the Hipparcos system.\n"
"Given (all FK5, equinox J2000.0, epoch J2000.0):\n"
"    r5         RA (radians)\n"
"    d5         Dec (radians)\n"
"    dr5        proper motion in RA (dRA/dt, rad/Jyear)\n"
"    dd5        proper motion in Dec (dDec/dt, rad/Jyear)\n"
"    px5        parallax (arcsec)\n"
"    rv5        radial velocity (km/s, positive = receding)\n"
"Returned (all Hipparcos, epoch J2000.0):\n"
"    rh         RA (radians)\n"
"    dh         Dec (radians)\n"
"    drh        proper motion in RA (dRA/dt, rad/Jyear)\n"
"    ddh        proper motion in Dec (dDec/dt, rad/Jyear)\n"
"    pxh        parallax (arcsec)\n"
"    rvh        radial velocity (km/s, positive = receding)");

static PyObject *
_erfa_fk5hip(PyObject *self)
{
    double r5h[3][3], s5h[3];
    eraFk5hip(r5h, s5h);
    return Py_BuildValue("OO", _to_py_matrix(r5h), _to_py_vector(s5h));
}

PyDoc_STRVAR(_erfa_fk5hip_doc,
"\nfk5hip() -> r5h, s5h\n\n"
"FK5 to Hipparcos rotation and spin.\n"
"Returned:\n"
"    r5h        r-matrix: FK5 rotation wrt Hipparcos \n"
"    s5         r-vector: FK5 spin wrt Hipparcos.");

static PyObject *
_erfa_fk5hz(PyObject *self, PyObject *args)
{
    double *r5, *d5, *d1, *d2, *rh, *dh;
    PyObject *pyr5, *pyd5, *pyd1, *pyd2;
    PyObject *ar5, *ad5, *ad1, *ad2;
    PyArrayObject *pyrh = NULL, *pydh = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pyr5,
                                 &PyArray_Type, &pyd5,
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2))
        return NULL;
    ar5 = PyArray_FROM_OTF(pyr5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad5 = PyArray_FROM_OTF(pyd5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ar5 == NULL || ad5 == NULL || ad1 == NULL || ad1 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(ar5);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(ar5);
    if (dims[0] != PyArray_DIMS(ad5)[0] ||
        dims[0] != PyArray_DIMS(ad1)[0] || dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyrh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydh = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyrh || NULL == pydh) goto fail;
    r5 = (double *)PyArray_DATA(ar5);
    d5 = (double *)PyArray_DATA(ad5);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    rh = (double *)PyArray_DATA(pyrh);
    dh = (double *)PyArray_DATA(pydh);

    for (i=0;i<dims[0];i++) {
        eraFk5hz(r5[i], d5[i], d1[i], d2[i], &rh[i], &dh[i]);
    }
    Py_DECREF(ar5);
    Py_DECREF(ad5);
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyrh);
    Py_INCREF(pydh);
    return Py_BuildValue("OO", pyrh, pydh);

fail:
    Py_XDECREF(ar5);
    Py_XDECREF(ad5);
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyrh);
    Py_XDECREF(pydh);
    return NULL;        
}

PyDoc_STRVAR(_erfa_fk5hz_doc,
"\nfk5hz(r5, d5, d1, d2) -> rh, dh\n\n"
"Transform an FK5 (J2000.0) star position into the system of the\n"
"Hipparcos catalogue, assuming zero Hipparcos proper motion.\n"
"Given:\n"
"    r5         FK5 RA (radians), equinox J2000.0, at date d1,d2\n"
"    d5         FK5 Dec (radians), equinox J2000.0, at date d1,d2\n"
"    d1,d2      TDB date \n"
"Returned:\n"
"    rh         Hipparcos RA (radians)\n"
"    dh         Hipparcos Dec (radians)");

static PyObject *
_erfa_h2fk5(PyObject *self, PyObject *args)
{
    double *rh, *dh, *drh, *ddh, *pxh, *rvh;
    double *r5, *d5, *dr5, *dd5, *px5, *rv5;
    PyObject *pyrh, *pydh, *pydrh, *pyddh, *pypxh, *pyrvh;
    PyObject *arh, *adh, *adrh, *addh, *apxh, *arvh;
    PyArrayObject *pyr5 = NULL, *pyd5 = NULL, *pydr5 = NULL, *pydd5 = NULL, *pypx5 = NULL, *pyrv5 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", 
                                 &PyArray_Type, &pyrh,
                                 &PyArray_Type, &pydh,
                                 &PyArray_Type, &pydrh,
                                 &PyArray_Type, &pyddh,
                                 &PyArray_Type, &pypxh,
                                 &PyArray_Type, &pyrvh))
        return NULL;
    arh = PyArray_FROM_OTF(pyrh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adh = PyArray_FROM_OTF(pydh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adrh = PyArray_FROM_OTF(pydrh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    addh = PyArray_FROM_OTF(pyddh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    apxh = PyArray_FROM_OTF(pypxh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arvh = PyArray_FROM_OTF(pyrvh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arh == NULL || adh == NULL || adrh == NULL ||
        apxh == NULL || addh == NULL || arvh == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(arh);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(arh);
    if (dims[0] != PyArray_DIMS(adh)[0] ||
        dims[0] != PyArray_DIMS(adrh)[0] || dims[0] != PyArray_DIMS(apxh)[0] ||
        dims[0] != PyArray_DIMS(addh)[0] || dims[0] != PyArray_DIMS(arvh)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyr5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyd5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydr5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydd5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypx5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyrv5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyr5 || NULL == pyd5 || NULL == pydr5 ||
        NULL == pydd5 || NULL == pypx5 || NULL == pyrv5) goto fail;
    rh = (double *)PyArray_DATA(arh);
    dh = (double *)PyArray_DATA(adh);
    drh = (double *)PyArray_DATA(adrh);
    ddh = (double *)PyArray_DATA(addh);
    pxh = (double *)PyArray_DATA(apxh);
    rvh = (double *)PyArray_DATA(arvh);
    r5 = (double *)PyArray_DATA(pyr5);
    d5 = (double *)PyArray_DATA(pyd5);
    dr5 = (double *)PyArray_DATA(pydr5);
    dd5 = (double *)PyArray_DATA(pydd5);
    px5 = (double *)PyArray_DATA(pypx5);
    rv5 = (double *)PyArray_DATA(pyrv5);

    for (i=0;i<dims[0];i++) {
        eraH2fk5(rh[i], dh[i], drh[i], ddh[i], pxh[i], rvh[i],
                 &r5[i], &d5[i], &dr5[i], &dd5[i], &px5[i], &rv5[i]);
    }
    Py_DECREF(arh);
    Py_DECREF(adh);
    Py_DECREF(adrh);
    Py_DECREF(addh);
    Py_DECREF(apxh);
    Py_DECREF(arvh);
    Py_INCREF(pyr5);
    Py_INCREF(pyd5);
    Py_INCREF(pydr5);
    Py_INCREF(pydd5);
    Py_INCREF(pypx5);
    Py_INCREF(pyrv5);
    return Py_BuildValue("OOOOOO", pyr5, pyd5, pydr5, pydd5, pypx5, pyrv5);

fail:
    Py_XDECREF(arh);
    Py_XDECREF(adh);
    Py_XDECREF(adrh);
    Py_XDECREF(addh);
    Py_XDECREF(apxh);
    Py_XDECREF(arvh);
    Py_XDECREF(pyr5);
    Py_XDECREF(pyd5);
    Py_XDECREF(pydr5);
    Py_XDECREF(pydd5);
    Py_XDECREF(pypx5);
    Py_XDECREF(pyrv5);
    return NULL;
}

PyDoc_STRVAR(_erfa_h2fk5_doc,
"\nh2fk5(rh, dh, drh, ddh, pxh, rvh) -> r5, d5, dr5, dd5, px5, rv5\n\n"
"Transform Hipparcos star data into the FK5 (J2000.0) system.\n"
"Given (all Hipparcos, epoch J2000.0):\n"
"     rh    RA (radians)\n"
"     dh    Dec (radians)\n"
"     drh   proper motion in RA (dRA/dt, rad/Jyear)\n"
"     ddh   proper motion in Dec (dDec/dt, rad/Jyear)\n"
"     pxh   parallax (arcsec)\n"
"     rvh   radial velocity (km/s, positive = receding)\n"
"Returned (all FK5, equinox J2000.0, epoch J2000.0):\n"
"     r5    RA (radians)\n"
"     d5    Dec (radians)\n"
"     dr5   proper motion in RA (dRA/dt, rad/Jyear)\n"
"     dd5   proper motion in Dec (dDec/dt, rad/Jyear)\n"
"     px5   parallax (arcsec)\n"
"     rv5   radial velocity (km/s, positive = receding)");

static PyObject *
_erfa_hfk5z(PyObject *self, PyObject *args)
{
    double *rh, *dh, *d1, *d2;
    double *r5, *d5, *dr5, *dd5;
    PyObject *pyrh, *pydh, *pyd1, *pyd2;
    PyObject *arh, *adh, *ad1, *ad2;
    PyArrayObject *pyr5 = NULL, *pyd5 = NULL, *pydr5 = NULL, *pydd5 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                                 &PyArray_Type, &pyrh,
                                 &PyArray_Type, &pydh,
                                 &PyArray_Type, &pyd1,
                                 &PyArray_Type, &pyd2))
        return NULL;
    arh = PyArray_FROM_OTF(pyrh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adh = PyArray_FROM_OTF(pydh, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad1 = PyArray_FROM_OTF(pyd1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ad2 = PyArray_FROM_OTF(pyd2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arh == NULL || adh == NULL || ad1 == NULL || ad2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(arh);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(arh);
    if (dims[0] != PyArray_DIMS(adh)[0] ||
        dims[0] != PyArray_DIMS(ad1)[0] ||
        dims[0] != PyArray_DIMS(ad2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    pyr5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyd5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydr5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydd5 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyr5 || NULL == pyd5 || NULL == pydr5 || NULL == pydd5) goto fail;
    rh = (double *)PyArray_DATA(arh);
    dh = (double *)PyArray_DATA(adh);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    r5 = (double *)PyArray_DATA(pyr5);
    d5 = (double *)PyArray_DATA(pyd5);
    dr5 = (double *)PyArray_DATA(pydr5);
    dd5 = (double *)PyArray_DATA(pydd5);

    for (i=0;i<dims[0];i++) {
        eraHfk5z(rh[i], dh[i], d1[i], d2[i],
                 &r5[i], &d5[i], &dr5[i], &dd5[i]);
    }
    Py_DECREF(arh);
    Py_DECREF(adh);
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyr5);
    Py_INCREF(pyd5);
    Py_INCREF(pydr5);
    Py_INCREF(pydd5);
    return Py_BuildValue("OOOO", pyr5, pyd5, pydr5, pydd5);

fail:
    Py_XDECREF(arh);
    Py_XDECREF(adh);
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyr5);
    Py_XDECREF(pyd5);
    Py_XDECREF(pydr5);
    Py_XDECREF(pydd5);
    return NULL;
}

PyDoc_STRVAR(_erfa_hfk5z_doc,
"\nhfk5z(rh, dh, d1, d2) -> r5, d5, dr5, dd5\n\n"
"Transform a Hipparcos star position into FK5 J2000.0, assuming\n"
"zero Hipparcos proper motion.\n"
"Given:\n"
"     rh    Hipparcos RA (radians)\n"
"     dh    Hipparcos Dec (radians)\n"
"     d1,d2 TDB date\n"
"Returned (all FK5, equinox J2000.0, date date1+date2):\n"
"     r5    RA (radians)\n"
"     d5    Dec (radians)\n"
"     dr5   FK5 RA proper motion (rad/year)\n"
"     dd5   Dec proper motion (rad/year)");

static PyObject *
_erfa_eform(PyObject *self, PyObject *args)
{
    int n, status;
    double a, f;
    if (!PyArg_ParseTuple(args, "i", &n)) {
        return NULL;
    }
    status = eraEform(n, &a, &f);
    if (status) {
        PyErr_SetString(_erfaError, "illegal identifier; n should be 1,2 or 3");
        return NULL;
    }
    return Py_BuildValue("dd",a,f);
}

PyDoc_STRVAR(_erfa_eform_doc,
"\neform(n) -> a, f\n\n"
"Earth reference ellipsoids.\n"
"Given:\n"
"    n          ellipsoid identifier\n\n"
"    n=1 WGS84\n"
"    n=2 GRS80\n"
"    n=3 WGS72\n"
"Returned:\n"
"    a          equatorial radius (meters)\n"
"    f          flattening");

static PyObject *
_erfa_gc2gd(PyObject *self, PyObject *args)
{
    double xyz[3], *elong, *phi, *height;
    int n, status;
    PyObject *pyxyz;
    PyObject *axyz = NULL;
    PyObject *xyz_iter = NULL;
    PyArrayObject *pyelong = NULL, *pyphi = NULL, *pyheight = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "nO!", &n, &PyArray_Type, &pyxyz))
        return NULL;
    xyz_iter = PyArray_IterNew((PyObject*)pyxyz);
    if (xyz_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    ndim = PyArray_NDIM(pyxyz);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyxyz);
    dim_out[0] = dims[0];
    pyelong = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyphi = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyheight = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyelong || NULL == pyphi || NULL == pyheight ) goto fail;
    elong = (double *)PyArray_DATA(pyelong);
    phi = (double *)PyArray_DATA(pyphi);
    height = (double *)PyArray_DATA(pyheight);

    for (i=0;i<dims[0];i++) {
        int j;
        for (j=0;j<3;j++) {
            axyz = PyArray_GETITEM(pyxyz, PyArray_ITER_DATA(xyz_iter));
            if (axyz == NULL) {
                PyErr_SetString(_erfaError, "cannot retrieve data from args");
                goto fail;
            }
            Py_INCREF(axyz);
            xyz[j] = (double)PyFloat_AsDouble(axyz);
            Py_DECREF(axyz);
            PyArray_ITER_NEXT(xyz_iter);
        }
        status = eraGc2gd(n, xyz, &elong[i], &phi[i], &height[i]);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal identifier; n should be 1,2 or 3");
            goto fail;
        }
        if (status == -2) {
            PyErr_SetString(_erfaError, "internal error");
            goto fail;
        }
    }
    Py_DECREF(axyz);
    Py_DECREF(xyz_iter);
    Py_INCREF(pyelong); Py_INCREF(pyphi); Py_INCREF(pyheight);
    return Py_BuildValue("OOO", pyelong, pyphi, pyheight);

fail:
    Py_XDECREF(axyz);
    Py_XDECREF(xyz_iter);
    Py_XDECREF(pyelong); Py_XDECREF(pyphi); Py_XDECREF(pyheight);
    return NULL;
}

PyDoc_STRVAR(_erfa_gc2gd_doc,
"\ngc2gd(n, xyz) -> elong, phi, height\n\n"
"Transform geocentric coordinates to geodetic\n"
"using the specified reference ellipsoid.\n"
"Given:\n"
"    n          ellipsoid identifier\n"
"    xyz        geocentric vector\n"
"Returned:\n"
"    elong      longitude (radians, east +ve)\n"
"    phi        latitude (geodetic, radians)\n"
"    height     height above ellipsoid (geodetic)");

static PyObject *
_erfa_gc2gde(PyObject *self, PyObject *args)
{
    double xyz[3], a, f, *elong, *phi, *height;
    int status;
    PyObject *pyxyz;
    PyObject *axyz = NULL;
    PyObject *xyz_iter = NULL;
    PyArrayObject *pyelong = NULL, *pyphi = NULL, *pyheight = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "ddO!", &a, &f, &PyArray_Type, &pyxyz))
        return NULL;
    xyz_iter = PyArray_IterNew((PyObject*)pyxyz);
    if (xyz_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    ndim = PyArray_NDIM(pyxyz);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyxyz);
    dim_out[0] = dims[0];
    pyelong = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyphi = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    pyheight = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyelong || NULL == pyphi || NULL == pyheight ) goto fail;
    elong = (double *)PyArray_DATA(pyelong);
    phi = (double *)PyArray_DATA(pyphi);
    height = (double *)PyArray_DATA(pyheight);

    for (i=0;i<dims[0];i++) {
        int j;
        for (j=0;j<3;j++) {
            axyz = PyArray_GETITEM(pyxyz, PyArray_ITER_DATA(xyz_iter));
            if (axyz == NULL) {
                PyErr_SetString(_erfaError, "cannot retrieve data from args");
                goto fail;
            }
            Py_INCREF(axyz);
            xyz[j] = (double)PyFloat_AsDouble(axyz);
            Py_DECREF(axyz);
            PyArray_ITER_NEXT(xyz_iter);
        }
        status = eraGc2gde(a, f, xyz, &elong[i], &phi[i], &height[i]);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal f");
            goto fail;
        }
        if (status == -2) {
            PyErr_SetString(_erfaError, "illegal a");
            goto fail;
        }
    }
    Py_DECREF(axyz);
    Py_DECREF(xyz_iter);
    Py_INCREF(pyelong); Py_INCREF(pyphi); Py_INCREF(pyheight);
    return Py_BuildValue("OOO", pyelong, pyphi, pyheight);

fail:
    Py_XDECREF(axyz);
    Py_XDECREF(xyz_iter);
    Py_XDECREF(pyelong); Py_XDECREF(pyphi); Py_XDECREF(pyheight);
    return NULL;
}

PyDoc_STRVAR(_erfa_gc2gde_doc,
"\ngc2gde(a, f, xyz) -> elong, phi, height\n\n"
"Transform geocentric coordinates to geodetic\n"
" for a reference ellipsoid of specified form.\n"
"Given:\n"
"    a          equatorial radius\n"
"    f          flattening\n"
"    xyz        geocentric vector\n"
"Returned:\n"
"    elong      longitude (radians, east +ve)\n"
"    phi        latitude (geodetic, radians)\n"
"    height     height above ellipsoid (geodetic)");

static PyObject *
_erfa_gmst00(PyObject *self, PyObject *args)
{
    double *uta, *utb, *tta, *ttb, *g;
    PyObject *pyuta = NULL, *pyutb = NULL, *pytta = NULL, *pyttb = NULL;
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
_erfa_gmst06(PyObject *self, PyObject *args)
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
        g[i] = eraGmst06(uta[i], utb[i], tta[i], ttb[i]);
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

PyDoc_STRVAR(_erfa_gmst06_doc,
"\ngmst06(uta, utb, tta, ttb) -> gmst\n\n"
"Greenwich mean sidereal time\n"
"(model consistent with IAU 2006resolutions).\n"
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
_erfa_gst00a(PyObject *self, PyObject *args)
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
        g[i] = eraGst00a(uta[i], utb[i], tta[i], ttb[i]);
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

PyDoc_STRVAR(_erfa_gst00a_doc,
"\ngst00a(uta, utb, tta, ttb) -> gast\n\n"
"Greenwich apparent sidereal time\n"
"(model consistent with IAU 2000resolutions).\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"    tta,ttb    TT as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich apparent sidereal time (radians)");

static PyObject *
_erfa_gst00b(PyObject *self, PyObject *args)
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
        g[i] = eraGst00b(d1[i], d2[i]);
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

PyDoc_STRVAR(_erfa_gst00b_doc,
"\ngst00b(uta, utb) -> gast\n\n"
"Greenwich apparent sidereal time (model consistent with IAU 2000\n"
"resolutions but using the truncated nutation model IAU 2000B).\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich apparent sidereal time (radians)");

static PyObject *
_erfa_gst06(PyObject *self, PyObject *args)
{
    double *uta, *utb, *tta, *ttb, *g, rnpb[3][3];
    PyObject *pyuta, *pyutb, *pytta, *pyttb, *pyrnpb;
    PyObject *in_iter = NULL;
    PyObject *auta, *autb, *atta, *attb, *arnpb;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", 
                                 &PyArray_Type, &pyuta,
                                 &PyArray_Type, &pyutb,
                                 &PyArray_Type, &pytta,
                                 &PyArray_Type, &pyttb,
                                 &PyArray_Type, &pyrnpb))
        return NULL;

    auta = PyArray_FROM_OTF(pyuta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autb = PyArray_FROM_OTF(pyutb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atta = PyArray_FROM_OTF(pytta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    attb = PyArray_FROM_OTF(pyttb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (auta == NULL || autb == NULL || atta == NULL || attb == NULL) {
        goto fail;
    }
    in_iter = PyArray_IterNew((PyObject*)pyrnpb);
    if (in_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
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
        int j, k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {
                arnpb = PyArray_GETITEM(pyrnpb, PyArray_ITER_DATA(in_iter));
                if (arnpb == NULL) {
                    PyErr_SetString(_erfaError, "cannot retrieve data from args");
                    goto fail;
                }
                Py_INCREF(arnpb);
                PyArray_ITER_NEXT(in_iter);
                rnpb[j][k] = (double)PyFloat_AsDouble(arnpb);
                Py_DECREF(arnpb);
            }
        }
        g[i] = eraGst06(uta[i], utb[i], tta[i], ttb[i], rnpb);
    }
    Py_DECREF(auta);
    Py_DECREF(autb);
    Py_DECREF(atta);
    Py_DECREF(attb);
    Py_DECREF(in_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(auta);
    Py_XDECREF(autb);
    Py_XDECREF(atta);
    Py_XDECREF(attb);
    Py_XDECREF(in_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_gst06_doc,
"\ngst06(uta, utb, tta, ttb, rnpb[3][3]) -> gast\n\n"
"Greenwich apparent sidereal time, IAU 2006, given the NPB matrix.\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"    tta,ttb    TT as a 2-part Julian Date\n"
"    rnpb       nutation x precession x bias matrix\n"
"Returned:\n"
"    g          Greenwich apparent sidereal time");

static PyObject *
_erfa_gst06a(PyObject *self, PyObject *args)
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
        g[i] = eraGst06a(uta[i], utb[i], tta[i], ttb[i]);
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

PyDoc_STRVAR(_erfa_gst06a_doc,
"\ngst06a(uta, utb, tta, ttb) -> gast\n\n"
"Greenwich apparent sidereal time\n"
"(model consistent with IAU 2000and 2006 resolutions).\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"    tta,ttb    TT as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich apparent sidereal time (radians)");

static PyObject *
_erfa_gst94(PyObject *self, PyObject *args)
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
        g[i] = eraGst94(d1[i], d2[i]);
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

PyDoc_STRVAR(_erfa_gst94_doc,
"\ngst94(uta, utb) -> gast\n\n"
"Greenwich apparent sidereal time\n"
"(consistent with IAU 1982/94 resolutions)\n"
"Given:\n"
"    uta,utb    UT1 as a 2-part Julian Date\n"
"Returned:\n"
"    g          Greenwich mean sidereal time (radians)");

static PyObject *
_erfa_jd2cal(PyObject *self, PyObject *args)
{
    double *d1, *d2, *fd;
    int *iy, *im, *id, status;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyiy = NULL, *pyim = NULL, *pyid = NULL, *pyfd = NULL;
    PyArray_Descr *dsc, *idsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    idsc = PyArray_DescrFromType(NPY_INT32);
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
    pyiy = (PyArrayObject *) PyArray_Zeros(ndim, dims, idsc, 0);
    pyim = (PyArrayObject *) PyArray_Zeros(ndim, dims, idsc, 0);
    pyid = (PyArrayObject *) PyArray_Zeros(ndim, dims, idsc, 0);
    pyfd= (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    iy = (int *)PyArray_DATA(pyiy);
    im = (int *)PyArray_DATA(pyim);
    id = (int *)PyArray_DATA(pyid);
    fd = (double *)PyArray_DATA(pyfd);
    for (i=0;i<dims[0];i++) {
        status = eraJd2cal(d1[i], d2[i], &iy[i], &im[i], &id[i], &fd[i]);
        if (status) {
            PyErr_SetString(_erfaError, "unacceptable date");
            goto fail;
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyiy);Py_INCREF(pyim);Py_INCREF(pyid);Py_INCREF(pyfd);
    return Py_BuildValue("OOOO", pyiy, pyim, pyid, pyfd);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyiy);Py_XDECREF(pyim);Py_XDECREF(pyid);Py_XDECREF(pyfd);
    return NULL;
}

PyDoc_STRVAR(_erfa_jd2cal_doc,
"\njd2cal(dj1, dj2) -> year, month, day, fraction of day\n\n"
"Julian Date to Gregorian year, month, day, and fraction of a day.\n"
"Given:\n"
"     dj1,dj2   2-part Julian Date\n"
"Returned:\n"
"     iy    year\n"
"     im    month\n"
"     id    day\n"
"     fd    fraction of day");

static PyObject *
_erfa_jdcalf(PyObject *self, PyObject *args)
{
    double *d1, *d2;
    PyObject *pyd1, *pyd2;
    int ndp, status, iymdf[4];
    PyObject *ad1, *ad2;
    PyArrayObject *pyiymdf = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_LONG);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "iO!O!",
                                 &ndp,
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
    dim_out[0] = dims[0];
    dim_out[1] = 4;
    pyiymdf = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyiymdf) goto fail;
    out_iter = PyArray_IterNew((PyObject*)pyiymdf);
    if (out_iter == NULL) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    
    for (i=0;i<dims[0];i++) {
        status = eraJdcalf(ndp, d1[i], d2[i], iymdf);
        if (status == -1) {
            PyErr_SetString(_erfaError, "date out of range");
            goto fail;
        }
        if (status == 1) {
            PyErr_SetString(_erfaError, "n not in 0-9 (interpreted as 0)");
            goto fail;
        }
        int k;
        for (k=0;k<4;k++) {
            if (PyArray_SETITEM(pyiymdf, PyArray_ITER_DATA(out_iter), PyLong_FromLong((long)iymdf[k]))) {
                PyErr_SetString(_erfaError, "unable to set iymdf");
                goto fail;
            }
            PyArray_ITER_NEXT(out_iter);
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(out_iter);
    Py_INCREF(pyiymdf);
    return (PyObject *)pyiymdf;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyiymdf);
    return NULL;
}

PyDoc_STRVAR(_erfa_jdcalf_doc,
"\njdcalf(ndp, d1, d2) -> y, m, d, fd\n\n"
"Julian Date to Gregorian Calendar, expressed in a form convenient\n"
"for formatting messages:  rounded to a specified precision.\n"
"Given:\n"
"     ndp       number of decimal places of days in fraction\n"
"     d1,d2     2-part Julian Date\n"
"Returned:\n"
"     y         year\n"
"     m         month\n"
"     d         day\n"
"     fd        fraction of day");

static PyObject *
_erfa_num00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrmatn = NULL;
    PyObject *rmatn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrmatn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrmatn) {
        goto fail;
    }
    rmatn_iter = PyArray_IterNew((PyObject*)pyrmatn);
    if (rmatn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraNum00a(d1[i], d2[i], rmatn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrmatn, PyArray_ITER_DATA(rmatn_iter), PyFloat_FromDouble(rmatn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rmatn_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rmatn_iter);
    Py_INCREF(pyrmatn);
    return (PyObject *)pyrmatn;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rmatn_iter);
    Py_XDECREF(pyrmatn);
    return NULL;
}

PyDoc_STRVAR(_erfa_num00a_doc,
"\nnum00a(d1, d2) -> rmatn\n\n"
"Form the matrix of nutation for a given date, IAU 2000A model.\n"
"Given:\n"
"     d1,d2     TT as a 2-part Julian Date\n"
"Returned:\n"
"     rmatn     nutation matrix");

static PyObject *
_erfa_num00b(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrmatn = NULL;
    PyObject *rmatn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrmatn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrmatn) {
        goto fail;
    }
    rmatn_iter = PyArray_IterNew((PyObject*)pyrmatn);
    if (rmatn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraNum00b(d1[i], d2[i], rmatn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrmatn, PyArray_ITER_DATA(rmatn_iter), PyFloat_FromDouble(rmatn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rmatn_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rmatn_iter);
    Py_INCREF(pyrmatn);
    return (PyObject *)pyrmatn;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rmatn_iter);
    Py_XDECREF(pyrmatn);
    return NULL;
}

PyDoc_STRVAR(_erfa_num00b_doc,
"\nnum00b(d1, d2) -> rmatn\n\n"
"Form the matrix of nutation for a given date, IAU 2000B model.\n"
"Given:\n"
"     d1,d2     TT as a 2-part Julian Date\n"
"Returned:\n"
"     rmatn     nutation matrix");

static PyObject *
_erfa_num06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrmatn = NULL;
    PyObject *rmatn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrmatn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrmatn) {
        goto fail;
    }
    rmatn_iter = PyArray_IterNew((PyObject*)pyrmatn);
    if (rmatn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraNum06a(d1[i], d2[i], rmatn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrmatn, PyArray_ITER_DATA(rmatn_iter), PyFloat_FromDouble(rmatn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rmatn_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rmatn_iter);
    Py_INCREF(pyrmatn);
    return (PyObject *)pyrmatn;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rmatn_iter);
    Py_XDECREF(pyrmatn);
    return NULL;
}

PyDoc_STRVAR(_erfa_num06a_doc,
"\nnum06a(d1, d2) -> rmatn\n\n"
"Form the matrix of nutation for a given date, IAU 2006/2000A model.\n"
"Given:\n"
"     d1,d2     TT as a 2-part Julian Date\n"
"Returned:\n"
"     rmatn     nutation matrix");

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
_erfa_nut00b(PyObject *self, PyObject *args)
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
        eraNut00b(d1[i], d2[i], &dpsi, &deps);
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

PyDoc_STRVAR(_erfa_nut00b_doc,
"\nnut00b(d1, d2) -> dpsi, deps\n\n"
"Nutation, IAU 2000B.\n"
"Given:\n"
"    d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsi,deps   nutation, luni-solar + planetary\n");

static PyObject *
_erfa_nut06a(PyObject *self, PyObject *args)
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
        eraNut06a(d1[i], d2[i], &dpsi, &deps);
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

PyDoc_STRVAR(_erfa_nut06a_doc,
"\nnut06a(d1, d2) -> dpsi, deps\n\n"
"IAU 2000A nutation with adjustments to match the IAU 2006 precession.\n"
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
_erfa_nutm80(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrmatn = NULL;
    PyObject *rmatn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[1] = 3;
    dim_out[2] = 3;
    pyrmatn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrmatn) {
        goto fail;
    }
    rmatn_iter = PyArray_IterNew((PyObject*)pyrmatn);
    if (rmatn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraNutm80(d1[i], d2[i], rmatn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrmatn, PyArray_ITER_DATA(rmatn_iter), PyFloat_FromDouble(rmatn[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatn");
                    goto fail;
                }
                PyArray_ITER_NEXT(rmatn_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rmatn_iter);
    Py_INCREF(pyrmatn);
    return (PyObject *)pyrmatn;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rmatn_iter);
    Py_XDECREF(pyrmatn);
    return NULL;
}

PyDoc_STRVAR(_erfa_nutm80_doc,
"\nnutm80(d1, d2) -> rmatn\n\n"
"Form the matrix of nutation for a given date, IAU 1980 model.\n"
"Given:\n"
"    d1,d2      TDB as a 2-part Julian Date\n"
"Returned:\n"
"    rmatn      nutation matrix");

static PyObject *
_erfa_obl06(PyObject *self, PyObject *args)
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
        obl = eraObl06(d1[i], d2[i]);
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

PyDoc_STRVAR(_erfa_obl06_doc,
"\nobl06(d1, d2) -> obl\n\n"
"Mean obliquity of the ecliptic, IAU 2006 precession model.\n"
"Given:\n"
"    d1,d2      TT as a 2-part Julian Date\n"
"Returned:\n"
"    obl        obliquity of the ecliptic (radians)");

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
_erfa_p06e(PyObject *self, PyObject *args)
{
    double *d1, *d2, *eps0, *psia, *oma;
    double *bpa, *bqa, *pia, *bpia, *epsa;
    double *chia, *za, *zetaa, *thetaa;
    double *pa, *gam, *phi, *psi;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyeps0 = NULL, *pypsia = NULL, *pyoma = NULL;
    PyArrayObject *pybpa = NULL, *pybqa = NULL, *pypia = NULL;
    PyArrayObject *pybpia = NULL, *pyepsa = NULL, *pychia = NULL;
    PyArrayObject *pyza = NULL, *pyzetaa = NULL, *pythetaa = NULL;
    PyArrayObject *pypa = NULL, *pygam = NULL, *pyphi = NULL, *pypsi = NULL;
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
    pyeps0 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypsia = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyoma = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pybpa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pybqa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypia = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pybpia = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyepsa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pychia = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyza = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyzetaa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pythetaa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pygam = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyphi = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypsi = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyeps0 || NULL == pypsia || NULL == pyoma || NULL == pybpa || NULL == pybqa || NULL == pypia ||
        NULL == pybpia ||NULL == pyepsa || NULL == pychia || NULL == pyza || NULL == pyzetaa ||
        NULL == pythetaa || NULL == pypa || NULL == pygam || NULL == pyphi || NULL == pypsi) {
        goto fail;
    }
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    eps0 = (double *)PyArray_DATA(pyeps0);
    psia = (double *)PyArray_DATA(pypsia);
    oma = (double *)PyArray_DATA(pyoma);
    bpa = (double *)PyArray_DATA(pybpa);
    bqa = (double *)PyArray_DATA(pybqa);
    pia = (double *)PyArray_DATA(pypia);
    bpia = (double *)PyArray_DATA(pybpia);
    epsa = (double *)PyArray_DATA(pyepsa);
    chia = (double *)PyArray_DATA(pychia);
    za = (double *)PyArray_DATA(pyza);
    zetaa = (double *)PyArray_DATA(pyzetaa);
    thetaa = (double *)PyArray_DATA(pythetaa);
    pa = (double *)PyArray_DATA(pypa);
    gam = (double *)PyArray_DATA(pygam);
    phi = (double *)PyArray_DATA(pyphi);
    psi = (double *)PyArray_DATA(pypsi);

    for (i=0;i<dims[0];i++) {
        eraP06e(d1[i], d2[i],
                &eps0[i], &psia[i], &oma[i], &bpa[i],
                &bqa[i], &pia[i], &bpia[i], &epsa[i],
                &chia[i], &za[i], &zetaa[i], &thetaa[i],
                &pa[i], &gam[i], &phi[i], &psi[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pyeps0); Py_INCREF(pypsia); Py_INCREF(pyoma); Py_INCREF(pybpa);
    Py_INCREF(pybqa); Py_INCREF(pypia); Py_INCREF(pybpia); Py_INCREF(pyepsa);
    Py_INCREF(pychia); Py_INCREF(pyza); Py_INCREF(pyzetaa); Py_INCREF(pythetaa);
    Py_INCREF(pypa); Py_INCREF(pygam); Py_INCREF(pyphi); Py_INCREF(pypsi);
    return Py_BuildValue("OOOOOOOOOOOOOOOO",
                          pyeps0, pypsia, pyoma, pybpa,
                          pybqa, pypia, pybpia, pyepsa,
                          pychia, pyza, pyzetaa, pythetaa,
                          pypa, pygam, pyphi, pypsi);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pyeps0); Py_XDECREF(pypsia); Py_XDECREF(pyoma); Py_XDECREF(pybpa);
    Py_XDECREF(pybqa); Py_XDECREF(pypia); Py_XDECREF(pybpia); Py_XDECREF(pyepsa);
    Py_XDECREF(pychia); Py_XDECREF(pyza); Py_XDECREF(pyzetaa); Py_XDECREF(pythetaa);
    Py_XDECREF(pypa); Py_XDECREF(pygam); Py_XDECREF(pyphi); Py_XDECREF(pypsi);
    return NULL;    
}

PyDoc_STRVAR(_erfa_p06e_doc,
"\np06e(d1, d2) -> eps0,psia,oma,bpa,bqa,pia,bpia,epsa,chia,za,zetaa,thetaa,pa,gam,phi,psi\n\n"
"Precession angles, IAU 2006, equinox based.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   eps0   epsilon_0   obliquity at J2000.0\n"
"   psia   psi_A       luni-solar precession\n"
"   oma    omega_A     inclination of equator wrt J2000.0 ecliptic\n"
"   bpa    P_A         ecliptic pole x, J2000.0 ecliptic triad\n"
"   bqa    Q_A         ecliptic pole -y, J2000.0 ecliptic triad\n"
"   pia    pi_A        angle between moving and J2000.0 ecliptics\n"
"   bpia   Pi_A        longitude of ascending node of the ecliptic\n"
"   epsa   epsilon_A   obliquity of the ecliptic\n"
"   chia   chi_A       planetary precession\n"
"   za     z_A         equatorial precession: -3rd 323 Euler angle\n"
"   zetaa  zeta_A      equatorial precession: -1st 323 Euler angle\n"
"   thetaa theta_A     equatorial precession: 2nd 323 Euler angle\n"
"   pa     p_A         general precession\n"
"   gam    gamma_J2000 J2000.0 RA difference of ecliptic poles\n"
"   phi    phi_J2000   J2000.0 codeclination of ecliptic pole\n"
"   psi    psi_J2000   longitude difference of equator poles, J2000.0");

static PyObject *
_erfa_pb06(PyObject *self, PyObject *args)
{
    double *d1, *d2;
    double *bz, *bzeta, *btheta;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pybz = NULL, *pybzeta = NULL, *pybtheta = NULL;
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
    pybzeta = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pybz = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pybtheta = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pybz || NULL == pybzeta || NULL == pybtheta) {
        goto fail;
    }
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    bzeta = (double *)PyArray_DATA(pybzeta);
    bz = (double *)PyArray_DATA(pybz);
    btheta = (double *)PyArray_DATA(pybtheta);

    for (i=0;i<dims[0];i++) {
        eraPb06(d1[i], d2[i],
                 &bzeta[i], &bz[i], &btheta[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pybz); Py_INCREF(pybzeta); Py_INCREF(pybtheta);
    return Py_BuildValue("OOO",
                          pybzeta, pybz, pybtheta);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pybz); Py_XDECREF(pybzeta); Py_XDECREF(pybtheta);
    return NULL;    
}

PyDoc_STRVAR(_erfa_pb06_doc,
"\npb06(d1, d2) -> bzeta, bz, btheta\n\n"
"Forms three Euler angles which implement general\n"
"precession from epoch J2000.0, using the IAU 2006 model.\n"
"Framebias (the offset between ICRS and mean J2000.0) is included.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   bzeta   1st rotation: radians cw around z\n"
"   bz      3rd rotation: radians cw around z\n"
"   btheta  2nd rotation: radians ccw around y");

static PyObject *
_erfa_pfw06(PyObject *self, PyObject *args)
{
    double *d1, *d2, *gamb, *phib, *psib, *epsa;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pygamb = NULL, *pyphib = NULL, *pypsib = NULL, *pyepsa = NULL;
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
    pygamb = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyphib = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pypsib = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyepsa = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pygamb || NULL == pyphib || NULL == pypsib || NULL == pyepsa) {
        goto fail;
    }
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    gamb = (double *)PyArray_DATA(pygamb);
    phib = (double *)PyArray_DATA(pyphib);
    psib = (double *)PyArray_DATA(pypsib);
    epsa = (double *)PyArray_DATA(pyepsa);

    for (i=0;i<dims[0];i++) {
        eraPfw06(d1[i], d2[i],
                 &gamb[i], &phib[i], &psib[i], &epsa[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pygamb); Py_INCREF(pyphib); Py_INCREF(pypsib); Py_INCREF(pyepsa);
    return Py_BuildValue("OOOO",
                          pygamb, pyphib, pypsib, pyepsa);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pygamb); Py_XDECREF(pyphib); Py_XDECREF(pypsib); Py_XDECREF(pyepsa);
    return NULL;    
}

PyDoc_STRVAR(_erfa_pfw06_doc,
"\npfw06(d1, d2) -> gamb, phib, psib, epsa\n\n"
"Precession angles, IAU 2006 (Fukushima-Williams 4-angle formulation).\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   gamb    F-W angle gamma_bar (radians)\n"
"   phib    F-W angle phi_bar (radians)\n"
"   psib    F-W angle psi_bar (radians)\n"
"   epsa    F-W angle epsilon_A (radians)");

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
        if (status == 1) {
            PyErr_WarnEx(PyExc_Warning, "year outside range(1000:3000)", 1);
        }
        if (status == 2) {
            PyErr_WarnEx(PyExc_Warning,  "computation failed to converge", 1);
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
_erfa_pmat00(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbp[3][3];
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
    pyout = (PyArrayObject *)PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout)  goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);

    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraPmat00(d1[i], d2[i], rbp);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rbp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rbp");
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

PyDoc_STRVAR(_erfa_pmat00_doc,
"\npmat00(d1, d2) -> rbp\n\n"
"Precession matrix (including frame bias) from GCRS to a specified\n"
"date, IAU 2000 model.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   rbp         bias-precession matrix");

static PyObject *
_erfa_pmat06(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbp[3][3];
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
    pyout = (PyArrayObject *)PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyout)  goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);

    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        eraPmat06(d1[i], d2[i], rbp);
        int j,k;
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {            
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rbp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rbp");
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

PyDoc_STRVAR(_erfa_pmat06_doc,
"\npmat06(d1, d2) -> rbp\n\n"
"Precession matrix (including frame bias) from GCRS to a specified\n"
"date, IAU 2006 model.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   rbp         bias-precession matrix");

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
    Py_INCREF(pyepsa);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    Py_INCREF(pyrn);
    Py_INCREF(pyrbpn);
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
    Py_XDECREF(pyrb);
    Py_XDECREF(pyrp);
    Py_XDECREF(pyrbp);
    Py_XDECREF(pyrn);
    Py_XDECREF(pyrbpn);
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
_erfa_pn00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, dpsi, deps, epsa;
    double rb[3][3],rp[3][3],rbp[3][3],rn[3][3],rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pydpsi = NULL, *pydeps = NULL, *pyepsa = NULL, *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL, *pyrn = NULL, *pyrbpn = NULL;
    PyObject *dpsi_iter = NULL, *deps_iter = NULL, *epsa_iter = NULL, *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL, *rn_iter = NULL, *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3], dim_epsa[1];
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
    dim_epsa[0] = dims[0];
    pydpsi = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pydeps = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pyepsa = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    if (NULL == pyepsa|| NULL == pydpsi || NULL == pydeps) {
        goto fail;
    }
    dpsi_iter = PyArray_IterNew((PyObject*)pydpsi);
    deps_iter = PyArray_IterNew((PyObject*)pydeps);
    epsa_iter = PyArray_IterNew((PyObject*)pyepsa);
    if (dpsi_iter == NULL || deps_iter == NULL || epsa_iter == NULL) goto fail;
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
        eraPn00a(d1[i], d2[i],
                 &dpsi,& deps, &epsa,
                 rb, rp, rbp, rn, rbpn);
        if (PyArray_SETITEM(pydpsi, PyArray_ITER_DATA(dpsi_iter), PyFloat_FromDouble(dpsi))) {
            PyErr_SetString(_erfaError, "unable to set dpsi");
            goto fail;
        }
        PyArray_ITER_NEXT(dpsi_iter);
        if (PyArray_SETITEM(pydeps, PyArray_ITER_DATA(deps_iter), PyFloat_FromDouble(deps))) {
            PyErr_SetString(_erfaError, "unable to set deps");
            goto fail;
        }
        PyArray_ITER_NEXT(deps_iter);
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
    Py_DECREF(dpsi_iter);
    Py_DECREF(deps_iter);
    Py_DECREF(epsa_iter);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_DECREF(rn_iter);
    Py_DECREF(rbpn_iter);
    Py_INCREF(pydpsi);
    Py_INCREF(pydeps);
    Py_INCREF(pyepsa);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    Py_INCREF(pyrn);
    Py_INCREF(pyrbpn);
    return Py_BuildValue("OOOOOOOO", pydpsi, pydeps, pyepsa, pyrb, pyrp, pyrbp, pyrn, pyrbpn);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(dpsi_iter);
    Py_XDECREF(deps_iter);
    Py_XDECREF(epsa_iter);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    Py_XDECREF(rn_iter);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrb);
    Py_XDECREF(pyrp);
    Py_XDECREF(pyrbp);
    Py_XDECREF(pyrn);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pn00a_doc,
"\npn00a(d1,d2) -> dpsi,deps,epsa,rb,rp,rbp,rn,rbpn\n\n"
"Precession-nutation, IAU 2000A model:  a multi-purpose function,\n"
"supporting classical (equinox-based) use directly and CIO-based\n"
"use indirectly.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsi,deps   nutation\n"
"   epsa        mean obliquity\n"
"   rb          frame bias matrix\n"
"   rp          precession matrix\n"
"   rbp         bias-precession matrix\n"
"   rn          nutation matrix\n"
"   rbpn        GCRS-to-true matrix");

static PyObject *
_erfa_pn00b(PyObject *self, PyObject *args)
{
    double *d1, *d2, dpsi, deps, epsa;
    double rb[3][3],rp[3][3],rbp[3][3],rn[3][3],rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pydpsi = NULL, *pydeps = NULL, *pyepsa = NULL, *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL, *pyrn = NULL, *pyrbpn = NULL;
    PyObject *dpsi_iter = NULL, *deps_iter = NULL, *epsa_iter = NULL, *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL, *rn_iter = NULL, *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3], dim_epsa[1];
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
    dim_epsa[0] = dims[0];
    pydpsi = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pydeps = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pyepsa = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    if (NULL == pyepsa|| NULL == pydpsi || NULL == pydeps) {
        goto fail;
    }
    dpsi_iter = PyArray_IterNew((PyObject*)pydpsi);
    deps_iter = PyArray_IterNew((PyObject*)pydeps);
    epsa_iter = PyArray_IterNew((PyObject*)pyepsa);
    if (dpsi_iter == NULL || deps_iter == NULL || epsa_iter == NULL) goto fail;
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
        eraPn00b(d1[i], d2[i],
                 &dpsi,& deps, &epsa,
                 rb, rp, rbp, rn, rbpn);
        if (PyArray_SETITEM(pydpsi, PyArray_ITER_DATA(dpsi_iter), PyFloat_FromDouble(dpsi))) {
            PyErr_SetString(_erfaError, "unable to set dpsi");
            goto fail;
        }
        PyArray_ITER_NEXT(dpsi_iter);
        if (PyArray_SETITEM(pydeps, PyArray_ITER_DATA(deps_iter), PyFloat_FromDouble(deps))) {
            PyErr_SetString(_erfaError, "unable to set deps");
            goto fail;
        }
        PyArray_ITER_NEXT(deps_iter);
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
    Py_DECREF(dpsi_iter);
    Py_DECREF(deps_iter);
    Py_DECREF(epsa_iter);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_DECREF(rn_iter);
    Py_DECREF(rbpn_iter);
    Py_INCREF(pydpsi);
    Py_INCREF(pydeps);
    Py_INCREF(pyepsa);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    Py_INCREF(pyrn);
    Py_INCREF(pyrbpn);
    return Py_BuildValue("OOOOOOOO", pydpsi, pydeps, pyepsa, pyrb, pyrp, pyrbp, pyrn, pyrbpn);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(dpsi_iter);
    Py_XDECREF(deps_iter);
    Py_XDECREF(epsa_iter);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    Py_XDECREF(rn_iter);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrb);
    Py_XDECREF(pyrp);
    Py_XDECREF(pyrbp);
    Py_XDECREF(pyrn);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pn00b_doc,
"\npn00b(d1,d2) -> dpsi,deps,epsa,rb,rp,rbp,rn,rbpn\n\n"
"Precession-nutation, IAU 2000B model:  a multi-purpose function,\n"
"supporting classical (equinox-based) use directly and CIO-based\n"
"use indirectly.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsi,deps   nutation\n"
"   epsa        mean obliquity\n"
"   rb          frame bias matrix\n"
"   rp          precession matrix\n"
"   rbp         bias-precession matrix\n"
"   rn          nutation matrix\n"
"   rbpn        GCRS-to-true matrix");

static PyObject *
_erfa_pn06(PyObject *self, PyObject *args)
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
        eraPn06(d1[i], d2[i], dpsi[i], deps[i],
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
    Py_INCREF(pyepsa);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    Py_INCREF(pyrn);
    Py_INCREF(pyrbpn);
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
    Py_XDECREF(pyrb);
    Py_XDECREF(pyrp);
    Py_XDECREF(pyrbp);
    Py_XDECREF(pyrn);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pn06_doc,
"\npn06(d1,d2,dpsi,deps) -> epsa,rb,rp,rbp,rn,rbpn\n\n"
"Precession-nutation, IAU 2006 model:  a multi-purpose function,\n"
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
_erfa_pn06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, dpsi, deps, epsa;
    double rb[3][3],rp[3][3],rbp[3][3],rn[3][3],rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pydpsi = NULL, *pydeps = NULL, *pyepsa = NULL, *pyrb = NULL, *pyrp = NULL, *pyrbp = NULL, *pyrn = NULL, *pyrbpn = NULL;
    PyObject *dpsi_iter = NULL, *deps_iter = NULL, *epsa_iter = NULL, *rb_iter = NULL, *rp_iter = NULL, *rbp_iter = NULL, *rn_iter = NULL, *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3], dim_epsa[1];
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
    dim_epsa[0] = dims[0];
    pydpsi = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pydeps = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    pyepsa = (PyArrayObject *) PyArray_Zeros(1, dim_epsa, dsc, 0);
    if (NULL == pyepsa|| NULL == pydpsi || NULL == pydeps) {
        goto fail;
    }
    dpsi_iter = PyArray_IterNew((PyObject*)pydpsi);
    deps_iter = PyArray_IterNew((PyObject*)pydeps);
    epsa_iter = PyArray_IterNew((PyObject*)pyepsa);
    if (dpsi_iter == NULL || deps_iter == NULL || epsa_iter == NULL) goto fail;
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
        eraPn06a(d1[i], d2[i],
                 &dpsi,& deps, &epsa,
                 rb, rp, rbp, rn, rbpn);
        if (PyArray_SETITEM(pydpsi, PyArray_ITER_DATA(dpsi_iter), PyFloat_FromDouble(dpsi))) {
            PyErr_SetString(_erfaError, "unable to set dpsi");
            goto fail;
        }
        PyArray_ITER_NEXT(dpsi_iter);
        if (PyArray_SETITEM(pydeps, PyArray_ITER_DATA(deps_iter), PyFloat_FromDouble(deps))) {
            PyErr_SetString(_erfaError, "unable to set deps");
            goto fail;
        }
        PyArray_ITER_NEXT(deps_iter);
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
    Py_DECREF(dpsi_iter);
    Py_DECREF(deps_iter);
    Py_DECREF(epsa_iter);
    Py_DECREF(rb_iter);
    Py_DECREF(rp_iter);
    Py_DECREF(rbp_iter);
    Py_DECREF(rn_iter);
    Py_DECREF(rbpn_iter);
    Py_INCREF(pydpsi);
    Py_INCREF(pydeps);
    Py_INCREF(pyepsa);
    Py_INCREF(pyrb);
    Py_INCREF(pyrp);
    Py_INCREF(pyrbp);
    Py_INCREF(pyrn);
    Py_INCREF(pyrbpn);
    return Py_BuildValue("OOOOOOOO", pydpsi, pydeps, pyepsa, pyrb, pyrp, pyrbp, pyrn, pyrbpn);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(dpsi_iter);
    Py_XDECREF(deps_iter);
    Py_XDECREF(epsa_iter);
    Py_XDECREF(rb_iter);
    Py_XDECREF(rp_iter);
    Py_XDECREF(rbp_iter);
    Py_XDECREF(rn_iter);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrb);
    Py_XDECREF(pyrp);
    Py_XDECREF(pyrbp);
    Py_XDECREF(pyrn);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pn06a_doc,
"\npn06a(d1,d2) -> dpsi,deps,epsa,rb,rp,rbp,rn,rbpn\n\n"
"Precession-nutation, IAU 2006/2000A model:  a multi-purpose function,\n"
"supporting classical (equinox-based) use directly and CIO-based\n"
"use indirectly.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsi,deps   nutation\n"
"   epsa        mean obliquity\n"
"   rb          frame bias matrix\n"
"   rp          precession matrix\n"
"   rbp         bias-precession matrix\n"
"   rn          nutation matrix\n"
"   rbpn        GCRS-to-true matrix");

static PyObject *
_erfa_pnm00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrbpn = NULL;
    PyObject *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    pyrbpn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrbpn) {
        goto fail;
    }
    rbpn_iter = PyArray_IterNew((PyObject*)pyrbpn);
    if (rbpn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraPnm00a(d1[i], d2[i], rbpn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
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
    Py_DECREF(rbpn_iter);
    Py_INCREF(pyrbpn);
    return (PyObject*)pyrbpn;
fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pnm00a_doc,
"\npnm00a(d1, d2) -> rbpn\n\n"
"Form the matrix of precession-nutation for a given date (including\n"
"frame bias), equinox-based, IAU 2000A model.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   rbpn        classical NPB matrix");

static PyObject *
_erfa_pnm00b(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrbpn = NULL;
    PyObject *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    pyrbpn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrbpn) {
        goto fail;
    }
    rbpn_iter = PyArray_IterNew((PyObject*)pyrbpn);
    if (rbpn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraPnm00b(d1[i], d2[i], rbpn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
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
    Py_DECREF(rbpn_iter);
    Py_INCREF(pyrbpn);
    return (PyObject*)pyrbpn;
fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pnm00b_doc,
"\npnm00b(d1, d2) -> rbpn\n\n"
"Form the matrix of precession-nutation for a given date (including\n"
"frame bias), equinox-based, IAU 2000B model.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   rbpn        classical NPB matrix");

static PyObject *
_erfa_pnm06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, rbpn[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrbpn = NULL;
    PyObject *rbpn_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    pyrbpn = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrbpn) {
        goto fail;
    }
    rbpn_iter = PyArray_IterNew((PyObject*)pyrbpn);
    if (rbpn_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraPnm06a(d1[i], d2[i], rbpn);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
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
    Py_DECREF(rbpn_iter);
    Py_INCREF(pyrbpn);
    return (PyObject*)pyrbpn;
fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rbpn_iter);
    Py_XDECREF(pyrbpn);
    return NULL;
}

PyDoc_STRVAR(_erfa_pnm06a_doc,
"\npnm06a(d1, d2) -> rbpn\n\n"
"Form the matrix of precession-nutation for a given date (including\n"
"frame bias), IAU 2006 precession and IAU 2000A nutation models.\n"
"Given:\n"
"   d1,d2       TT as a 2-part Julian Date\n"
"Returned:\n"
"   rbpn        classical NPB matrix");

static PyObject *
_erfa_pnm80(PyObject *self, PyObject *args)
{
    double *d1, *d2, rmatp[3][3];
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pyrmatp = NULL;
    PyObject *rmatp_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
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
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    dim_out[2] = 3;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    pyrmatp = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pyrmatp) {
        goto fail;
    }
    rmatp_iter = PyArray_IterNew((PyObject*)pyrmatp);
    if (rmatp_iter == NULL) goto fail;

    for (i=0;i<dims[0];i++) {
        int j,k;
        eraPnm80(d1[i], d2[i], rmatp);
        for (j=0;j<3;j++) {
            for (k=0;k<3;k++) {                    
                if (PyArray_SETITEM(pyrmatp, PyArray_ITER_DATA(rmatp_iter), PyFloat_FromDouble(rmatp[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set rmatp");
                    goto fail;
                }
                PyArray_ITER_NEXT(rmatp_iter);
            }
        }
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_DECREF(rmatp_iter);
    Py_INCREF(pyrmatp);
    return (PyObject*)pyrmatp;
fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(rmatp_iter);
    Py_XDECREF(pyrmatp);
    return NULL;
}

PyDoc_STRVAR(_erfa_pnm80_doc,
"\npnm80(d1, d2) -> rmatp\n\n"
"Form the matrix of precession/nutation for a given date, IAU 1976\n"
"precession model, IAU 1980 nutation model.\n"
"Given:\n"
"   d1,d2           TDB date as a 2-part Julian Date\n"
"Returned:\n"
"   rmatp           combined precession/nutation matrix");

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
_erfa_pr00(PyObject *self, PyObject *args)
{
    double *d1, *d2, *dpsipr, *depspr;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pydpsipr = NULL, *pydepspr = NULL;
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
    pydpsipr = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pydepspr = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pydpsipr || NULL == pydepspr) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    dpsipr = (double *)PyArray_DATA(pydpsipr);
    depspr = (double *)PyArray_DATA(pydepspr);
    for (i=0;i<dims[0];i++) {
        eraPr00(d1[i], d2[i], &dpsipr[i], &depspr[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pydpsipr);
    Py_INCREF(pydepspr);
    return Py_BuildValue("OO", pydpsipr, pydepspr);

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pydpsipr);
    Py_XDECREF(pydepspr);
    return NULL;
}

PyDoc_STRVAR(_erfa_pr00_doc,
"\npr00(d1,d2) -> dpsipr,depspr\n\n"
"Precession-rate part of the IAU 2000 precession-nutation models\n"
"(part of MHB2000).\n"
"Given:\n"
"   d1,d2           TT as a 2-part Julian Date\n"
"Returned:\n"
"   dpsipr,depspr   precession corrections");

static PyObject *
_erfa_prec76(PyObject *self, PyObject *args)
{
    double *ep01, *ep02, *ep11, *ep12, *zeta, *z, *theta;
    PyObject *pyep01, *pyep02, *pyep11, *pyep12;
    PyObject *aep01, *aep02, *aep11, *aep12;
    PyArrayObject *pyzeta = NULL, *pyz = NULL, *pytheta = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                                 &PyArray_Type, &pyep01,
                                 &PyArray_Type, &pyep02,
                                 &PyArray_Type, &pyep11,
                                 &PyArray_Type, &pyep12)) {
        return NULL;
    }
    aep01 = PyArray_FROM_OTF(pyep01, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep02 = PyArray_FROM_OTF(pyep02, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep11 = PyArray_FROM_OTF(pyep11, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aep12 = PyArray_FROM_OTF(pyep12, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if ( aep01 == NULL || aep02 == NULL || aep11 == NULL || aep12 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aep01);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aep01);
    pyzeta = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyz = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytheta = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyzeta || NULL == pyz || NULL == pytheta) {
        goto fail;
    }
    ep01 = (double *)PyArray_DATA(aep01);
    ep02 = (double *)PyArray_DATA(aep02);
    ep11 = (double *)PyArray_DATA(aep11);
    ep12 = (double *)PyArray_DATA(aep12);
    zeta = (double *)PyArray_DATA(pyzeta);
    z = (double *)PyArray_DATA(pyz);
    theta = (double *)PyArray_DATA(pytheta);
    for (i=0;i<dims[0];i++) {
        eraPrec76(ep01[i], ep02[i], ep11[i], ep12[i], &zeta[i], &z[i], &theta[i]);
    }
    Py_DECREF(aep01);
    Py_DECREF(aep02);
    Py_DECREF(aep11);
    Py_DECREF(aep12);
    Py_INCREF(pyzeta); Py_INCREF(pyz); Py_INCREF(pytheta);
    return Py_BuildValue("OOO", pyzeta, pyz, pytheta);

fail:
    Py_XDECREF(aep01);
    Py_XDECREF(aep02);
    Py_XDECREF(aep11);
    Py_XDECREF(aep12);
    Py_XDECREF(pyzeta); Py_XDECREF(pyz); Py_XDECREF(pytheta);
    return NULL;
}

PyDoc_STRVAR(_erfa_prec76_doc,
"\nprec76(ep01, ep02, ep11, ep12) -> zeta, z, theta\n\n"
"IAU 1976 precession model.\n"
"Given:\n"
"   ep01,ep02   TDB starting epoch as 2-part Julian Date\n"
"   ep11,ep12   TDB ending epoch as 2-part Julian Date\n"
"Returned:\n"
"   zeta        1st rotation: radians cw around z\n"
"   z           3rd rotation: radians cw around z\n"
"   theta       2nd rotation: radians ccw around y");

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
_erfa_s00a(PyObject *self, PyObject *args)
{
    double *d1, *d2, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pys = NULL;
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
    pys = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pys) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    s = (double *)PyArray_DATA(pys);
    for (i=0;i<dims[0];i++) {
        s[i] = eraS00a(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pys);
    return (PyObject *)pys;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pys);
    return NULL;
}

PyDoc_STRVAR(_erfa_s00a_doc,
"\ns00a(d1, d2) -> s\n\n"
"The CIO locator s, positioning the Celestial Intermediate Origin on\n"
"the equator of the Celestial Intermediate Pole, using the IAU 2000A\n"
"precession-nutation model.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
"Returned:\n"
"   s       the CIO locator s in radians");

static PyObject *
_erfa_s00b(PyObject *self, PyObject *args)
{
    double *d1, *d2, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pys = NULL;
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
    pys = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pys) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    s = (double *)PyArray_DATA(pys);
    for (i=0;i<dims[0];i++) {
        s[i] = eraS00b(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pys);
    return (PyObject *)pys;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pys);
    return NULL;
}

PyDoc_STRVAR(_erfa_s00b_doc,
"\ns00b(d1, d2) -> s\n\n"
"The CIO locator s, positioning the Celestial Intermediate Origin on\n"
"the equator of the Celestial Intermediate Pole, using the IAU 2000B\n"
"precession-nutation model.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
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
_erfa_s06a(PyObject *self, PyObject *args)
{
    double *d1, *d2, *s;
    PyObject *pyd1, *pyd2;
    PyObject *ad1, *ad2;
    PyArrayObject *pys = NULL;
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
    pys = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pys) goto fail;
    d1 = (double *)PyArray_DATA(ad1);
    d2 = (double *)PyArray_DATA(ad2);
    s = (double *)PyArray_DATA(pys);
    for (i=0;i<dims[0];i++) {
        s[i] = eraS06a(d1[i], d2[i]);
    }
    Py_DECREF(ad1);
    Py_DECREF(ad2);
    Py_INCREF(pys);
    return (PyObject *)pys;

fail:
    Py_XDECREF(ad1);
    Py_XDECREF(ad2);
    Py_XDECREF(pys);
    return NULL;
}

PyDoc_STRVAR(_erfa_s06a_doc,
"\ns06a(d1, d2) -> s\n\n"
"The CIO locator s, positioning the Celestial Intermediate Origin on\n"
"the equator of the Celestial Intermediate Pole, using the IAU 2006\n"
"precession and IAU 2000A nutation model.\n"
"Given:\n"
"   d1,d2   TT as a 2-part Julian Date\n"
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
_erfa_taiut1(PyObject *self, PyObject *args)
{
    double *tai1, *tai2, *dta, *ut11, *ut12;
    int status;
    PyObject *pytai1, *pytai2, *pydta;
    PyObject *atai1, *atai2, *adta;
    PyArrayObject *pyut11 = NULL, *pyut12 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pytai1,
                                 &PyArray_Type, &pytai2,
                                 &PyArray_Type, &pydta))
        return NULL;

    atai1 = PyArray_FROM_OTF(pytai1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atai2 = PyArray_FROM_OTF(pytai2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adta = PyArray_FROM_OTF(pydta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atai1 == NULL || atai2 == NULL || adta == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atai1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atai1);
    if (dims[0] != PyArray_DIMS(atai2)[0] ||
        dims[0] != PyArray_DIMS(adta)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyut11 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyut12 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyut11 || NULL == pyut12) goto fail;
    tai1 = (double *)PyArray_DATA(atai1);
    tai2 = (double *)PyArray_DATA(atai2);
    dta = (double *)PyArray_DATA(adta);
    ut11 = (double *)PyArray_DATA(pyut11);
    ut12 = (double *)PyArray_DATA(pyut12);
    for (i=0;i<dims[0];i++) {
        status = eraTaiut1(tai1[i], tai2[i], dta[i], &ut11[i], &ut12[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error ...");
            goto fail;
        }
    }
    Py_DECREF(atai1);
    Py_DECREF(atai2);
    Py_DECREF(adta);
    Py_INCREF(pyut11); Py_INCREF(pyut12);
    return Py_BuildValue("OO", pyut11, pyut12);

fail:
    Py_XDECREF(atai1);
    Py_XDECREF(atai2);
    Py_XDECREF(adta);
    Py_XDECREF(pyut11); Py_XDECREF(pyut12);
    return NULL;
}

PyDoc_STRVAR(_erfa_taiut1_doc,
"\ntaiut1(tai1, tai2, dta) -> ut11, ut12\n\n"
"Time scale transformation:  International Atomic Time, TAI, to\n"
"Universal Time, UT1..\n"
"The argument dta, i.e. UT1-TAI, is an observed quantity, and is\n"
"available from IERS tabulations.\n"
"Given:\n"
"   tai1,tai2   TAI as a 2-part Julian Date\n"
"   dta         UT1-TAI in seconds\n"
"Returned:\n"
"   ut11,ut12     TT as a 2-part Julian Date");

static PyObject *
_erfa_taiutc(PyObject *self, PyObject *args)
{
    double *tai1, *tai2, *utc1, *utc2;
    int status;
    PyObject *pytai1, *pytai2;
    PyObject *atai1, *atai2;
    PyArrayObject *pyutc1 = NULL, *pyutc2 = NULL;
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

    pyutc1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyutc2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyutc1 || NULL == pyutc2) goto fail;
    tai1 = (double *)PyArray_DATA(atai1);
    tai2 = (double *)PyArray_DATA(atai2);
    utc1 = (double *)PyArray_DATA(pyutc1);
    utc2 = (double *)PyArray_DATA(pyutc2);
    for (i=0;i<dims[0];i++) {
        status = eraTaiutc(tai1[i], tai2[i], &utc1[i], &utc2[i]);
        if (status) {
            if (status == 1) {
                PyErr_SetString(_erfaError, "dubious year");
                goto fail;
            }
            else if (status == -1) {
                PyErr_SetString(_erfaError, "unacceptable date");
                goto fail;
            }
        }
    }
    Py_DECREF(atai1);
    Py_DECREF(atai2);
    Py_INCREF(pyutc1); Py_INCREF(pyutc2);
    return Py_BuildValue("OO", pyutc1, pyutc2);

fail:
    Py_XDECREF(atai1);
    Py_XDECREF(atai2);
    Py_XDECREF(pyutc1); Py_XDECREF(pyutc2);
    return NULL;
}

PyDoc_STRVAR(_erfa_taiutc_doc,
"\ntaiutc(tai1, tai2) -> utc1, utc2\n\n"
"Time scale transformation:  International Atomic Time, TAI, to\n"
"Coordinated Universal Time, UTC.\n"
"Given:\n"
"   tai1,tai2   TAI as a 2-part Julian Date\n"
"Returned:\n"
"   utc1,utc2   TT as a 2-part Julian Date");

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
_erfa_tdbtt(PyObject *self, PyObject *args)
{
    double *tdb1, *tdb2, *dtr, *tt1, *tt2;
    int status;
    PyObject *pytdb1, *pytdb2, *pydtr;
    PyObject *atdb1, *atdb2, *adtr;
    PyArrayObject *pytt1 = NULL, *pytt2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pytdb1,
                                 &PyArray_Type, &pytdb2,
                                 &PyArray_Type, &pydtr))
        return NULL;

    atdb1 = PyArray_FROM_OTF(pytdb1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atdb2 = PyArray_FROM_OTF(pytdb2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adtr = PyArray_FROM_OTF(pydtr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (atdb1 == NULL || atdb2 == NULL || adtr == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(atdb1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(atdb1);
    if (dims[0] != PyArray_DIMS(atdb2)[0] ||
        dims[0] != PyArray_DIMS(adtr)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytt1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytt2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytt1 || NULL == pytt2) goto fail;
    tdb1 = (double *)PyArray_DATA(atdb1);
    tdb2 = (double *)PyArray_DATA(atdb2);
    dtr = (double *)PyArray_DATA(adtr);
    tt1 = (double *)PyArray_DATA(pytt1);
    tt2 = (double *)PyArray_DATA(pytt2);
    for (i=0;i<dims[0];i++) {
        status = eraTdbtt(tdb1[i], tdb2[i], dtr[i], &tt1[i], &tt2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "should NOT occur");
            goto fail;
        }
    }
    Py_DECREF(atdb1);
    Py_DECREF(atdb2);
    Py_DECREF(adtr);
    Py_INCREF(pytt1); Py_INCREF(pytt2);
    return Py_BuildValue("OO", pytt1, pytt2);

fail:
    Py_XDECREF(atdb1);
    Py_XDECREF(atdb2);
    Py_XDECREF(adtr);
    Py_XDECREF(pytt1); Py_XDECREF(pytt2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tdbtt_doc,
"\ntdbtt(tdb1, tdb2, dtr) -> tt1, tt2\n\n"
"Time scale transformation: Barycentric Dynamical Time, TDB, to\n"
"Terrestrial Time, TT.\n"
"Given:\n"
"   tdb1,tdb2   TDB as a 2-part Julian Date\n"
"   dtr         TDB-TT in seconds\n"
"Returned:\n"
"   tt1,tt2   TT as a 2-part Julian Date");

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
_erfa_tttdb(PyObject *self, PyObject *args)
{
    double *tdb1, *tdb2, *dtr, *tt1, *tt2;
    int status;
    PyObject *pytt1, *pytt2, *pydtr;
    PyObject *att1, *att2, *adtr;
    PyArrayObject *pytdb1 = NULL, *pytdb2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pytt1,
                                 &PyArray_Type, &pytt2,
                                 &PyArray_Type, &pydtr))
        return NULL;

    att1 = PyArray_FROM_OTF(pytt1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    att2 = PyArray_FROM_OTF(pytt2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adtr = PyArray_FROM_OTF(pydtr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (att1 == NULL || att2 == NULL || adtr == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(att1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(att1);
    if (dims[0] != PyArray_DIMS(att2)[0] ||
        dims[0] != PyArray_DIMS(adtr)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytdb1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytdb2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytdb1 || NULL == pytdb2) goto fail;
    tt1 = (double *)PyArray_DATA(att1);
    tt2 = (double *)PyArray_DATA(att2);
    dtr = (double *)PyArray_DATA(adtr);
    tdb1 = (double *)PyArray_DATA(pytdb1);
    tdb2 = (double *)PyArray_DATA(pytdb2);
    for (i=0;i<dims[0];i++) {
        status = eraTttdb(tt1[i], tt2[i], dtr[i], &tdb1[i], &tdb2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error ...");
            goto fail;
        }
    }
    Py_DECREF(att1);
    Py_DECREF(att2);
    Py_DECREF(adtr);
    Py_INCREF(pytdb1); Py_INCREF(pytdb2);
    return Py_BuildValue("OO", pytdb1, pytdb2);

fail:
    Py_XDECREF(att1);
    Py_XDECREF(att2);
    Py_XDECREF(adtr);
    Py_XDECREF(pytdb1); Py_XDECREF(pytdb2);
    return NULL;
}

PyDoc_STRVAR(_erfa_tttdb_doc,
"\ntttdb(tt1, tt2, dtr) -> tdb1, tdb2\n\n"
"Time scale transformation: Terrestrial Time, TT, to\n"
"Barycentric Dynamical Time, TDB.\n"
"Given:\n"
"   tt1,tt2     TT as a 2-part Julian Date\n"
"   dtr         TDB-TT in seconds\n"
"Returned:\n"
"   tdb1,tdb2   TDB as a 2-part Julian Date");

static PyObject *
_erfa_ttut1(PyObject *self, PyObject *args)
{
    double *ut11, *ut12, *dt, *tt1, *tt2;
    int status;
    PyObject *pytt1, *pytt2, *pydt;
    PyObject *att1, *att2, *adt;
    PyArrayObject *pyut11 = NULL, *pyut12 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pytt1,
                                 &PyArray_Type, &pytt2,
                                 &PyArray_Type, &pydt))
        return NULL;

    att1 = PyArray_FROM_OTF(pytt1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    att2 = PyArray_FROM_OTF(pytt2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adt = PyArray_FROM_OTF(pydt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (att1 == NULL || att2 == NULL || adt == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(att1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(att1);
    if (dims[0] != PyArray_DIMS(att2)[0] ||
        dims[0] != PyArray_DIMS(adt)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyut11 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyut12 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyut11 || NULL == pyut12) goto fail;
    tt1 = (double *)PyArray_DATA(att1);
    tt2 = (double *)PyArray_DATA(att2);
    dt = (double *)PyArray_DATA(adt);
    ut11 = (double *)PyArray_DATA(pyut11);
    ut12 = (double *)PyArray_DATA(pyut12);
    for (i=0;i<dims[0];i++) {
        status = eraTtut1(tt1[i], tt2[i], dt[i], &ut11[i], &ut12[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error ...");
            goto fail;
        }
    }
    Py_DECREF(att1);
    Py_DECREF(att2);
    Py_DECREF(adt);
    Py_INCREF(pyut11); Py_INCREF(pyut12);
    return Py_BuildValue("OO", pyut11, pyut12);

fail:
    Py_XDECREF(att1);
    Py_XDECREF(att2);
    Py_XDECREF(adt);
    Py_XDECREF(pyut11); Py_XDECREF(pyut12);
    return NULL;
}

PyDoc_STRVAR(_erfa_ttut1_doc,
"\nttut1(tt1, tt2, dt) -> ut11, ut12\n\n"
"Time scale transformation: Terrestrial Time, TT, to\n"
"Universal Time UT1.\n"
"Given:\n"
"   tt1,tt2     TT as a 2-part Julian Date\n"
"   dt          TT-UT1 in seconds\n"
"Returned:\n"
"   ut11,ut12   UT1 as a 2-part Julian Date");

static PyObject *
_erfa_ut1tai(PyObject *self, PyObject *args)
{
    double *tai1, *tai2, *dta, *ut11, *ut12;
    int status;
    PyObject *pyut11, *pyut12, *pydta;
    PyObject *aut11, *aut12, *adta;
    PyArrayObject *pytai1 = NULL, *pytai2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyut11,
                                 &PyArray_Type, &pyut12,
                                 &PyArray_Type, &pydta))
        return NULL;

    aut11 = PyArray_FROM_OTF(pyut11, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aut12 = PyArray_FROM_OTF(pyut12, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adta = PyArray_FROM_OTF(pydta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aut11 == NULL || aut12 == NULL || adta == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aut11);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aut11);
    if (dims[0] != PyArray_DIMS(aut12)[0] ||
        dims[0] != PyArray_DIMS(adta)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytai1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytai2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytai1 || NULL == pytai2) goto fail;
    ut11 = (double *)PyArray_DATA(aut11);
    ut12 = (double *)PyArray_DATA(aut12);
    dta = (double *)PyArray_DATA(adta);
    tai1 = (double *)PyArray_DATA(pytai1);
    tai2 = (double *)PyArray_DATA(pytai2);
    for (i=0;i<dims[0];i++) {
        status = eraUt1tai(ut11[i], ut12[i], dta[i], &tai1[i], &tai2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error ...");
            goto fail;
        }
    }
    Py_DECREF(aut11);
    Py_DECREF(aut12);
    Py_DECREF(adta);
    Py_INCREF(pytai1); Py_INCREF(pytai2);
    return Py_BuildValue("OO", pytai1, pytai2);

fail:
    Py_XDECREF(aut11);
    Py_XDECREF(aut12);
    Py_XDECREF(adta);
    Py_XDECREF(pytai1); Py_XDECREF(pytai2);
    return NULL;
}

PyDoc_STRVAR(_erfa_ut1tai_doc,
"\nut1tai(ut11, ut12, dta) -> tai1, tai2\n\n"
"Time scale transformation: Universal Time, UT1, to\n"
"International Atomic Time, TAI.\n"
"The argument dta, i.e. UT1-TAI, is an observed quantity, and is\n"
"available from IERS tabulations.\n"
"Given:\n"
"   ut11,ut12     TT as a 2-part Julian Date\n"
"   dta         UT1-TAI in seconds\n"
"Returned:\n"
"   tai1,tai2   TAI as a 2-part Julian Date");

static PyObject *
_erfa_ut1tt(PyObject *self, PyObject *args)
{
    double *tt1, *tt2, *dt, *ut11, *ut12;
    int status;
    PyObject *pyut11, *pyut12, *pydt;
    PyObject *aut11, *aut12, *adt;
    PyArrayObject *pytt1 = NULL, *pytt2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyut11,
                                 &PyArray_Type, &pyut12,
                                 &PyArray_Type, &pydt))
        return NULL;

    aut11 = PyArray_FROM_OTF(pyut11, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aut12 = PyArray_FROM_OTF(pyut12, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adt = PyArray_FROM_OTF(pydt, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aut11 == NULL || aut12 == NULL || adt == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aut11);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aut11);
    if (dims[0] != PyArray_DIMS(aut12)[0] ||
        dims[0] != PyArray_DIMS(adt)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pytt1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytt2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytt1 || NULL == pytt2) goto fail;
    ut11 = (double *)PyArray_DATA(aut11);
    ut12 = (double *)PyArray_DATA(aut12);
    dt = (double *)PyArray_DATA(adt);
    tt1 = (double *)PyArray_DATA(pytt1);
    tt2 = (double *)PyArray_DATA(pytt2);
    for (i=0;i<dims[0];i++) {
        status = eraUt1tt(ut11[i], ut12[i], dt[i], &tt1[i], &tt2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "internal error ...");
            goto fail;
        }
    }
    Py_DECREF(aut11);
    Py_DECREF(aut12);
    Py_DECREF(adt);
    Py_INCREF(pytt1); Py_INCREF(pytt2);
    return Py_BuildValue("OO", pytt1, pytt2);

fail:
    Py_XDECREF(aut11);
    Py_XDECREF(aut12);
    Py_XDECREF(adt);
    Py_XDECREF(pytt1); Py_XDECREF(pytt2);
    return NULL;
}

PyDoc_STRVAR(_erfa_ut1tt_doc,
"\nut1tt(ut11, ut12, dt) -> tt1, tt2\n\n"
"Time scale transformation: Universal Time, UT1, to\n"
"Terrestrial Time, TT.\n"
"Given:\n"
"   ut11,ut12   UT1 as a 2-part Julian Date\n"
"   dt          TT-UT1 in seconds, dt is classical Delta T\n"
"Returned:\n"
"   tt1,tt2     TT as a 2-part Julian Date");

static PyObject *
_erfa_ut1utc(PyObject *self, PyObject *args)
{
    double *utc1, *utc2, *dut1, *ut11, *ut12;
    int status;
    PyObject *pyut11, *pyut12, *pydut1;
    PyObject *aut11, *aut12, *adut1;
    PyArrayObject *pyutc1 = NULL, *pyutc2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyut11,
                                 &PyArray_Type, &pyut12,
                                 &PyArray_Type, &pydut1))
        return NULL;

    aut11 = PyArray_FROM_OTF(pyut11, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aut12 = PyArray_FROM_OTF(pyut12, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adut1 = PyArray_FROM_OTF(pydut1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aut11 == NULL || aut12 == NULL || adut1 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aut11);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aut11);
    if (dims[0] != PyArray_DIMS(aut12)[0] ||
        dims[0] != PyArray_DIMS(adut1)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyutc1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyutc2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyutc1 || NULL == pyutc2) goto fail;
    ut11 = (double *)PyArray_DATA(aut11);
    ut12 = (double *)PyArray_DATA(aut12);
    dut1 = (double *)PyArray_DATA(adut1);
    utc1 = (double *)PyArray_DATA(pyutc1);
    utc2 = (double *)PyArray_DATA(pyutc2);
    for (i=0;i<dims[0];i++) {
        status = eraUt1utc(ut11[i], ut12[i], dut1[i], &utc1[i], &utc2[i]);
        if (status) {
            if (status == 1) {
                PyErr_SetString(_erfaError, "dubious year");
                goto fail;
            }
            else if (status == -1) {
                PyErr_SetString(_erfaError, "unacceptable date");
                goto fail;
            }
        }
    }
    Py_DECREF(aut11);
    Py_DECREF(aut12);
    Py_DECREF(adut1);
    Py_INCREF(pyutc1); Py_INCREF(pyutc2);
    return Py_BuildValue("OO", pyutc1, pyutc2);

fail:
    Py_XDECREF(aut11);
    Py_XDECREF(aut12);
    Py_XDECREF(adut1);
    Py_XDECREF(pyutc1); Py_XDECREF(pyutc2);
    return NULL;
}

PyDoc_STRVAR(_erfa_ut1utc_doc,
"\nut1utc(ut11, ut12, dut1) -> utc1, utc2\n\n"
"Time scale transformation: Universal Time, UT1, to\n"
"Coordinated Universal Time, UTC.\n"
"Given:\n"
"   ut11,ut12   UT1 as a 2-part Julian Date\n"
"   dut1        UT1-UTC in seconds, Delta UT1\n"
"Returned:\n"
"   utc1,utc2   UTC as a 2-part Julian Date");

static PyObject *
_erfa_utctai(PyObject *self, PyObject *args)
{
    double *utc1, *utc2, *tai1, *tai2;
    int status;
    PyObject *pyutc1, *pyutc2;
    PyObject *autc1, *autc2;
    PyArrayObject *pytai1 = NULL, *pytai2 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!",
                                 &PyArray_Type, &pyutc1,
                                 &PyArray_Type, &pyutc2))
        return NULL;

    autc1 = PyArray_FROM_OTF(pyutc1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autc2 = PyArray_FROM_OTF(pyutc2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (autc1 == NULL || autc2 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(autc1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(autc1);
    if (dims[0] != PyArray_DIMS(autc2)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }

    pytai1 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pytai2 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pytai1 || NULL == pytai2) goto fail;
    utc1 = (double *)PyArray_DATA(autc1);
    utc2 = (double *)PyArray_DATA(autc2);
    tai1 = (double *)PyArray_DATA(pytai1);
    tai2 = (double *)PyArray_DATA(pytai2);
    for (i=0;i<dims[0];i++) {
        status = eraUtctai(utc1[i], utc2[i], &tai1[i], &tai2[i]);
        if (status) {
            PyErr_SetString(_erfaError, "should NOT occur...");
            goto fail;
        }
    }
    Py_DECREF(autc1);
    Py_DECREF(autc2);
    Py_INCREF(pytai1); Py_INCREF(pytai2);
    return Py_BuildValue("OO", pytai1, pytai2);

fail:
    Py_XDECREF(autc1);
    Py_XDECREF(autc2);
    Py_XDECREF(pytai1); Py_XDECREF(pytai2);
    return NULL;
}

PyDoc_STRVAR(_erfa_utctai_doc,
"\nutctai(utc1, utc2) -> tai1, tai2\n\n"
"Time scale transformation: Coordinated Universal Time, UTC, to\n"
"International Atomic Time, TAI.\n"
"Given:\n"
"   utc1,uc12   UTC as a 2-part Julian Date\n"
"Returned:\n"
"   tai1,tai2   TAI as a 2-part Julian Date");

static PyObject *
_erfa_utcut1(PyObject *self, PyObject *args)
{
    double *utc1, *utc2, *dut1, *ut11, *ut12;
    int status;
    PyObject *pyutc1, *pyutc2, *pydut1;
    PyObject *autc1, *autc2, *adut1;
    PyArrayObject *pyut11 = NULL, *pyut12 = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims;
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!", 
                                 &PyArray_Type, &pyutc1,
                                 &PyArray_Type, &pyutc2,
                                 &PyArray_Type, &pydut1))
        return NULL;

    autc1 = PyArray_FROM_OTF(pyutc1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    autc2 = PyArray_FROM_OTF(pyutc2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    adut1 = PyArray_FROM_OTF(pydut1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (autc1 == NULL || autc2 == NULL || adut1 == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(autc1);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(autc1);
    if (dims[0] != PyArray_DIMS(autc2)[0] ||
        dims[0] != PyArray_DIMS(adut1)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    

    pyut11 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    pyut12 = (PyArrayObject *) PyArray_Zeros(ndim, dims, dsc, 0);
    if (NULL == pyut11 || NULL == pyut12) goto fail;
    utc1 = (double *)PyArray_DATA(autc1);
    utc2 = (double *)PyArray_DATA(autc2);
    dut1 = (double *)PyArray_DATA(adut1);
    ut11 = (double *)PyArray_DATA(pyut11);
    ut12 = (double *)PyArray_DATA(pyut12);
    for (i=0;i<dims[0];i++) {
        status = eraUtcut1(utc1[i], utc2[i], dut1[i], &ut11[i], &ut12[i]);
        if (status) {
            if (status == 1) {
                PyErr_SetString(_erfaError, "dubious year");
                goto fail;
            }
            else if (status == -1) {
                PyErr_SetString(_erfaError, "unacceptable date");
                goto fail;
            }
        }
    }
    Py_DECREF(autc1);
    Py_DECREF(autc2);
    Py_DECREF(adut1);
    Py_INCREF(pyut11); Py_INCREF(pyut12);
    return Py_BuildValue("OO", pyut11, pyut12);

fail:
    Py_XDECREF(autc1);
    Py_XDECREF(autc2);
    Py_XDECREF(adut1);
    Py_XDECREF(pyut11); Py_XDECREF(pyut12);
    return NULL;
}

PyDoc_STRVAR(_erfa_utcut1_doc,
"\nutcut1(utc1, utc2, dut1) -> ut11, ut12\n\n"
"Time scale transformation: Coordinated Universal Time, UTC, to\n"
"Universal Time, UT1.\n"
"Given:\n"
"   utc1,utc2   UTC as a 2-part Julian Date\n"
"   dut1        UT1-UTC in seconds, Delta UT1\n"
"Returned:\n"
"   ut11,ut12   UT1 as a 2-part Julian Date");

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
_erfa_xys00b(PyObject *self, PyObject *args)
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
        eraXys00b(d1[i], d2[i], &x[i], &y[i], &s[i]);
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

PyDoc_STRVAR(_erfa_xys00b_doc,
"\nxys00b(d1, d2) -> x, y, s\n\n"
"For a given TT date, compute the X,Y coordinates of the Celestial\n"
"Intermediate Pole and the CIO locator s, using the IAU 2000B\n"
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
_erfa_af2a(PyObject *self, PyObject *args)
{
    int ideg, iamin, status;
    double asec, rad;
    char sign;
    PyObject *pyin, *aideg, *aiamin, *aasec;
    PyObject *in_iter = NULL, *out_iter = NULL;
    PyArrayObject *pyout = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[1];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyin))
        return NULL;
    ndim = PyArray_NDIM(pyin);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyin);
    dim_out[0] = dims[0];
    pyout = (PyArrayObject *) PyArray_Zeros(1, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    in_iter = PyArray_IterNew((PyObject*)pyin);
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (in_iter == NULL || out_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        aideg = PyArray_GETITEM(pyin, PyArray_ITER_DATA(in_iter));
        if (aideg == NULL) {
            PyErr_SetString(_erfaError, "cannot retrieve data from args");
            goto fail;
        }
        Py_INCREF(aideg);
        PyArray_ITER_NEXT(in_iter);
        aiamin = PyArray_GETITEM(pyin, PyArray_ITER_DATA(in_iter));
        if (aiamin == NULL) {
            PyErr_SetString(_erfaError, "cannot retrieve data from args");
            goto fail;
        }
        Py_INCREF(aiamin);
        PyArray_ITER_NEXT(in_iter);
        aasec = PyArray_GETITEM(pyin, PyArray_ITER_DATA(in_iter));
        if (aasec == NULL) {
            PyErr_SetString(_erfaError, "cannot retrieve data from args");
            goto fail;
        }
        Py_INCREF(aasec);
        PyArray_ITER_NEXT(in_iter);
        ideg = (int)PyLong_AsLong(aideg);
        if (ideg == -1 && PyErr_Occurred()) goto fail;
        iamin = (int)PyLong_AsLong(aiamin);
        if (iamin == -1 && PyErr_Occurred()) goto fail;
        asec = (double)PyFloat_AsDouble(aasec);
        if (asec == -1 && PyErr_Occurred()) goto fail;
        if (ideg < 0) sign = '-'; else sign = '+';
        status = eraAf2a(sign, abs(ideg), iamin, asec, &rad);
        if (status == 1) {
            PyErr_WarnEx(PyExc_Warning, "ideg outside range 0-359", 1);
        }
        if (status == 2) {
            PyErr_WarnEx(PyExc_Warning, "iamin outside range 0-59", 1);
        }
        if (status == 3) {
            PyErr_WarnEx(PyExc_Warning, "asec outside range 0-59.999...", 1);
        }
        if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(rad))) {
            PyErr_SetString(_erfaError, "unable to set output rad");
            goto fail;
        }
        PyArray_ITER_NEXT(out_iter);
        Py_DECREF(aideg);
        Py_DECREF(aiamin);
        Py_DECREF(aasec);
    }
    Py_DECREF(in_iter);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;    

fail:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_af2a_doc,
"\naf2a(d, m, s) -> r\n"
"Convert degrees, arcminutes, arcseconds to radians.\n"
"Given:\n"
"/*    sign       '-' = negative, otherwise positive*/\n"
"    d          degrees\n"
"    m          arcminutes\n"
"    s          arcseconds\n"
"Returned:\n"
"    r          angle in radians");

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
_erfa_gd2gc(PyObject *self, PyObject *args)
{
    double *elong, *phi, *height, xyz[3];
    int n, status;
    PyObject *pyelong, *pyphi, *pyheight;
    PyObject *aelong = NULL, *aphi = NULL, *aheight = NULL;
    PyArrayObject *pyout = NULL;
    PyObject *out_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "iO!O!O!",
                                 &n,
                                 &PyArray_Type, &pyelong,
                                 &PyArray_Type, &pyphi,
                                 &PyArray_Type, &pyheight))      
        return NULL;
    ndim = PyArray_NDIM(pyelong);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(pyelong);
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    aelong = PyArray_FROM_OTF(pyelong, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphi = PyArray_FROM_OTF(pyphi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aheight = PyArray_FROM_OTF(pyheight, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aelong == NULL || aphi == NULL || aheight == NULL) goto fail;
    pyout = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyout) goto fail;
    out_iter = PyArray_IterNew((PyObject*)pyout);
    if (out_iter == NULL) goto fail;
    elong = (double *)PyArray_DATA(aelong);
    phi = (double *)PyArray_DATA(aphi);
    height = (double *)PyArray_DATA(aheight);
    for (i=0;i<dims[0];i++) {
        status = eraGd2gc(n, elong[i], phi[i], height[i], xyz);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal identifier; n should be 1,2 or 3");
            goto fail;
        }
        else if (status == -2) {
            PyErr_SetString(_erfaError, "illegal case");
            goto fail;
        }
        else {
            int j;
            for (j=0;j<3;j++) {
                if (PyArray_SETITEM(pyout, PyArray_ITER_DATA(out_iter), PyFloat_FromDouble(xyz[j]))) {
                    PyErr_SetString(_erfaError, "unable to set xyz");
                    goto fail;
                }
                PyArray_ITER_NEXT(out_iter);
            }
        }
    }
    Py_DECREF(aelong);
    Py_DECREF(aphi);
    Py_DECREF(aheight);
    Py_DECREF(out_iter);
    Py_INCREF(pyout);
    return (PyObject *)pyout;

fail:
    Py_XDECREF(aelong);
    Py_XDECREF(aphi);
    Py_XDECREF(aheight);
    Py_XDECREF(out_iter);
    Py_XDECREF(pyout);
    return NULL;
}

PyDoc_STRVAR(_erfa_gd2gc_doc,
"\ngd2gc(n, elong, phi, height) -> xyz\n\n"
"Transform geodetic coordinates to geocentric\n"
" using the specified reference ellipsoid.\n"
"Given:\n"
"    n          ellipsoid identifier\n"
"    elong      longitude (radians, east +ve)\n"
"    phi        latitude (geodetic, radians)\n"
"    height     height above ellipsoid (geodetic in meters)\n"
"  Returned:\n"
"    xyz        geocentric vector in meters");

static PyObject *
_erfa_gd2gce(PyObject *self, PyObject *args)
{
    double a, f, *elong, *phi, *height, xyz[3];
    int status;
    PyObject *pyelong, *pyphi, *pyheight;
    PyObject *aelong = NULL, *aphi = NULL, *aheight = NULL;
    PyArrayObject *pyxyz = NULL;
    PyObject *xyz_iter = NULL;
    PyArray_Descr * dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[2];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "ddO!O!O!",
                                 &a, &f,
                                 &PyArray_Type, &pyelong,
                                 &PyArray_Type, &pyphi,
                                 &PyArray_Type, &pyheight))
        return NULL;
    aelong = PyArray_FROM_OTF(pyelong, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphi = PyArray_FROM_OTF(pyphi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aheight = PyArray_FROM_OTF(pyheight, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aelong == NULL || aphi == NULL || aheight == NULL) {
        goto fail;
    }

    ndim = PyArray_NDIM(aelong);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aelong);
    if (dims[0] != PyArray_DIMS(aphi)[0] ||
        dims[0] != PyArray_DIMS(aheight)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    dim_out[0] = dims[0];
    dim_out[1] = 3;
    pyxyz = (PyArrayObject *) PyArray_Zeros(2, dim_out, dsc, 0);
    if (NULL == pyxyz) goto fail;
    xyz_iter = PyArray_IterNew((PyObject*)pyxyz);
    if (xyz_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    elong = (double *)PyArray_DATA(aelong);
    phi = (double *)PyArray_DATA(aphi);
    height = (double *)PyArray_DATA(aheight);
    for (i=0;i<dims[0];i++) {
        status = eraGd2gce(a, f, elong[i], phi[i], height[i], xyz);
        if (status == -1) {
            PyErr_SetString(_erfaError, "illegal case");
            goto fail;
        }
        else {
            int j;
            for (j=0;j<3;j++) {
                if (PyArray_SETITEM(pyxyz, PyArray_ITER_DATA(xyz_iter), PyFloat_FromDouble(xyz[j]))) {
                    PyErr_SetString(_erfaError, "unable to set xyz");
                    goto fail;
                }
                PyArray_ITER_NEXT(xyz_iter);
            }
        }
    }
    Py_DECREF(aelong);
    Py_DECREF(aphi);
    Py_DECREF(aheight);
    Py_DECREF(xyz_iter);
    Py_INCREF(pyxyz);
    return (PyObject *)pyxyz;

fail:
    Py_XDECREF(aelong);
    Py_XDECREF(aphi);
    Py_XDECREF(aheight);
    Py_XDECREF(xyz_iter);
    Py_XDECREF(pyxyz);
    return NULL;
}

PyDoc_STRVAR(_erfa_gd2gce_doc,
"\ngd2gce(a, f, elong, phi, height) -> xyz\n\n"
"Transform geodetic coordinates to geocentric for a reference\n"
" for a reference ellipsoid of specified form\n"
"Given:\n"
"    a          equatorial radius in meters\n"
"    f          flattening\n"
"    elong      longitude (radians, east +ve)\n"
"    phi        latitude (geodetic, radians)\n"
"    height     height above ellipsoid (geodetic in meters)\n"
"  Returned:\n"
"    xyz        geocentric vector in meters");

static PyObject *
_erfa_pvtob(PyObject *self, PyObject *args)
{
    double *elong, *phi, *hm, *xp, *yp, *sp, *theta, pv[2][3];
    PyObject *pyelong, *pyphi, *pyhm, *pyxp, *pyyp, *pysp, *pytheta;
    PyObject *aelong, *aphi, *ahm, *axp, *ayp, *asp, *atheta;
    PyArrayObject *pypv = NULL;
    PyObject *pv_iter = NULL;
    PyArray_Descr *dsc;
    dsc = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims, dim_out[3];
    int ndim, i;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
                                 &PyArray_Type, &pyelong,
                                 &PyArray_Type, &pyphi,
                                 &PyArray_Type, &pyhm,
                                 &PyArray_Type, &pyxp,
                                 &PyArray_Type, &pyyp,
                                 &PyArray_Type, &pysp,
                                 &PyArray_Type, &pytheta)) {
        return NULL;
    }
    aelong = PyArray_FROM_OTF(pyelong, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    aphi = PyArray_FROM_OTF(pyphi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ahm = PyArray_FROM_OTF(pyhm, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    axp = PyArray_FROM_OTF(pyxp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ayp = PyArray_FROM_OTF(pyyp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    asp = PyArray_FROM_OTF(pysp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    atheta = PyArray_FROM_OTF(pytheta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aelong == NULL || aphi == NULL || ahm == NULL || axp == NULL ||
        ayp == NULL || asp == NULL || atheta == NULL) {
        goto fail;
    }
    ndim = PyArray_NDIM(aelong);
    if (!ndim) {
        PyErr_SetString(_erfaError, "argument is ndarray of length 0");
        goto fail;
    }
    dims = PyArray_DIMS(aelong);
    if (dims[0] != PyArray_DIMS(aphi)[0] || dims[0] != PyArray_DIMS(ahm)[0] ||
        dims[0] != PyArray_DIMS(axp)[0] || dims[0] != PyArray_DIMS(ayp)[0] ||
        dims[0] != PyArray_DIMS(asp)[0] || dims[0] != PyArray_DIMS(atheta)[0]) {
        PyErr_SetString(_erfaError, "arguments have incompatible shape ");
        goto fail;
    }    
    elong = (double *)PyArray_DATA(aelong);
    phi = (double *)PyArray_DATA(aphi);
    hm = (double *)PyArray_DATA(ahm);
    xp = (double *)PyArray_DATA(axp);
    yp = (double *)PyArray_DATA(ayp);
    sp = (double *)PyArray_DATA(asp);
    theta = (double *)PyArray_DATA(atheta);
    dim_out[0] = dims[0];
    dim_out[1] = 2;
    dim_out[2] = 3;
    pypv = (PyArrayObject *) PyArray_Zeros(3, dim_out, dsc, 0);
    if (NULL == pypv) {
        goto fail;
    }
    pv_iter = PyArray_IterNew((PyObject*)pypv);
    if (pv_iter == NULL) {
        PyErr_SetString(_erfaError, "cannot create iterators");
        goto fail;
    }
    for (i=0;i<dims[0];i++) {
        eraPvtob(elong[i], phi[i], hm[i], xp[i], yp[i], sp[i], theta[i], pv);
        int j,k;
        for (j=0;j<2;j++) {
            for (k=0;k<3;k++) {
                if (PyArray_SETITEM(pypv, PyArray_ITER_DATA(pv_iter), PyFloat_FromDouble(pv[j][k]))) {
                    PyErr_SetString(_erfaError, "unable to set pv");
                    goto fail;
                }
                PyArray_ITER_NEXT(pv_iter);
            }
        }
    }
    Py_DECREF(aelong);
    Py_DECREF(aphi);
    Py_DECREF(ahm);
    Py_DECREF(axp);
    Py_DECREF(ayp);
    Py_DECREF(asp);
    Py_DECREF(atheta);
    Py_DECREF(pv_iter);
    Py_INCREF(pypv);
    return PyArray_Return(pypv);

fail:
    Py_XDECREF(aelong);
    Py_XDECREF(aphi);
    Py_XDECREF(ahm);
    Py_XDECREF(axp);
    Py_XDECREF(ayp);
    Py_XDECREF(asp);
    Py_XDECREF(atheta);
    Py_XDECREF(pv_iter);
    Py_XDECREF(pypv);
    return NULL;
}

PyDoc_STRVAR(_erfa_pvtob_doc,
"\npvtob(elong, phi, hm, xp, yp, sp, theta) -> pv[2][3]\n"
"Position and velocity of a terrestrial observing station.\n"
"Given:\n"
"    elong   double       longitude (radians, east +ve)\n"
"    phi     double       latitude (geodetic, radians))\n"
"    hm      double       height above ref. ellipsoid (geodetic, m)\n"
"    xp,yp   double       coordinates of the pole (radians))\n"
"    sp      double       the TIO locator s' (radians))\n"
"    theta   double       Earth rotation angle (radians)\n"
"Returned:\n"
"    pv      position/velocity vector (m, m/s, CIRS)");

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
    {"apcg", _erfa_apcg, METH_VARARGS, _erfa_apcg_doc},
    {"apcg13", _erfa_apcg13, METH_VARARGS, _erfa_apcg13_doc},
    {"apci", _erfa_apci, METH_VARARGS, _erfa_apci_doc},
    {"apco", _erfa_apco, METH_VARARGS, _erfa_apco_doc},
    {"apcs", _erfa_apcs, METH_VARARGS, _erfa_apcs_doc},
    {"ld", _erfa_ld, METH_VARARGS, _erfa_ld_doc},
    {"pmsafe", _erfa_pmsafe, METH_VARARGS, _erfa_pmsafe_doc},
    {"pvstar", _erfa_pvstar, METH_VARARGS, _erfa_pvstar_doc},
    {"starpv", _erfa_starpv, METH_VARARGS, _erfa_starpv_doc},
    {"starpm", _erfa_starpm, METH_VARARGS, _erfa_starpm_doc},
    {"bi00", (PyCFunction)_erfa_bi00, METH_NOARGS, _erfa_bi00_doc},
    {"bp00", _erfa_bp00, METH_VARARGS, _erfa_bp00_doc},
    {"bp06", _erfa_bp06, METH_VARARGS, _erfa_bp06_doc},
    {"bpn2xy", _erfa_bpn2xy, METH_VARARGS, _erfa_bpn2xy_doc},
    {"c2i00a", _erfa_c2i00a, METH_VARARGS, _erfa_c2i00a_doc},
    {"c2i00b", _erfa_c2i00b, METH_VARARGS, _erfa_c2i00b_doc},
    {"c2i06a", _erfa_c2i06a, METH_VARARGS, _erfa_c2i06a_doc},
    {"c2ibpn", _erfa_c2ibpn, METH_VARARGS, _erfa_c2ibpn_doc},
    {"c2ixy", _erfa_c2ixy, METH_VARARGS, _erfa_c2ixy_doc},
    {"c2ixys", _erfa_c2ixys, METH_VARARGS, _erfa_c2ixys_doc},
    {"c2t00a", _erfa_c2t00a, METH_VARARGS, _erfa_c2t00a_doc},
    {"c2t00b", _erfa_c2t00b, METH_VARARGS, _erfa_c2t00b_doc},
    {"c2t06a", _erfa_c2t06a, METH_VARARGS, _erfa_c2t06a_doc},
    {"c2tcio", _erfa_c2tcio, METH_VARARGS, _erfa_c2tcio_doc},
    {"c2teqx", _erfa_c2teqx, METH_VARARGS, _erfa_c2teqx_doc},
    {"c2tpe", _erfa_c2tpe, METH_VARARGS, _erfa_c2tpe_doc},
    {"c2txy", _erfa_c2txy, METH_VARARGS, _erfa_c2txy_doc},
    {"eo06a", _erfa_eo06a, METH_VARARGS, _erfa_eo06a_doc},
    {"eors", _erfa_eors, METH_VARARGS, _erfa_eors_doc},
    {"fw2m", _erfa_fw2m, METH_VARARGS, _erfa_fw2m_doc},
    {"fw2xy", _erfa_fw2xy, METH_VARARGS, _erfa_fw2xy_doc},
    {"cal2jd", _erfa_cal2jd, METH_VARARGS, _erfa_cal2jd_doc},
    {"d2dtf", _erfa_d2dtf, METH_VARARGS, _erfa_d2dtf_doc},
    {"dat", _erfa_dat, METH_VARARGS, _erfa_dat_doc},
    {"dtdb", _erfa_dtdb, METH_VARARGS, _erfa_dtdb_doc},
    {"dtf2d", _erfa_dtf2d, METH_VARARGS, _erfa_dtf2d_doc},
    {"ee00", _erfa_ee00, METH_VARARGS, _erfa_ee00_doc},
    {"ee00a", _erfa_ee00a, METH_VARARGS, _erfa_ee00a_doc},
    {"ee00b", _erfa_ee00b, METH_VARARGS, _erfa_ee00b_doc},
    {"ee06a", _erfa_ee06a, METH_VARARGS, _erfa_ee06a_doc},
    {"eect00", _erfa_eect00, METH_VARARGS, _erfa_eect00_doc},
    {"epb", _erfa_epb, METH_VARARGS, _erfa_epb_doc},
    {"epb2jd", _erfa_epb2jd, METH_VARARGS, _erfa_epb2jd_doc},
    {"epj", _erfa_epj, METH_VARARGS, _erfa_epj_doc},
    {"epj2jd", _erfa_epj2jd, METH_VARARGS, _erfa_epj2jd_doc},
    {"epv00", _erfa_epv00, METH_VARARGS, _erfa_epv00_doc},
    {"eqeq94", _erfa_eqeq94, METH_VARARGS, _erfa_eqeq94_doc},
    {"era00", _erfa_era00, METH_VARARGS, _erfa_era00_doc},
    {"fad03", _erfa_fad03, METH_VARARGS, _erfa_fad03_doc},
    {"fae03", _erfa_fae03, METH_VARARGS, _erfa_fae03_doc},
    {"faf03", _erfa_faf03, METH_VARARGS, _erfa_faf03_doc},
    {"faju03", _erfa_faju03, METH_VARARGS, _erfa_faju03_doc},
    {"fal03", _erfa_fal03, METH_VARARGS, _erfa_fal03_doc},
    {"falp03", _erfa_falp03, METH_VARARGS, _erfa_falp03_doc},
    {"fama03", _erfa_fama03, METH_VARARGS, _erfa_fama03_doc},
    {"fame03", _erfa_fame03, METH_VARARGS, _erfa_fame03_doc},
    {"fane03", _erfa_fane03, METH_VARARGS, _erfa_fane03_doc},
    {"faom03", _erfa_faom03, METH_VARARGS, _erfa_faom03_doc},
    {"fapa03", _erfa_fapa03, METH_VARARGS, _erfa_fapa03_doc},
    {"fasa03", _erfa_fasa03, METH_VARARGS, _erfa_fasa03_doc},
    {"faur03", _erfa_faur03, METH_VARARGS, _erfa_faur03_doc},
    {"fave03", _erfa_fave03, METH_VARARGS, _erfa_fave03_doc},
    {"fk52h", _erfa_fk52h, METH_VARARGS, _erfa_fk52h_doc},
    {"fk5hip", (PyCFunction)_erfa_fk5hip, METH_NOARGS, _erfa_fk5hip_doc},
    {"fk5hz", _erfa_fk5hz, METH_VARARGS, _erfa_fk5hz_doc},
    {"h2fk5", _erfa_h2fk5, METH_VARARGS, _erfa_h2fk5_doc},
    {"hfk5z", _erfa_hfk5z, METH_VARARGS, _erfa_hfk5z_doc},
    {"eform", _erfa_eform, METH_VARARGS, _erfa_eform_doc},
    {"gc2gd", _erfa_gc2gd, METH_VARARGS, _erfa_gc2gd_doc},
    {"gc2gde", _erfa_gc2gde, METH_VARARGS, _erfa_gc2gde_doc},
    {"gmst00", _erfa_gmst00, METH_VARARGS, _erfa_gmst00_doc},
    {"gmst06", _erfa_gmst06, METH_VARARGS, _erfa_gmst06_doc},
    {"gmst82", _erfa_gmst82, METH_VARARGS, _erfa_gmst82_doc},
    {"gst00a", _erfa_gst00a, METH_VARARGS, _erfa_gst00a_doc},
    {"gst00b", _erfa_gst00b, METH_VARARGS, _erfa_gst00b_doc},
    {"gst06", _erfa_gst06, METH_VARARGS, _erfa_gst06_doc},
    {"gst06a", _erfa_gst06a, METH_VARARGS, _erfa_gst06a_doc},
    {"gst94", _erfa_gst94, METH_VARARGS, _erfa_gst94_doc},
    {"jd2cal", _erfa_jd2cal, METH_VARARGS, _erfa_jd2cal_doc},
    {"jdcalf", _erfa_jdcalf, METH_VARARGS, _erfa_jdcalf_doc},
    {"num00a", _erfa_num00a, METH_VARARGS, _erfa_num00a_doc},
    {"num00b", _erfa_num00b, METH_VARARGS, _erfa_num00b_doc},
    {"num06a", _erfa_num06a, METH_VARARGS, _erfa_num06a_doc},
    {"numat", _erfa_numat, METH_VARARGS, _erfa_numat_doc},
    {"nut00a", _erfa_nut00a, METH_VARARGS, _erfa_nut00a_doc},
    {"nut00b", _erfa_nut00b, METH_VARARGS, _erfa_nut00b_doc},
    {"nut06a", _erfa_nut06a, METH_VARARGS, _erfa_nut06a_doc},
    {"nut80", _erfa_nut80, METH_VARARGS, _erfa_nut80_doc},
    {"nutm80", _erfa_nutm80, METH_VARARGS, _erfa_nutm80_doc},
    {"obl06", _erfa_obl06, METH_VARARGS, _erfa_obl06_doc},
    {"obl80", _erfa_obl80, METH_VARARGS, _erfa_obl80_doc},
    {"p06e", _erfa_p06e, METH_VARARGS, _erfa_p06e_doc},
    {"pb06", _erfa_pb06, METH_VARARGS, _erfa_pb06_doc},
    {"pfw06", _erfa_pfw06, METH_VARARGS, _erfa_pfw06_doc},
    {"plan94", _erfa_plan94, METH_VARARGS, _erfa_plan94_doc},
    {"pmat00", _erfa_pmat00, METH_VARARGS, _erfa_pmat00_doc},
    {"pmat06", _erfa_pmat06, METH_VARARGS, _erfa_pmat06_doc},
    {"pmat76", _erfa_pmat76, METH_VARARGS, _erfa_pmat76_doc},
    {"pn00", _erfa_pn00, METH_VARARGS, _erfa_pn00_doc},
    {"pn00a", _erfa_pn00a, METH_VARARGS, _erfa_pn00a_doc},
    {"pn00b", _erfa_pn00b, METH_VARARGS, _erfa_pn00b_doc},
    {"pn06", _erfa_pn06, METH_VARARGS, _erfa_pn06_doc},
    {"pn06a", _erfa_pn06a, METH_VARARGS, _erfa_pn06a_doc},
    {"pnm00a", _erfa_pnm00a, METH_VARARGS, _erfa_pnm00a_doc},
    {"pnm00b", _erfa_pnm00b, METH_VARARGS, _erfa_pnm00b_doc},
    {"pnm06a", _erfa_pnm06a, METH_VARARGS, _erfa_pnm06a_doc},
    {"pnm80", _erfa_pnm80, METH_VARARGS, _erfa_pnm80_doc},
    {"pom00", _erfa_pom00, METH_VARARGS, _erfa_pom00_doc},
    {"pr00", _erfa_pr00, METH_VARARGS, _erfa_pr00_doc},
    {"prec76", _erfa_prec76, METH_VARARGS, _erfa_prec76_doc},
    {"s00", _erfa_s00, METH_VARARGS, _erfa_s00_doc},
    {"s00a", _erfa_s00a, METH_VARARGS, _erfa_s00a_doc},
    {"s00b", _erfa_s00b, METH_VARARGS, _erfa_s00b_doc},
    {"s06", _erfa_s06, METH_VARARGS, _erfa_s06_doc},
    {"s06a", _erfa_s06a, METH_VARARGS, _erfa_s06a_doc},
    {"sp00", _erfa_sp00, METH_VARARGS, _erfa_sp00_doc},
    {"taitt", _erfa_taitt, METH_VARARGS, _erfa_taitt_doc},
    {"taiut1", _erfa_taiut1, METH_VARARGS, _erfa_taiut1_doc},
    {"taiutc", _erfa_taiutc, METH_VARARGS, _erfa_taiutc_doc},
    {"tcbtdb", _erfa_tcbtdb, METH_VARARGS, _erfa_tcbtdb_doc},
    {"tcgtt", _erfa_tcgtt, METH_VARARGS, _erfa_tcgtt_doc},
    {"tdbtcb", _erfa_tdbtcb, METH_VARARGS, _erfa_tdbtcb_doc},
    {"tdbtt", _erfa_tdbtt, METH_VARARGS, _erfa_tdbtt_doc},
    {"tttai", _erfa_tttai, METH_VARARGS, _erfa_tttai_doc},
    {"tttcg", _erfa_tttcg, METH_VARARGS, _erfa_tttcg_doc},
    {"tttdb", _erfa_tttdb, METH_VARARGS, _erfa_tttdb_doc},
    {"ttut1", _erfa_ttut1, METH_VARARGS, _erfa_ttut1_doc},
    {"ut1tai", _erfa_ut1tai, METH_VARARGS, _erfa_ut1tai_doc},
    {"ut1tt", _erfa_ut1tt, METH_VARARGS, _erfa_ut1tt_doc},
    {"ut1utc", _erfa_ut1utc, METH_VARARGS, _erfa_ut1utc_doc},
    {"utctai", _erfa_utctai, METH_VARARGS, _erfa_utctai_doc},
    {"utcut1", _erfa_utcut1, METH_VARARGS, _erfa_utcut1_doc},
    {"xy06", _erfa_xy06, METH_VARARGS, _erfa_xy06_doc},
    {"xys00a", _erfa_xys00a, METH_VARARGS, _erfa_xys00a_doc},
    {"xys00b", _erfa_xys00b, METH_VARARGS, _erfa_xys00b_doc},
    {"xys06a", _erfa_xys06a, METH_VARARGS, _erfa_xys06a_doc},
    {"a2af", _erfa_a2af, METH_VARARGS, _erfa_a2af_doc},
    {"a2tf", _erfa_a2tf, METH_VARARGS, _erfa_a2tf_doc},
    {"af2a", _erfa_af2a, METH_VARARGS, _erfa_af2a_doc},
    {"anp", _erfa_anp, METH_VARARGS, _erfa_anp_doc},
    {"cr", _erfa_cr, METH_VARARGS, _erfa_cr_doc},
    {"gd2gc", _erfa_gd2gc, METH_VARARGS, _erfa_gd2gc_doc},
    {"gd2gce", _erfa_gd2gce, METH_VARARGS, _erfa_gd2gce_doc},
    {"pvtob", _erfa_pvtob, METH_VARARGS, _erfa_pvtob_doc},
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
