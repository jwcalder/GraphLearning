/* cextensions.c: C code for acceleration of graphlearning library 
 *
 * *  Author: Jeff Calder, 2019.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "lp_iterate.h"
#include "mnist_benchmark.h"
#include "dijkstra.h"
#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h>

static PyObject* lp_iterate(PyObject* self, PyObject* args)
{

   double p;
   double Td;
   double tol;
   double progd;
   PyArrayObject *uu_array;
   PyArrayObject *ul_array;
   PyArrayObject *I_array;
   PyArrayObject *J_array;
   PyArrayObject *W_array;
   PyArrayObject *ind_array;
   PyArrayObject *val_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dddd", &PyArray_Type, &uu_array, &PyArray_Type, &ul_array, &PyArray_Type, &I_array, &PyArray_Type, &J_array, &PyArray_Type, &W_array,  &PyArray_Type, &ind_array, &PyArray_Type, &val_array, &p, &Td, &tol, &progd))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(uu_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(I_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(ind_array);
   int m = dim[0]; //Number labeled points

   double *uu = (double *) PyArray_DATA(uu_array);
   double *ul = (double *) PyArray_DATA(ul_array);
   int *I = (int *) PyArray_DATA(I_array);
   int *J = (int *) PyArray_DATA(J_array);
   double *W = (double *) PyArray_DATA(W_array);
   int *ind = (int *) PyArray_DATA(ind_array);
   double *val = (double *) PyArray_DATA(val_array);
   bool prog = (bool)progd;
   int T = (int)Td;

   //Call main function from C code
   lp_iterate_main(uu,ul,I,J,W,ind,val,p,T,tol,prog,n,M,m);

   Py_INCREF(Py_None);
   return Py_None;
}


static PyObject* volume_mbo(PyObject* self, PyObject* args)
{

   double progd, lcountd, Td, volume_mult;
   PyArrayObject *u_array;
   PyArrayObject *I_array;
   PyArrayObject *J_array;
   PyArrayObject *W_array;
   PyArrayObject *ind_array;
   PyArrayObject *val_array;
   PyArrayObject *classCounts_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dddd", &PyArray_Type, &u_array, &PyArray_Type, &I_array, &PyArray_Type, &J_array, &PyArray_Type, &W_array,  &PyArray_Type, &ind_array, &PyArray_Type, &val_array, &PyArray_Type, &classCounts_array, &lcountd, &progd, &Td, &volume_mult))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(I_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(ind_array);
   int m = dim[0]; //Number labeled points

   int *u = (int *) PyArray_DATA(u_array);
   int *I = (int *) PyArray_DATA(I_array);
   int *J = (int *) PyArray_DATA(J_array);
   float *W = (float *) PyArray_DATA(W_array);
   int *ind = (int *) PyArray_DATA(ind_array);
   int *val = (int *) PyArray_DATA(val_array);
   int *classCounts = (int *) PyArray_DATA(classCounts_array);
   bool prog = (bool)progd;
   int lcount = (int)lcountd;
   float T = (float)Td;

   //Call main function from C code
   mbo_main(u,I,J,W,ind,val,classCounts,prog,n,M,m,lcount,100,1e-6,T,2-volume_mult,volume_mult);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* dijkstra(PyObject* self, PyObject* args)
{

   double progd;
   PyArrayObject *d_array;
   PyArrayObject *l_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *I_array;
   PyArrayObject *g_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!d", &PyArray_Type, &d_array, &PyArray_Type, &l_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &I_array, &PyArray_Type, &g_array, &progd))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(d_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(I_array);
   int k = dim[0]; //Number labeled points

   double *d = (double *) PyArray_DATA(d_array);
   int *l = (int *) PyArray_DATA(l_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *I = (int *) PyArray_DATA(I_array);
   double *g = (double *) PyArray_DATA(g_array);
   bool prog = (bool)progd;

   //Call main function from C code
   dijkstra_main(d,l,WI,K,WV,I,g,prog,n,M,k);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* HJsolver(PyObject* self, PyObject* args)
{

   double progd, p, solver;
   PyArrayObject *d_array;
   PyArrayObject *l_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *I_array;
   PyArrayObject *g_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!ddd", &PyArray_Type, &d_array, &PyArray_Type, &l_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &I_array,  &PyArray_Type, &g_array, &progd, &p, &solver))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(d_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(I_array);
   int k = dim[0]; //Number labeled points

   double *d = (double *) PyArray_DATA(d_array);
   int *l = (int *) PyArray_DATA(l_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *I = (int *) PyArray_DATA(I_array);
   int *g = (int *) PyArray_DATA(g_array);
   bool prog = (bool)progd;

   int method = (int)solver;

   //Call main function from C code
   if(method == 0)
      HJsolver_fmm(d,l,WI,K,WV,I,g,prog,n,M,k,p);
   else
      HJsolver_jacobi(d,l,WI,K,WV,I,g,prog,n,M,k,p);

   Py_INCREF(Py_None);
   return Py_None;
}


/*  define functions in module */
static PyMethodDef CExtensionsMethods[] =
{
   {"lp_iterate", lp_iterate, METH_VARARGS, "C Code acceleration for Lplearning"},
   {"volume_mbo", volume_mbo, METH_VARARGS, "Volume Constrained MBO"},
   {"dijkstra", dijkstra, METH_VARARGS, "Dijkstra's algorithm"},
   {"HJsolver", HJsolver, METH_VARARGS, "Hamilton-Jacobi solver via Fast Marching"},
   {NULL, NULL, 0, NULL}
};

/* module initialization */
static struct PyModuleDef cModPyDem =
{
   PyModuleDef_HEAD_INIT,
   "cextensions", 
   "C code accelereation for graphlearning Python package",
   -1,
   CExtensionsMethods
};

PyMODINIT_FUNC PyInit_cextensions(void)
{
   import_array(); //This is not in fputs
   return PyModule_Create(&cModPyDem);
}





