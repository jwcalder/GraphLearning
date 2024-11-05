/* cextensions.c: C code for acceleration of graphlearning library 
 *
 * *  Author: Jeff Calder, 2019.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "lp_iterate.h"
#include "mnist_benchmark.h"
#include "hjsolvers.h"
//#include <stdio.h>
//#include <stdlib.h>
#include "tsne.h"

static PyObject* lp_iterate(PyObject* self, PyObject* args)
{

   double p;
   double Td;
   double tol;
   double progd;
   PyArrayObject *uu_array;
   PyArrayObject *ul_array;
   PyArrayObject *II_array;
   PyArrayObject *J_array;
   PyArrayObject *W_array;
   PyArrayObject *ind_array;
   PyArrayObject *val_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dddd", &PyArray_Type, &uu_array, &PyArray_Type, &ul_array, &PyArray_Type, &II_array, &PyArray_Type, &J_array, &PyArray_Type, &W_array,  &PyArray_Type, &ind_array, &PyArray_Type, &val_array, &p, &Td, &tol, &progd))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(uu_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(II_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(ind_array);
   int m = dim[0]; //Number labeled points

   double *uu = (double *) PyArray_DATA(uu_array);
   double *ul = (double *) PyArray_DATA(ul_array);
   int *II = (int *) PyArray_DATA(II_array);
   int *J = (int *) PyArray_DATA(J_array);
   double *W = (double *) PyArray_DATA(W_array);
   int *ind = (int *) PyArray_DATA(ind_array);
   double *val = (double *) PyArray_DATA(val_array);
   bool prog = (bool)progd;
   int T = (int)Td;

   //Call main function from C code
   lp_iterate_main(uu,ul,II,J,W,ind,val,p,T,tol,prog,n,M,m);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* lip_iterate(PyObject* self, PyObject* args)
{

   double Td;
   double tol;
   double progd;
   double weightedd;
   double alpha;
   double beta;
   PyArrayObject *u_array;
   PyArrayObject *II_array;
   PyArrayObject *J_array;
   PyArrayObject *W_array;
   PyArrayObject *ind_array;
   PyArrayObject *val_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!dddddd", &PyArray_Type, &u_array, &PyArray_Type, &II_array, &PyArray_Type, &J_array, &PyArray_Type, &W_array, &PyArray_Type, &ind_array, &PyArray_Type, &val_array, &Td, &tol, &progd, &weightedd, &alpha, &beta))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(II_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(ind_array);
   int m = dim[0]; //Number labeled points

   double *u = (double *) PyArray_DATA(u_array);
   int *II = (int *) PyArray_DATA(II_array);
   int *J = (int *) PyArray_DATA(J_array);
   double *W = (double *) PyArray_DATA(W_array);
   int *ind = (int *) PyArray_DATA(ind_array);
   double *val = (double *) PyArray_DATA(val_array);
   bool prog = (bool)progd;
   bool weighted = (bool)weightedd;
   int T = (int)Td;

   //Call main function from C code
   if(weighted)
      lip_iterate_weighted_main(u,II,J,W,ind,val,T,tol,prog,n,M,m);
   else
      lip_iterate_main(u,II,J,W,ind,val,T,tol,prog,n,M,m,alpha,beta);

   Py_INCREF(Py_None);
   return Py_None;
}



static PyObject* volume_mbo(PyObject* self, PyObject* args)
{

   double progd, lcountd, Td, volume_mult;
   PyArrayObject *u_array;
   PyArrayObject *II_array;
   PyArrayObject *J_array;
   PyArrayObject *W_array;
   PyArrayObject *ind_array;
   PyArrayObject *val_array;
   PyArrayObject *classCounts_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dddd", &PyArray_Type, &u_array, &PyArray_Type, &II_array, &PyArray_Type, &J_array, &PyArray_Type, &W_array,  &PyArray_Type, &ind_array, &PyArray_Type, &val_array, &PyArray_Type, &classCounts_array, &lcountd, &progd, &Td, &volume_mult))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(II_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(ind_array);
   int m = dim[0]; //Number labeled points

   int *u = (int *) PyArray_DATA(u_array);
   int *II = (int *) PyArray_DATA(II_array);
   int *J = (int *) PyArray_DATA(J_array);
   float *W = (float *) PyArray_DATA(W_array);
   int *ind = (int *) PyArray_DATA(ind_array);
   int *val = (int *) PyArray_DATA(val_array);
   int *classCounts = (int *) PyArray_DATA(classCounts_array);
   bool prog = (bool)progd;
   int lcount = (int)lcountd;
   float T = (float)Td;

   //Call main function from C code
   mbo_main(u,II,J,W,ind,val,classCounts,prog,n,M,m,lcount,100,1e-6,T,2-volume_mult,volume_mult);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* dijkstra_hl(PyObject* self, PyObject* args)
{

   double progd, max_radius;
   PyArrayObject *d_array;
   PyArrayObject *l_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *II_array;
   PyArrayObject *g_array;
   PyArrayObject *f_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!dd", &PyArray_Type, &d_array, &PyArray_Type, &l_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &II_array, &PyArray_Type, &g_array,  &PyArray_Type, &f_array, &progd, &max_radius))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(d_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(II_array);
   int k = dim[0]; //Number labeled points

   double *d = (double *) PyArray_DATA(d_array);
   int *l = (int *) PyArray_DATA(l_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *II = (int *) PyArray_DATA(II_array);
   double *g = (double *) PyArray_DATA(g_array);
   double *f = (double *) PyArray_DATA(f_array);
   bool prog = (bool)progd;

   //Call main function from C code
   dijkstra_hl_main(d,l,WI,K,WV,II,g,f,prog,n,M,k,max_radius);

   Py_INCREF(Py_None);
   return Py_None;
}
static PyObject* dijkstra(PyObject* self, PyObject* args)
{

   double progd, max_radius;
   PyArrayObject *d_array;
   PyArrayObject *l_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *II_array;
   PyArrayObject *g_array;
   PyArrayObject *f_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!dd", &PyArray_Type, &d_array, &PyArray_Type, &l_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &II_array, &PyArray_Type, &g_array,  &PyArray_Type, &f_array, &progd, &max_radius))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(d_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(II_array);
   int k = dim[0]; //Number labeled points

   double *d = (double *) PyArray_DATA(d_array);
   int *l = (int *) PyArray_DATA(l_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *II = (int *) PyArray_DATA(II_array);
   double *g = (double *) PyArray_DATA(g_array);
   double *f = (double *) PyArray_DATA(f_array);
   bool prog = (bool)progd;

   //Call main function from C code
   dijkstra_main(d,l,WI,K,WV,II,g,f,prog,n,M,k,max_radius);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* peikonal(PyObject* self, PyObject* args)
{

   double p, max_num_itd, converg_tol, num_bisection_itd,progd;
   PyArrayObject *u_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *bdy_set_array;
   PyArrayObject *f_array;
   PyArrayObject *g_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!ddddd", &PyArray_Type, &u_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &bdy_set_array, &PyArray_Type, &f_array,  &PyArray_Type, &g_array, &p, &max_num_itd, &converg_tol, &num_bisection_itd, &progd))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(bdy_set_array);
   int k = dim[0]; //Number labeled points

   double *u = (double *) PyArray_DATA(u_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *bdy_set = (int *) PyArray_DATA(bdy_set_array);
   double *f = (double *) PyArray_DATA(f_array);
   double *g = (double *) PyArray_DATA(g_array);
   int max_num_it = (int)max_num_itd;
   int num_bisection_it = (int) num_bisection_itd;
   bool prog = (bool)progd;

   //Call main function from C code
   peikonal_main(u,WI,K,WV,bdy_set,f,g,p,max_num_it,converg_tol,num_bisection_it,prog,n,M,k);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* peikonal_fmm(PyObject* self, PyObject* args)
{

   double p, num_bisection_itd;
   PyArrayObject *u_array;
   PyArrayObject *WI_array;
   PyArrayObject *K_array;
   PyArrayObject *WV_array;
   PyArrayObject *bdy_set_array;
   PyArrayObject *f_array;
   PyArrayObject *g_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!dd", &PyArray_Type, &u_array, &PyArray_Type, &WI_array, &PyArray_Type, &K_array, &PyArray_Type, &WV_array, &PyArray_Type, &bdy_set_array, &PyArray_Type, &f_array,  &PyArray_Type, &g_array, &p, &num_bisection_itd))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int n = dim[0]; //Number of vertices
   dim =  PyArray_DIMS(WI_array);
   int M = dim[0]; //Number nonzero entries in weight matrix
   dim =  PyArray_DIMS(bdy_set_array);
   int k = dim[0]; //Number labeled points

   double *u = (double *) PyArray_DATA(u_array);
   int *WI = (int *) PyArray_DATA(WI_array);
   int *K = (int *) PyArray_DATA(K_array);
   double *WV = (double *) PyArray_DATA(WV_array);
   int *bdy_set = (int *) PyArray_DATA(bdy_set_array);
   double *f = (double *) PyArray_DATA(f_array);
   double *g = (double *) PyArray_DATA(g_array);
   int num_bisection_it = (int) num_bisection_itd;

   //Call main function from C code
   peikonal_fmm_main(u,WI,K,WV,bdy_set,f,g,p,num_bisection_it,n,M,k);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject* ars(PyObject* self, PyObject* args)
{

   int origN, N, D, no_dims, max_iter, num_early;
   double perplexity, theta, *data, theta1, theta2, alpha, time_step;
   int rand_seed = -1;
   bool prog;

   PyArrayObject *X_array, *Y_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!iddiddddip",
                               &PyArray_Type, &X_array,
                               &PyArray_Type, &Y_array, 
                               &no_dims, &perplexity, &theta, &max_iter,
                               &time_step, &theta1, &theta2, &alpha, &num_early, &prog))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(X_array);
   N = dim[0]; //Number of data points
   D = dim[1]; //Number of dimensions of X

   double *X = (double *) PyArray_DATA(X_array);
   double *Y = (double *) PyArray_DATA(Y_array);

   tsne_run(X, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 250, 250, time_step, theta1, theta2, alpha, num_early, prog);

   Py_INCREF(Py_None);
   return Py_None;
}

/*  define functions in module */
static PyMethodDef CExtensionsMethods[] =
{
   {"lp_iterate", lp_iterate, METH_VARARGS, "C Code acceleration for Lplearning"},
   {"lip_iterate", lip_iterate, METH_VARARGS, "C Code acceleration for unweighted Lipschitz learning"},
   {"volume_mbo", volume_mbo, METH_VARARGS, "Volume Constrained MBO"},
   {"dijkstra", dijkstra, METH_VARARGS, "Dijkstra's algorithm"},
   {"dijkstra_hl", dijkstra_hl, METH_VARARGS, "Dijkstra's algorithm in Hopf-Lax formula"},
   {"peikonal", peikonal, METH_VARARGS, "C code version of p-eikonal solver via Gauss-Seidel"},
   {"peikonal_fmm", peikonal_fmm, METH_VARARGS, "C code version of p-eikonal solver via Fast Marching"},
   {"ars", ars, METH_VARARGS, "Attraction-Repulsion Swarming t-SNE"},
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





