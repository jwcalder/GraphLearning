/* dijkstra.h: C code for acceleration of graphlearning library 
 *
 * *  Author: Jeff Calder, 2020.
 *
 */

#include <stdio.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"


void dijkstra_main(double *d, int *l, int *WI, int *K, double *WV, int *I, double *g, bool prog, int n, int M, int k);
void HJsolver_fmm(double *d, int *l, int *WI, int *K, double *WV, int *I, int *g, bool prog, int n, int M, int k, double p);
void HJsolver_jacobi(double *d, int *l, int *WI, int *K, double *WV, int *I, int *g, bool prog, int n, int M, int k, double p);
