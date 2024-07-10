/* dijkstra.h: C code for acceleration of graphlearning library 
 *
 * *  Author: Jeff Calder, 2020.
 *
 */

#include <stdio.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"


void dijkstra_main(double *d, int *l, int *WI, int *K, double *WV, int *II, double *g, double *f, bool prog, int n, int M, int k, double max_dist);
void dijkstra_hl_main(double *d, int *l, int *WI, int *K, double *WV, int *II, double *g, double *f, bool prog, int n, int M, int k, double max_dist);
void peikonal_main(double *u, int *WI, int *K, double *WV, int *II, double *f,  double *g, double p_val, int max_num_it, double tol, int num_bisection_it, bool prog, int n, int M, int k);
void peikonal_fmm_main(double *u, int *WI, int *K, double *WV, int *II, double *f,  double *g, double p_val, int num_bisection_it, int n, int M, int k);
