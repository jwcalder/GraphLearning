/* lplearn.h - C code acceleration for Lplearning
 *
 *
 *  Used by the graphlearning package
 *
 * Inputs:
 *  uu = initial upper values nx1 array
 *  ul = initial lower values nx1 array
 *  (I,J) = indices of adjacency matrix, each nx1
 *  W = nx1 vector of weights for (I,J) entry
 *  ind = indices of Dirichlet conditions
 *  val = values of Dirichlet conditions
 *  T = number of iterations
 *  tol = tolerance
 *  prog = toggles progress indicator
 *
 * Outputs:
 *  uu = upper learned function nx1
 *  ul = lowerf learned function nx1
 *  r = residual nx1
 *
 * NOTE: Must supply uu,ul with uu=ul at ind points, uu a supersolution, ul a subsolution, and uu>=ul everywhere
 *
 *  Author: Jeff Calder, 2019.
 *
 */


#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"

void lp_iterate_main(double *uu, double *ul, int *II, int *J, double *W, int *ind, double *val, double p, int T, double tol, bool prog, int n, int M, int m);
void lip_iterate_main(double *u, int *II, int *J, double *W, int *ind, double *val, int T, double tol, bool prog, int n, int M, int m, double alpha, double beta);
void lip_iterate_weighted_main(double *u, int *II, int *J, double *W, int *ind, double *val, int T, double tol, bool prog, int n, int M, int m);


