/* memory_allocation.h - 
 *
 *  Basic vector and matrix memory allocation
 *
 *  Author: Jeff Calder, 2018.
 */

#include "stdbool.h"
int** array_int(int m, int n, int val);
bool* vector_bool(int m, bool val);
int* vector_int(int m, int val);
double* vector_double(int m, double val);
double** array_double(int m, int n, double val);

