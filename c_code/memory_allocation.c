/* memory_allocation.c - 
 *
 *  Basic vector and matrix memory allocation
 *
 *  Author: Jeff Calder, 2018.
 */
 
#include "stdlib.h"
#include "stdbool.h"
#include "math.h"
#include "memory_allocation.h"

/*Allocate memory for a mxn array of int and initialize to val*/
int** array_int(int m, int n, int val){

   int **ptr = (int**)malloc(m*sizeof(int*));
   ptr[0] = (int*)malloc(m*n*sizeof(int));
   int i,j;
   for(i=0;i<m;i++){
      ptr[i] = ptr[0] + n*i;
      for(j=0;j<n;j++){
         ptr[i][j] = val;
      }
   }
   return ptr;
}

/*Allocate memory for a length m array of ints and initialize to val*/
bool* vector_bool(int m, bool val){

   bool *ptr = (bool*)malloc(m*sizeof(bool));
   int i;
   for(i=0;i<m;i++){
      ptr[i] = val;
   }
   return ptr;
}

/*Allocate memory for a length m array of ints and initialize to val*/
int* vector_int(int m, int val){

   int *ptr = (int*)malloc(m*sizeof(int));
   int i;
   for(i=0;i<m;i++){
      ptr[i] = val;
   }
   return ptr;
}

/*Allocate memory for a length m array of doubles and initialize to val*/
double* vector_double(int m, double val){

   double *ptr = (double*)malloc(m*sizeof(double));
   int i;
   for(i=0;i<m;i++){
      ptr[i] = val;
   }
   return ptr;
}

/*Allocate memory for a mxn array of doubles and initialize to val*/
double** array_double(int m, int n, double val){

   double **ptr = (double**)malloc(m*sizeof(double*));
   ptr[0] = (double*)malloc(m*n*sizeof(double));
   int i,j;
   for(i=0;i<m;i++){
      ptr[i] = ptr[0] + n*i;
      for(j=0;j<n;j++){
         ptr[i][j] = val;
      }
   }
   return ptr;
}
