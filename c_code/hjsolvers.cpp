/* dijkstra.c: C code for acceleration of graphlearning library 
 *
 * *  Author: Jeff Calder, 2020.
 *
 */


#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "vector_operations.h"
#include "memory_allocation.h"
#include "hjsolvers.h"

//Strucure to hold the weights and values of nearest neighbors
typedef struct {
   double w;
   double u;
} NearestNeighbors;

//Comparison function for NearestNeighbors
int NNcompare (const void * a, const void * b)
{
    double da = ((NearestNeighbors*)a)->u;
    double db = ((NearestNeighbors*)b)->u;
    return (da > db) - (da < db);
}
//Heap functions
//d = values in heap (indexed by graph vertex)
//h = heap (contains indices of graph elements in heap)
//p = pointers from graph back to heap (are updated with heap operations)
//s = number of elements in heap

//Sift up
//i = heap index of element to be sifted up
void SiftUp(double *d, int *h, int s, int *p, int i){
   
   int pi = (int)(i/2);  //Parent index in heap
   while(pi != 0){
      if(d[h[pi]] > d[h[i]]){  //If parent larger, then swap
         //Swap in heap
         int tmp = h[pi];
         h[pi] = h[i];
         h[i] = tmp;

         //Update pointers to heap
         p[h[i]] = i;
         p[h[pi]] = pi;

         //Update parent/child indices
         i = pi;
         pi = (int)(i/2);
      }else{
         pi = 0;
      }
   }
}
            
//Sift down
//i = heap index of element to be sifted down
void SiftDown(double *d, int *h, int s, int *p, int i){
   
   int ci = 2*i;  //child index in heap
   while(ci <= s){
      if(d[h[ci+1]] < d[h[ci]] && ci+1 <= s)  //Choose smallest child
         ci++;
      if(d[h[ci]] < d[h[i]]){  //If child smaller, then swap
         //Swap in heap
         int tmp = h[ci];
         h[ci] = h[i];
         h[i] = tmp;

         //Update pointers to heap
         p[h[i]] = i;
         p[h[ci]] = ci;

         //Update parent/child indices
         i = ci;
         ci = 2*i;
      }else{
         ci = s+1;
      }
   }
}

//Pop smallest off of heap
//Returns index of smallest and size of new heap
int PopHeap(double *d, int *h, int s, int *p){
    
   //Index of smallest in heap
   int i = h[1];

   //Put last element on top of heap
   h[1] = h[s];

   //Update pointer
   p[h[1]] = 1;

   //Sift down the heap
   SiftDown(d,h,s-1,p,1);

   return i;
     
} 

//Push element onto heap
//i = Graph index to add to heap
void PushHeap(double *d, int *h, int s, int *p, int i){

   h[s+1] = i;  //add to heap at end
   p[i] = s+1;  //Update pointer to heap
   SiftUp(d,h,s+1,p,s+1);

}

void dijkstra_hl_main(double *d, int *l, int *WI, int *K, double *WV, int *I, double *g, double *f, bool prog, int n, int M, int k, double max_dist){


   //Initialization
   int i,j,jj;
   int s = 0;                       //Size of heap
   int *h = vector_int(n+1,-1);     //Active points heap (indices of active points)
   bool *A = vector_bool(n,0);      //Active flag
   int *p = vector_int(n,-1);       //Pointer back to heap
   bool *V = vector_bool(n,0);      //Finalized flag

   //Build active points heap and set distance = 0 for initial points
   for(i=0; i<k; i++){
      d[I[i]] = g[i];   //Initialize distance to g
      A[I[i]] = 1;      //Set active flag to true
      l[I[i]] = I[i];   //Set index of closest label
      PushHeap(d,h,s,p,I[i]);
      s++;
   }
   
   //Dijkstra's algorithm 
   while(s > 0){
      i = PopHeap(d,h,s,p); //Pop smallest element off of heap
      s--;

      //Finalize this point
      V[i] = 1;  //Mark as finalized
      A[i] = 0;  //Set active flag to false
      if(d[i] > max_dist)
         break;

      //Update neighbors
      for(jj=K[i]; jj < K[i+1]; jj++){
         j = WI[jj];
         if(j != i && V[j] == 0){
            //double tmp_dist = d[i] + WV[jj]*f[i];
            double fwij = f[i]*WV[jj];
            double tmp_dist = ( fwij + sqrt(fwij*fwij + 4*d[i]*d[i]) )/2.0;
            if(A[j]){  //If j is already active
               if(tmp_dist < d[j]){ //Need to update heap
                  d[j] = tmp_dist;
                  SiftUp(d,h,s,p,p[j]);
                  l[j] = l[i];
               }
            }else{ //If j is not active
               //Add to heap and initialize distance, active flag, and label index
               d[j] = tmp_dist;
               A[j] = 1;  
               l[j] = l[i];
               PushHeap(d,h,s,p,j);
               s++;
            }
         }
      }
   }
}
void dijkstra_main(double *d, int *l, int *WI, int *K, double *WV, int *I, double *g, double *f, bool prog, int n, int M, int k, double max_dist){


   //Initialization
   int i,j,jj;
   int s = 0;                       //Size of heap
   int *h = vector_int(n+1,-1);     //Active points heap (indices of active points)
   bool *A = vector_bool(n,0);      //Active flag
   int *p = vector_int(n,-1);       //Pointer back to heap
   bool *V = vector_bool(n,0);      //Finalized flag

   //Build active points heap and set distance = 0 for initial points
   for(i=0; i<k; i++){
      d[I[i]] = g[i];   //Initialize distance to g
      A[I[i]] = 1;      //Set active flag to true
      l[I[i]] = I[i];   //Set index of closest label
      PushHeap(d,h,s,p,I[i]);
      s++;
   }

  
   //Dijkstra's algorithm 
   while(s > 0){
      i = PopHeap(d,h,s,p); //Pop smallest element off of heap
      s--;

      //Finalize this point
      V[i] = 1;  //Mark as finalized
      A[i] = 0;  //Set active flag to false
      if(d[i] > max_dist)
         break;

      //Update neighbors
      for(jj=K[i]; jj < K[i+1]; jj++){
         j = WI[jj];
         if(j != i && V[j] == 0){
            if(A[j]){  //If j is already active
               double tmp_dist = d[i] + WV[jj]*f[i];
               if(tmp_dist < d[j]){ //Need to update heap
                  d[j] = tmp_dist;
                  SiftUp(d,h,s,p,p[j]);
                  l[j] = l[i];
               }
            }else{ //If j is not active
               //Add to heap and initialize distance, active flag, and label index
               d[j] = d[i] + WV[jj]*f[i];
               A[j] = 1;  
               l[j] = l[i];
               PushHeap(d,h,s,p,j);
               s++;
            }
         }
      }
   }
}

double peikonal_solver(NearestNeighbors *neighbors, double f, int n, double p, int num_bisection_it){

   int i,j;
   double min_val = neighbors[0].u;
   double max_val = min_val;
   double degree = 0;
   for(i=0; i<n; i++){
      min_val = MIN(neighbors[i].u,min_val);
      max_val = MAX(neighbors[i].u,max_val);
      degree += neighbors[i].w;
   }

   //Initial bounds for bisection
   double inc = f/degree;
   if(p>1)
      inc = pow(inc,1.0/p);
   double a = min_val + inc;
   double b = max_val + inc;

   for(j=0; j<num_bisection_it; j++){
      double op = 0.0;
      double t = (a+b)/2.0;
      for(i=0; i<n; i++){
         double v = MAX(t-neighbors[i].u,0);
         if(p!=1)
            v = pow(v,p);
         op += v*neighbors[i].w;
      }
      if(op > f)
         b = t;
      else
         a = t;
   }
   return (a+b)/2.0;
}

double peikonal_solver_fast(NearestNeighbors *neighbors, double f, int n){

   int k;
   double weighted_sum, deg, t;

   //Sort neighbors by values of u
   qsort(neighbors, n, sizeof(NearestNeighbors), NNcompare);
   neighbors[n].u = neighbors[n-1].u + f/neighbors[n-1].w + 1;  //Upper bound
  
   weighted_sum = neighbors[0].u * neighbors[0].w;
   deg = neighbors[0].w;
   t = (f + weighted_sum)/deg;
   k = 0;
   while(t > neighbors[k+1].u){
      k++;
      weighted_sum += neighbors[k].u * neighbors[k].w;
      deg += neighbors[k].w;
      t = (f + weighted_sum)/deg;
   }

   return t;
}


void peikonal_main(double *u, int *WI, int *K, double *WV, int *I, double *f,  double *g, double p_val, int max_num_it, double tol, int num_bisection_it, bool prog, int n, int M, int k){


   //Initialization
   int i,j,ii,kk;
   bool *A = vector_bool(n,1);      //Indicates labeled nodes
   NearestNeighbors *neighbors = (NearestNeighbors*)malloc((n+1)*sizeof(NearestNeighbors));

   //Set mask for labeled nodes
   for(i=0; i<k; i++){
      u[I[i]] = g[i];   //Initialize distance
      A[I[i]] = 0;      //Set flag
   }
   
   //Iteration for solving
   int T = 0;
   double err = tol+1;
   double newu;
   while(T++ < max_num_it && err > tol){
      err = 0;
      for(j=0; j<n; j++){
         if(A[j]){      
            //Grab neighbors
            int num_nn = 0;
            for(ii=K[j]; ii < K[j+1]; ii++){
               kk = WI[ii];
               neighbors[num_nn].u = u[kk];
               neighbors[num_nn].w = WV[ii];
               num_nn++;
            }
            if(num_nn>0){

               if(p_val == 1)
                  newu = peikonal_solver_fast(neighbors,f[j],num_nn);
               else
                  newu = peikonal_solver(neighbors,f[j],num_nn,p_val,num_bisection_it);

               err = MAX(ABS(newu - u[j]),err);
               u[j] = newu;
            }else{
               printf("Warning: Some points have no neighbors!\n");
            }
         }
      }
      if(prog)
         printf("T=%d, err=%f\n",T,err);
   }

   //Free memory
   free(neighbors);
   free(A);
}

void peikonal_fmm_main(double *u, int *WI, int *K, double *WV, int *I, double *f,  double *g, double p_val, int num_bisection_it, int n, int M, int k){

   //Initialization
   int i,j,jj,ii,kk;
   int s = 0;                       //Size of heap
   int *h = vector_int(n+1,-1);     //Active points heap (indices of active points)
   bool *A = vector_bool(n,0);      //Active flag
   int *p = vector_int(n,-1);       //Pointer back to heap
   bool *V = vector_bool(n,0);      //Finalized flag
   bool *L = vector_bool(n,1);      //Indicates labeled nodes
   NearestNeighbors *neighbors = (NearestNeighbors*)malloc((n+1)*sizeof(NearestNeighbors));

   //Build active points heap and set distance = g for initial points
   for(i=0; i<k; i++){
      u[I[i]] = g[i];   //Initialize distance to g
      A[I[i]] = 1;      //Set active flag to true
      L[I[i]] = 0;      //Set flag for labeled points
      PushHeap(u,h,s,p,I[i]);
      s++;
   }
  
   //Dijkstra's algorithm 
   while(s > 0){
      i = PopHeap(u,h,s,p); //Pop smallest element off of heap
      s--;

      //Finalize this point
      V[i] = 1;  //Mark as finalized
      A[i] = 0;  //Set active flag to false

      //Update neighbors
      for(jj=K[i]; jj < K[i+1]; jj++){
         j = WI[jj];
         if(j != i && V[j] == 0 && L[j]){
            //Grab neighbors
            int num_nn = 0;
            double tmp_dist = u[j];
            for(ii=K[j]; ii < K[j+1]; ii++){
               kk = WI[ii];
               if(kk!=j && (A[kk] | V[kk])){
                  neighbors[num_nn].u = u[kk];
                  neighbors[num_nn].w = WV[ii];
                  num_nn++;
               }
            }
            if(num_nn>0){
               if(p_val == 1)
                  tmp_dist = peikonal_solver_fast(neighbors,f[j],num_nn);
               else
                  tmp_dist = peikonal_solver(neighbors,f[j],num_nn,p_val,num_bisection_it);
            }else{
               printf("Warning: Some points have no neighbors!\n");
            } 

            if(A[j]){  //If j is already active
               if(tmp_dist < u[j]){ //Need to update heap
                  u[j] = tmp_dist;
                  SiftUp(u,h,s,p,p[j]);
               }
            }else{ //If j is not active

               //Add to heap and initialize distance, active flag, and label index
               u[j] = tmp_dist;
               A[j] = 1;  
               PushHeap(u,h,s,p,j);
               s++;
            }
         }
      }
   }

   //Free memory
   free(neighbors);
   free(L);
   free(h);
   free(A);
   free(p);
   free(V);
}

