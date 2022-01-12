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
struct NearestNeighbors {
   double w;
   double u;
};
//Comparison function for NearestNeighbors
int compare (const void * a, const void * b)
{
    return ( (*(struct NearestNeighbors*)b).u - (*(struct NearestNeighbors*)a).u );
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

void dijkstra_main(double *d, int *l, int *WI, int *K, double *WV, int *I, double *g, bool prog, int n, int M, int k, double max_dist){


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
               double tmp_dist = d[i] + WV[jj];
               if(tmp_dist < d[j]){ //Need to update heap
                  d[j] = tmp_dist;
                  SiftUp(d,h,s,p,p[j]);
                  l[j] = l[i];
               }
            }else{ //If j is not active
               //Add to heap and initialize distance, active flag, and label index
               d[j] = d[i] + WV[jj];
               A[j] = 1;  
               l[j] = l[i];
               PushHeap(d,h,s,p,j);
               s++;
            }
         }
      }
   }
}
/*double stencil_solver(double ui, double *u, double *w, int n, double p){

   double min_val = u[0];
   double max_val = u[0];
   double min_w = w[0];
   for(int i=0; i<n; i++){
      min_val = MIN(u[i],min_val);
      max_val = MAX(u[i],max_val);
      min_w = MIN(w[i],min_w);
   }

   //Initial bounds for bisection
   double a = min_val;
   double b;
   if(p==1)
      b = max_val + (1.0/(double)n)/min_w;
   else
      b = max_val + pow(1.0/(double)n,1.0/p)/min_w;

   while(b-a > 1e-5){
      double f = 0.0;
      double t = (a+b)/2.0;
      for(int i=0; i<n; i++){
         double v = MAX(w[i]*(t-u[i]),0);
         if(p!=1)
            v = pow(v,p);
         f+=v;
      }
      if(f > 1)
         b = t;
      else
         a = t;
   }

   double sol = (a+b)/2.0;
   double f = 0.0;
   for(int i=0; i<n; i++)
      f += pow(MAX(w[i]*(sol-u[i]),0),p);
   if(ABS(f - 1) > 1e-2)
      printf("f=%f\n",f);

   return sol;
}*/
double stencil_solver(double ui, double *u, double *w, int n, double p){

   double min_val = u[0];
   double max_val = u[0];
   double max_w = w[0];
   for(int i=0; i<n; i++){
      min_val = MIN(u[i],min_val);
      max_val = MAX(u[i],max_val);
      max_w = MAX(w[i],max_w);
   }
   max_w=1;

   //Initial bounds for bisection
   double a = min_val;
   double b;
   if(p==1)
      b = max_val + max_w/(double)n;
   else
      b = max_val + pow(max_w/(double)n,1.0/p);

   while(b-a > 1e-5){
      double f = 0.0;
      double t = (a+b)/2.0;
      for(int i=0; i<n; i++){
         double v = MAX(t-u[i],0);
         if(p!=1)
            v = pow(v,p);
         f+=v;
      }
      if(f > max_w)
         b = t;
      else
         a = t;
   }

   double sol = (a+b)/2.0;

   /*double f = 0.0;
   for(int i=0; i<n; i++)
      f += pow(MAX(sol-u[i],0),p);
   printf("f=%f\n",f-max_w);*/

   return sol;
}

void HJsolver_fmm(double *d, int *l, int *WI, int *K, double *WV, int *I, int *g, bool prog, int n, int M, int k, double p_val){


   //Initialization
   int i,j,ii,jj,kk;
   int s = 0;                       //Size of heap
   int *h = vector_int(n+1,-1);     //Active points heap (indices of active points)
   bool *A = vector_bool(n,0);      //Active flag
   int *p = vector_int(n,-1);       //Pointer back to heap
   bool *V = vector_bool(n,0);      //Finalized flag
   double *u_vals = vector_double(n,0);
   double *w_vals = vector_double(n,0);

   //Invert distances
   /*for(i=0; i<M; i++)
      WV[i] = 1.0;///WV[i];*/

   //Initialize to infinity
   for(i=0; i<n; i++)
      d[i] = INFINITY;

   //Build active points heap and set distance = 0 for initial points
   for(i=0; i<k; i++){
      s++;
      d[I[i]] = g[i];   //Initialize distance to zero
      A[I[i]] = 1;      //Set active flag to true
      l[I[i]] = I[i];   //Set index of closest label
      PushHeap(d,h,s,p,I[i]);
   }
   
   int num_nn;
   //Dijkstra's algorithm 
   while(s > 0){
      i = PopHeap(d,h,s,p); //Pop smallest element off of heap
      s--;

      //Finalize this point
      V[i] = 1;  //Mark as finalized
      A[i] = 0;  //Set active flag to false
      
      //Update neighbors
      for(jj=K[i]; jj < K[i+1]; jj++){
         j = WI[jj];
         if(j != i && V[j] == 0){
            num_nn = 0;
            for(ii=K[j]; ii < K[j+1]; ii++){
               kk = WI[ii];
               //Only grab neighbors that are not infinity
               //if(d[kk] < INFINITY && V[kk]){
               if(V[kk]){
                  u_vals[num_nn] = d[kk];
                  w_vals[num_nn] = WV[kk];
                  //w_vals[num_nn] = 1;
                  num_nn++;
               }
            }
            double tmp = stencil_solver(d[j],u_vals,w_vals,num_nn,p_val);
            if(A[j]){  //If j is already active
               if(tmp < d[j]){ //Need to update heap
                  d[j] = tmp;
                  l[j] = l[i];
                  SiftUp(d,h,s,p,p[j]);
               }
            }else{ //If j is not active
               //Add to heap and initialize distance, active flag, and label index
               s++;
               d[j] = tmp;
               A[j] = 1;  
               l[j] = l[i];
               PushHeap(d,h,s,p,j);
            }
         }
      }
   }

   bool *Blah = vector_bool(n,0);     
   for(i=0; i<k; i++)
      Blah[I[i]] = 1;    

   for(i=0; i<n; i++){
      if(V[i]==0)
         printf("Someone not visited!\n");
   }
   double err = 0;
   for(i=0; i<n; i++){
      double f = 0.0;
      double max_w = -1.0;
      for(ii=K[i]; ii < K[i+1]; ii++){
         kk = WI[ii];
         double v = MAX(d[i]-d[kk],0);
         max_w = MAX(max_w,WV[kk]);
         //double v = MAX(d[i]-d[kk],0);
         if(p_val!=1)
            v = pow(v,p_val);
         f+=v;
      }

      max_w=1;
      if(Blah[i]==0){
         if(ABS(f - max_w) > 1e-2){
            printf("diff=%f\n",ABS(f - max_w));
         }
         err = MAX(ABS(f-max_w),err);
      }
   }
   //printf("err=%f\n",err);

}

void HJsolver_jacobi(double *d, int *l, int *WI, int *K, double *WV, int *I, int *g, bool prog, int n, int M, int k, double p_val){


   //Initialization
   int i,j,ii,kk;
   bool *A = vector_bool(n,1);      //Indicates labeled nodes
   double *u_vals = vector_double(n,0);
   double *w_vals = vector_double(n,0);

   //Invert distances
   /*for(i=0; i<M; i++)
      WV[i] = 1.0/WV[i];*/

   //Initialize to zero
   for(i=0; i<n; i++)
      d[i] = INFINITY;

   //Set mask for labeled nodes
   for(i=0; i<k; i++){
      d[I[i]] = g[i];   //Initialize distance
      A[I[i]] = 0;      //Set flag
   }
   

   //Iteration for solving
   int T = 0;
   double err = 1;
   while(T++ < 1e5 && err > 1e-2){
      for(j=0; j<n; j++){
         if(A[j]){      
            //Grab neighbors
            int num_nn = 0;
            for(ii=K[j]; ii < K[j+1]; ii++){
               kk = WI[ii];
               if(d[kk] < INFINITY){
                  u_vals[num_nn] = d[kk];
                  w_vals[num_nn] = WV[ii];
                  num_nn++;
               }
            }
            if(num_nn>0)
               d[j] = stencil_solver(d[j],u_vals,w_vals,num_nn,p_val);
         }
      }

      err = 0;
      for(j=0; j<n; j++){
         if(A[j]){
            double f = 0.0;
            double max_w = -1.0;
            for(ii=K[j]; ii < K[j+1]; ii++){
               kk = WI[ii];
               max_w = MAX(max_w,WV[kk]);
               double v = MAX(d[j]-d[kk],0);
               if(p_val!=1)
                  v = pow(v,p_val);
               f+=v;
            }
            max_w=1;
            err = MAX(ABS(f-max_w),err);
         }
      }
      //printf("err=%f\n",err);
   }
   //printf("T=%d\n",T);
}

double peikonal_solver(double *u, double *w, double f, int n, double p, int num_bisection_it){

   int i,j;
   double min_val = u[0];
   double max_val = u[0];
   double degree = 0;
   for(i=0; i<n; i++){
      min_val = MIN(u[i],min_val);
      max_val = MAX(u[i],max_val);
      degree += w[i];
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
         double v = MAX(t-u[i],0);
         if(p!=1)
            v = pow(v,p);
         op+=w[i]*v;
      }
      if(op > f)
         b = t;
      else
         a = t;
   }
   return (a+b)/2.0;
}

double peikonal_solver_fast(double *u, double *w, double f, int n){

   int i,j;
   double min_val = u[0];
   double max_val = u[0];
   double degree = 0;
   for(i=0; i<n; i++){
      min_val = MIN(u[i],min_val);
      max_val = MAX(u[i],max_val);
      degree += w[i];
   }

   //Initial bounds for bisection
   double inc = f/degree;
   double a = min_val + inc;
   double b = max_val + inc;

   for(j=0; j<30; j++){
      double op = 0.0;
      double t = (a+b)/2.0;
      for(i=0; i<n; i++){
         double v = MAX(t-u[i],0);
         op+=w[i]*v;
      }
      if(op > f)
         b = t;
      else
         a = t;
   }
   return (a+b)/2.0;
}


void peikonal_main(double *u, int *WI, int *K, double *WV, int *I, double *f,  double *g, double p_val, int max_num_it, double tol, int num_bisection_it, bool prog, int n, int M, int k){


   //Initialization
   int i,j,ii,kk;
   bool *A = vector_bool(n,1);      //Indicates labeled nodes
   double *u_vals = vector_double(n,0);
   double *w_vals = vector_double(n,0);

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
               u_vals[num_nn] = u[kk];
               w_vals[num_nn] = WV[ii];
               num_nn++;
            }
            if(num_nn>0){

               if(p_val == 1)
                  newu = peikonal_solver_fast(u_vals,w_vals,f[j],num_nn);
               else
                  newu = peikonal_solver(u_vals,w_vals,f[j],num_nn,p_val,num_bisection_it);

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
}


