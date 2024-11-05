/* lplearn.c - C code acceleration for Lplearning
 *
 *
 *  Used by the graphpy package
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


#include <stdio.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"

//Main subroutine for iteration
void lp_iterate_main(double *uu, double *ul, int *I, int *J, double *W, int *ind, double *val, double p, int T, double tol, bool prog, int n, int M, int m){

   int i, j, it;
   
   double alpha = 1/p;
   double delta = 1-2/p;
   double dt = 0.9/(alpha + 2*delta);

   //Compute number of neighbors of each vertex and vertex degrees
   int *num = vector_int(n,0);    //number of neighbors
   int *start = vector_int(n,0);    //start of neighbor
   double *invdeg = vector_double(n,0); //Degrees
   j=0;
   for(i=0;i<n;i++){
      start[i] = j;
      invdeg[i] = 0;
      while((J[j]==i) & (j < M)){
         num[i]++;
         invdeg[i] += W[j];
         j++;
      }
      invdeg[i] = alpha/invdeg[i];
   }

   //Normalize weight matrix 
   double maxWGT = 0;
   for(i=0;i<M;i++)
      maxWGT = MAX(maxWGT,W[i]);
   dt = dt/maxWGT;
   /*
   for(i=0;i<M;i++)
      W[i] = W[i]/maxWGT;*/


   double *vu = vector_double(n,0);    
   double *vl = vector_double(n,0);    
   double *temp;
   
   //Main loop
   for(it=0;it<T;it++){
      if(prog){
         printf("Iter=%d, ",it); fflush(stdout); 
      }
      double err = 0;
      for(i=0;i<n;i++){
         //Upper function
         double minw=0, maxw=0, sumw=0;
         for(j=start[i];j<start[i]+num[i];j++){ //loop over neighbors
            minw = MIN(W[j]*(uu[I[j]]-uu[i]),minw);
            maxw = MAX(W[j]*(uu[I[j]]-uu[i]),maxw);
            sumw += W[j]*(uu[I[j]]-uu[i]);
         }
         vu[i] = uu[i] + dt*(invdeg[i]*sumw + delta*(minw + maxw));

         //Lower function
         minw=0; maxw=0; sumw=0;
         for(j=start[i];j<start[i]+num[i];j++){ //loop over neighbors
            minw = MIN(W[j]*(ul[I[j]]-ul[i]),minw);
            maxw = MAX(W[j]*(ul[I[j]]-ul[i]),maxw);
            sumw += W[j]*(ul[I[j]]-ul[i]);
         }
         vl[i] = ul[i] + dt*(invdeg[i]*sumw + delta*(minw + maxw));
         err = MAX(uu[i] - ul[i],err);
      }

      //Progress update
      if(prog){
         printf("err=%.15f\n",err); fflush(stdout);
      }

      //Set Dirichlet conditions
      for(j=0;j<m;j++){
         i = ind[j];
         vu[i] = val[j];
         vl[i] = val[j];
      }

      //Check error condition
      if(err < tol && it > 10)
         break;

      //Swap pointers
      temp = uu; 
      uu = vu;
      vu = temp;

      temp = ul; 
      ul = vl;
      vl = temp;
   }
}


//Main subroutine for iteration
void lip_iterate_main(double *u, int *I, int *J, double *W, int *ind, double *val, int T, double tol, bool prog, int n, int M, int m, double alpha, double beta){

   int i, j, it;
   
   //printf("Unweighted\n");
   //Compute number of neighbors of each vertex and vertex degrees
   int *num = vector_int(n,0);    //number of neighbors
   int *start = vector_int(n,0);    //start of neighbor
   bool *label_mask = vector_bool(n,true);
   j=0;
   for(i=0;i<n;i++){
      start[i] = j;
      while((J[j]==i) & (j < M)){
         num[i]++;
         j++;
      }
   }
  
   //Set Dirichlet conditions and label mask
   for(j=0;j<m;j++){
      u[ind[j]] = val[j];
      label_mask[ind[j]] = false;
   }
   
   //Main loop
   for(it=0;it<T;it++){
      if(prog){
         printf("Iter=%d, ",it); fflush(stdout); 
      }
      double err = 0;
      for(i=0;i<n;i++){

         //Update if not a label
         if(label_mask[i]){
            double minu=u[I[start[i]]], maxu=u[I[start[i]]];
            double sumu = 0.0;
            double deg = 0.0;
            for(j=start[i];j<start[i]+num[i];j++){ //loop over neighbors
               sumu += W[j]*u[I[j]];
               deg += W[j];
               minu = MIN(u[I[j]],minu);
               maxu = MAX(u[I[j]],maxu);
            }
            double ne = alpha*sumu/deg + beta*(minu + maxu)/2;
            err = MAX(ABS(u[i] - ne),err);
            u[i] = ne;
         }
      }

      //Progress update
      if(prog){
         printf("err=%.15f\n",err); fflush(stdout);
      }

      //Check error condition
      if(err < tol && it > 20)
         break;
   }
}

//Main subroutine for iteration
void lip_iterate_weighted_main(double *u, int *I, int *J, double *W, int *ind, double *val, int T, double tol, bool prog, int n, int M, int m){

   int i, j, k, it;
   
   //printf("Weighted\n");
   //Compute number of neighbors of each vertex and vertex degrees
   int *num = vector_int(n,0);    //number of neighbors
   int *start = vector_int(n,0);    //start of neighbor
   bool *label_mask = vector_bool(n,true);
   j=0;
   for(i=0;i<n;i++){
      start[i] = j;
      while((J[j]==i) & (j < M)){
         num[i]++;
         j++;
      }
   }
  
   //Set Dirichlet conditions and label mask
   for(j=0;j<m;j++){
      u[ind[j]] = val[j];
      label_mask[ind[j]] = false;
   }
   
   //Main loop
   for(it=0;it<T;it++){
      if(prog){
         printf("Iter=%d, ",it); fflush(stdout); 
      }
      double err = 0;
      for(i=0;i<n;i++){

         //Update if not a label
         if(label_mask[i]){
            double minu=u[I[start[i]]], maxu=u[I[start[i]]];
            for(j=start[i];j<start[i]+num[i];j++){ //loop over neighbors
               minu = MIN(u[I[j]],minu);
               maxu = MAX(u[I[j]],maxu);
            }
            //Bisection search
            double a=minu, b=maxu;
            for(k=0;k<30;k++){ 
               double minw=0, maxw=0;
               double t = (a + b)/2.0;
               for(j=start[i];j<start[i]+num[i];j++){ //loop over neighbors
                  minw = MIN(W[j]*(t - u[I[j]]),minw);
                  maxw = MAX(W[j]*(t - u[I[j]]),maxw);
               }
               double inflap = minw + maxw;
               if(inflap > 0)
                  b = t;
               else
                  a = t;
            }
            double ne = (a + b)/2.0;
            err = MAX(ABS(u[i] - ne),err);
            u[i] = ne;
         }
      }

      //Progress update
      if(prog){
         printf("err=%.15f\n",err); fflush(stdout);
      }

      //Check error condition
      if(err < tol && it > 20)
         break;
   }
}


