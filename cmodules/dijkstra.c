/* dijkstra.c: C code for acceleration of graphpy library 
 *
 * *  Author: Jeff Calder, 2020.
 *
 */


#include <stdio.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"
#include "dijkstra.h"

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

void dijkstra_main(double *d, int *l, int *WI, int *K, double *WV, int *I, bool prog, int n, int M, int k){


   //Initialization
   int i,j,jj;
   int s = 0;                       //Size of heap
   int *h = vector_int(n+1,-1);     //Active points heap (indices of active points)
   bool *A = vector_bool(n,0);      //Active flag
   int *p = vector_int(n,-1);       //Pointer back to heap
   bool *V = vector_bool(n,0);      //Finalized flag

   //Build active points heap and set distance = 0 for initial points
   for(i=0; i<k; i++){
      PushHeap(d,h,s,p,I[i]);
      s++;
      d[I[i]] = 0;      //Initialize distance to zero
      A[I[i]] = 1;      //Set active flag to true
      l[I[i]] = I[i];   //Set index of closest label
   }
   
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
            if(A[j]){  //If j is already active
               double tmp_dist = d[i] + WV[j];
               if(tmp_dist < d[j]){ //Need to update heap
                  d[j] = tmp_dist;
                  SiftUp(d,h,s,p,p[j]);
                  l[j] = l[i];
               }
            }else{ //If j is not active
               //Add to heap and initialize distance, active flag, and label index
               PushHeap(d,h,s,p,j);
               s++;
               d[j] = d[i] + WV[j];
               A[j] = 1;  
               l[j] = l[i];
            }
         }
      }
   }
}
