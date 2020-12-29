#ifndef MAJ_DIJKSTRA_H
#define MAJ_DIJKSTRA_H
#define key_type distance_type

#include "maj_implicit_heap.h"
#include <float.h>



/*static void single_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, unsigned int dstride){
    
    unsigned int i;
    
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        distance_type eWeight=edgeWeights[i];
        distance_type current=distances[index*dstride];
        distance_type possible=eWeight+minDist;
        if(possible<current){
            distances[index*dstride]=possible;
            unsigned int location=heap->locations[index];
            decrease_key(heap,location,possible);
        }
        
    }
    delete_min(heap);
    
}

static void dijkstra_distances(int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances,  int pcount,  int dstride){
    
    implicit_heap heap=create_heap_with_batch(distances,pcount, dstride);
    
    
    unsigned int iter=0;
    
    
    // printf("added\n");
    while(!empty(&heap)){
        
        
        single_dijkstra_iteration(&heap, indicies, edgeWeights, nnCounts, distances, dstride);
        iter++;
        
    }
    
    free_heap(&heap);

}*/

static void labeled_single_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, unsigned int dstride){
    
    int i;
    
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        distance_type eWeight=edgeWeights[i];
        distance_type current=distances[index*dstride];
        distance_type possible=eWeight+minDist;
        if(possible<current){
            distances[index*dstride]=possible;
            labels[index]=labels[minPosition];
            if(current==FLT_MAX){
                insert_node(heap, index, possible);
            }else{
                unsigned int location=heap->locations[index];
                decrease_key(heap,location,possible);
            }
           
        }
        
    }
    delete_min(heap);
    
}




static void labeled_dijkstra(int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, int pcount,  int dstride){
    
   int i;
    //int sq=sqrt(pcount);
    //implicit_heap heap=create_heap_with_batch(distances,pcount, dstride);
    implicit_heap heap=create_empty_heap_with_locations(pcount);
    for(i=0;i<pcount;i++){
        if(distances[i*dstride]!=FLT_MAX){
            insert_node(&heap,i,distances[i*dstride]);
        }
    }
    
    unsigned int iter=0;
    
    
    // printf("added\n");
    while(!empty(&heap)){
        
        
        labeled_single_dijkstra_iteration(&heap, indicies, edgeWeights, nnCounts, distances, labels, dstride);
        iter++;
        
    }
    
    
    free_heap(&heap);
}


/*static void k_centers_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, unsigned char *seen){
    
    unsigned int i;
    
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        distance_type eWeight=edgeWeights[i];
        distance_type current=distances[index];
        distance_type possible=eWeight+minDist;
        if(possible<current){
            distances[index]=possible;
            labels[index]=labels[minPosition];
            if(!seen[index]){
                insert_node(heap, index, possible);
                seen[index]=1;
            }else{
                unsigned int location=heap->locations[index];
                decrease_key(heap,location,possible);
            }
            
        }
        
    }
    delete_min(heap);
    
}

static void k_centers_initialization(int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, int pcount, int lcount){
    
    implicit_heap heap=create_empty_heap_with_locations(pcount);
    for(int i=0;i<pcount;i++){
            distances[i]=FLT_MAX;
    }
    for(int l=0;l<lcount;l++){
        unsigned char *seen=calloc(pcount,1);
        if(l==0){
            int seed=rand()%pcount;
            seen[seed]=1;
            distances[seed]=0;
            labels[seed]=l;
            insert_node(&heap,seed,distances[seed]);
        }else{
            int maxIndex=0;
            float max=0;
            for(int i=0;i<pcount;i++){
                if(distances[i]>max){
                    max=distances[i];
                    maxIndex=i;
                }
            }
            int seed=maxIndex;
            seen[seed]=1;
            distances[seed]=0;
            labels[seed]=l;
            insert_node(&heap,seed,distances[seed]);
        }
        while(!empty(&heap)){
            k_centers_dijkstra_iteration(&heap, indicies, edgeWeights, nnCounts, distances, labels, seen);
        }
        free(seen);
    }
    
    free_heap(&heap);
}
*/


void internal_distances_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels){
    int i;
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        if(labels[index]==labels[minPosition]){
            distance_type eWeight=edgeWeights[i];
            distance_type current=distances[index];
            distance_type possible=eWeight+minDist;
            if(possible<current){
                distances[index]=possible;
                if(current==FLT_MAX){
                    insert_node(heap, index, possible);
                }else{
                    unsigned int location=heap->locations[index];
                    decrease_key(heap,location,possible);
                }
            }
        }
    }
    delete_min(heap);
}



void calc_internal_distances(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, int pcount, int lcount){
    
    int i;
    clear_heap(heap);
    for(i=0;i<pcount;i++){
        if(distances[i]<FLT_MAX){
            insert_node(heap,i,distances[i]);
        }
    }
    while(!empty(heap)){
        internal_distances_dijkstra_iteration(heap, indicies, edgeWeights, nnCounts, distances, labels);
    }
}





#endif






