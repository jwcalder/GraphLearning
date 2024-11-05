#include <stdlib.h>
#ifndef BRANCHING_NUMBER
#define BRANCHING_NUMBER 8
#endif
#ifndef MAJ_SIMPLE_IMPLICIT_HEAP
#define MAJ_SIMPLE_IMPLICIT_HEAP

typedef struct{
    k_type key;
    int originalIndex;
}s_heap_node;

typedef struct{
    s_heap_node *root;
    int count;
}s_heap;



static void s_heap_swap_nodes(s_heap *heap,int nodeIndex1, int nodeIndex2){
    int i1=heap->root[nodeIndex1].originalIndex;
    int i2=heap->root[nodeIndex2].originalIndex;
    k_type key1=heap->root[nodeIndex1].key;
    k_type key2=heap->root[nodeIndex2].key;
    heap->root[nodeIndex1].originalIndex=i2;
    heap->root[nodeIndex2].originalIndex=i1;
    heap->root[nodeIndex1].key=key2;
    heap->root[nodeIndex2].key=key1;
}

static void s_heap_add_node_to_bottom(s_heap *heap, int originalIndex, k_type key){
    
    int count=heap->count;
    heap->root[count].originalIndex=originalIndex;
    heap->root[count].key=key;
    heap->count++;
}

static void s_heap_bubble_up(s_heap *heap, int nodeIndex){
    
    while (nodeIndex>0) {
        k_type myKey=heap->root[nodeIndex].key;
        int parentIndex=nodeIndex/BRANCHING_NUMBER-((nodeIndex%BRANCHING_NUMBER)==0);
        k_type parentKey=heap->root[parentIndex].key;
        if(myKey<parentKey){
            s_heap_swap_nodes(heap,nodeIndex,parentIndex);
            nodeIndex=parentIndex;
        }else{
            break;
        }
       
    }
}

static void s_heap_push_down(s_heap *heap, int nodeIndex){
    
    int i;
    int count=heap->count;
    int childIndex=nodeIndex*BRANCHING_NUMBER+1;
    while(childIndex<count){
        int minIndex=childIndex;
        k_type myKey=heap->root[nodeIndex].key;
        k_type min=myKey;
        for(i=0;i<BRANCHING_NUMBER;i++){
            if(childIndex+i<count){
                k_type childKey=heap->root[childIndex+i].key;
                if(childKey<min){
                    min=childKey;
                    minIndex=childIndex+i;
                }
            }
        }
        if(min<myKey){
            s_heap_swap_nodes(heap,nodeIndex,minIndex);
            nodeIndex=minIndex;
            childIndex=nodeIndex*BRANCHING_NUMBER+1;
        }else{
            break;
        }
        
    }
}

/*static void s_heap_delete_min(s_heap *heap){
    int count=heap->count;
    s_heap_swap_nodes(heap,count-1,0);
    heap->count--;
    s_heap_push_down(heap,0);
}


static void s_heap_decrease_key(s_heap *heap, int nodeIndex, k_type newKey){
   
    heap->root[nodeIndex].key=newKey;
    s_heap_bubble_up(heap,nodeIndex);
}*/


static void s_heap_insert_node(s_heap *heap, int nodeIndex, k_type newKey){
    s_heap_add_node_to_bottom(heap,nodeIndex,newKey);
    s_heap_bubble_up(heap,heap->count-1);
    
}

static s_heap s_heap_create_empty_heap(int pcount){
    s_heap heap;
    heap.count=0;
    heap.root=(s_heap_node*)malloc(pcount*sizeof(s_heap_node));
    return heap;
}


/*static int s_heap_empty(s_heap *heap){
    return heap->count==0;
}*/

static void s_heap_free_heap(s_heap *heap){
   
    free(heap->root);
}


#endif






