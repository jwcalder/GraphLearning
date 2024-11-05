#include <stdlib.h>
#ifndef BRANCHING_NUMBER
#define BRANCHING_NUMBER 8
#endif
#ifndef MAJ_IMPLICIT_HEAP
#define MAJ_IMPLICIT_HEAP

typedef struct{
    key_type key;
    int originalIndex;
}implicit_heap_node;

typedef struct{
    implicit_heap_node *root;
    int *locations;
    int count;
}implicit_heap;



static void swap_nodes(implicit_heap *heap,int nodeIndex1, int nodeIndex2){
    int i1=heap->root[nodeIndex1].originalIndex;
    int i2=heap->root[nodeIndex2].originalIndex;
    key_type key1=heap->root[nodeIndex1].key;
    key_type key2=heap->root[nodeIndex2].key;
    heap->locations[i1]=nodeIndex2;
    heap->locations[i2]=nodeIndex1;
    heap->root[nodeIndex1].originalIndex=i2;
    heap->root[nodeIndex2].originalIndex=i1;
    heap->root[nodeIndex1].key=key2;
    heap->root[nodeIndex2].key=key1;
}

static void add_node_to_bottom(implicit_heap *heap, int originalIndex, key_type key){
    
    int count=heap->count;
    heap->root[count].originalIndex=originalIndex;
    heap->root[count].key=key;
    heap->locations[originalIndex]=count;
    heap->count++;
}

static void bubble_up(implicit_heap *heap, int nodeIndex){
    
    while (nodeIndex>0) {
        key_type myKey=heap->root[nodeIndex].key;
        int parentIndex=nodeIndex/BRANCHING_NUMBER-((nodeIndex%BRANCHING_NUMBER)==0);
        key_type parentKey=heap->root[parentIndex].key;
        if(myKey<parentKey){
            swap_nodes(heap,nodeIndex,parentIndex);
            nodeIndex=parentIndex;
        }else{
            break;
        }
       
    }
}

static void push_down(implicit_heap *heap, int nodeIndex){
    
    int i;
    int count=heap->count;
    int childIndex=nodeIndex*BRANCHING_NUMBER+1;
    while(childIndex<count){
        int minIndex=childIndex;
        key_type myKey=heap->root[nodeIndex].key;
        key_type min=myKey;
        for(i=0;i<BRANCHING_NUMBER;i++){
            if(childIndex+i<count){
                key_type childKey=heap->root[childIndex+i].key;
                if(childKey<min){
                    min=childKey;
                    minIndex=childIndex+i;
                }
            }
        }
        if(min<myKey){
            swap_nodes(heap,nodeIndex,minIndex);
            nodeIndex=minIndex;
            childIndex=nodeIndex*BRANCHING_NUMBER+1;
        }else{
            break;
        }
        
    }
}

static void delete_min(implicit_heap *heap){
    int count=heap->count;
    swap_nodes(heap,count-1,0);
    heap->count--;
    push_down(heap,0);
}


static void decrease_key(implicit_heap *heap, int nodeIndex, key_type newKey){
   
    heap->root[nodeIndex].key=newKey;
    bubble_up(heap,nodeIndex);
}


static void insert_node(implicit_heap *heap, int nodeIndex, key_type newKey){
    add_node_to_bottom(heap,nodeIndex,newKey);
    bubble_up(heap,heap->locations[nodeIndex]);
    
}

static implicit_heap create_empty_heap_with_locations(int pcount){
    implicit_heap heap;
    heap.count=0;
    heap.root=(implicit_heap_node*)malloc(pcount*sizeof(implicit_heap_node));
    heap.locations=(int*)malloc(pcount*sizeof(int));
    return heap;
}

/*static implicit_heap create_heap_with_batch(key_type *keys, int pcount, int dstride){
    int i;
    implicit_heap heap;
    heap.root=malloc(pcount*sizeof(implicit_heap_node));
    heap.locations=malloc(pcount*sizeof(int));
    heap.count=0;
    
    
   
    for(i=0;i<pcount;i++){
        add_node_to_bottom(&heap,i,keys[i*dstride]);
    }
    
    for(i=1;i<=pcount;i++){
        push_down(&heap,pcount-i);
        
    }
   
    return heap;
}*/


static void clear_heap(implicit_heap *heap){
    heap->count=0;
}


static int empty(implicit_heap *heap){
    return heap->count==0;
}

static void free_heap(implicit_heap *heap){
    free(heap->locations);
    free(heap->root);
}

#endif







