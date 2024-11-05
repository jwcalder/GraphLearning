#define distance_type float
#define k_type float
#include "maj_simple_implicit_heap.h"
#include "mbo_convolution.h"
#include <float.h>
#include "maj_dijkstra.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <assert.h>



void run_auction_phase(s_heap *heapHolder, float *weights, float *prices, int *labels, int *unassigned, int *volumes, int pcount, int lcount, int *unassignedCount, float epsilon){
    int i,l;
    int uc=0;
    for(i=0;i<(*unassignedCount);i++){
        int index=unassigned[i];
        float bestValue=-FLT_MAX;
        float nextBest=-FLT_MAX;
        int bidItem=0;
        for(l=0;l<lcount;l++){
            if(weights[lcount*index+l]-prices[l]>bestValue){
                nextBest=bestValue;
                bestValue=weights[lcount*index+l]-prices[l];
                bidItem=l;
            }else if(weights[lcount*index+l]-prices[l]>nextBest){
                nextBest=weights[lcount*index+l]-prices[l];
                
            }
        }
        float bid=prices[bidItem]+epsilon+bestValue-nextBest;
        if(heapHolder[bidItem].count<volumes[bidItem]){
            s_heap_insert_node(&heapHolder[bidItem],index,bid);
            labels[index]=bidItem;
        }else{
            int minIndex=heapHolder[bidItem].root[0].originalIndex;
            labels[minIndex]=-1;
            heapHolder[bidItem].root[0].key=bid;
            heapHolder[bidItem].root[0].originalIndex=index;
            s_heap_push_down(&heapHolder[bidItem], 0);
            labels[index]=bidItem;
            unassigned[uc]=minIndex;
            prices[bidItem]=heapHolder[bidItem].root[0].key;
            uc++;
        }
    }
    *unassignedCount=uc;
}
void prepare_auction_phase(s_heap *heapHolder, int *unassigned, unsigned char *fixedLabels, int *unassignedCount, int pcount, int lcount){
    int i,l;
    int uc=0;
    for(l=0;l<lcount;l++){
        heapHolder[l].count=0;
    }
    for(i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            unassigned[uc]=i;
            uc++;
        }
    }
    *unassignedCount=uc;
}


void reset_auction_phase(s_heap *heapHolder, float *weights, float *prices, int *labels, int *unassigned, unsigned char *fixedPoints, int *volumes, int *unassignedCount, int pcount, int lcount, float epsilon){
    int i,l;
    int uc=0;
    for(l=0;l<lcount;l++){
        heapHolder[l].count=0;
    }
    for(i=0;i<pcount;i++){
        if(!fixedPoints[i]){
            float bestValue=-FLT_MAX;
            float nextBest=-FLT_MAX;
            for(l=0;l<lcount;l++){
                if(weights[lcount*i+l]-prices[l]>bestValue){
                    nextBest=bestValue;
                    bestValue=weights[lcount*i+l]-prices[l];
                }else if(weights[lcount*i+l]-prices[l]>nextBest){
                    nextBest=weights[lcount*i+l]-prices[l];
                }
            }
            if(bestValue-nextBest>epsilon){
                fixedPoints[i]=1;
                volumes[labels[i]]--;
                
            }else{
                unassigned[uc]=i;
                uc++;
            }
            
        }
    }
    *unassignedCount=uc;
}



//thresholding with upper bounds
void volume_preserving_auction(float *weights, int *labels, int *volumes, unsigned char *fixedLabels, int pcount, int lcount, float epsilonMin){
    int l;
    //s_heap heapHolder[lcount];
    s_heap *heapHolder = (s_heap*) malloc(lcount*sizeof(s_heap));
    for(l=0;l<lcount;l++){
        heapHolder[l]=s_heap_create_empty_heap(volumes[l]);
    }
    //float prices[lcount];
    //memset(prices,0,lcount*sizeof(float));
    float *prices = (float*) calloc(lcount, sizeof(float)); 
    int *unassigned = (int*)calloc(pcount,sizeof(int));
    unsigned char *fixedPoints=(unsigned char*)calloc(pcount,1);
    memcpy(fixedPoints,fixedLabels,pcount);
    float scalingFactor=4;
    float epsilon=epsilonMin*1.1;//*scalingFactor;
    int unassignedCount=0;
    
    prepare_auction_phase(heapHolder, unassigned, fixedLabels, &unassignedCount,  pcount,lcount);
    int auctionCount=0;
    while(epsilon>epsilonMin){
        while(unassignedCount>0){
           
            run_auction_phase(heapHolder, weights, prices, labels, unassigned, volumes, pcount, lcount, &unassignedCount, epsilon);
            auctionCount++;
        }
        reset_auction_phase(heapHolder, weights, prices, labels, unassigned,fixedPoints, volumes, &unassignedCount, pcount, lcount,epsilon);
        epsilon/=scalingFactor;
    }
    
    for(l=0;l<lcount;l++){
        s_heap_free_heap(&heapHolder[l]);
    }
    //printf("number of auction phases %d\n",auctionCount);
    free(unassigned);
    free(fixedPoints);
}


void run_reverse_auction_phase(s_heap heap, float *weights, unsigned char *fixedLabels, float *prices, int *labels, int *volumes, int *currentVolumes, int pcount, int lcount,  float epsilon){
    int i,l;
    for(l=0;l<lcount;l++){
        int numMissing=volumes[l]-currentVolumes[l];
        if(numMissing>0){
            heap.count=0;
            for(i=0;i<pcount;i++){
                if(labels[i]!=l&&!fixedLabels[i]){
                    float currentValue=weights[lcount*i+labels[i]]-prices[labels[i]];
                    float delta=weights[lcount*i+l]-prices[l]-currentValue;
                    
                    if(heap.count<numMissing){
                        s_heap_insert_node(&heap, i, delta);
                    }else if(heap.root[0].key<delta){
                        heap.root[0].key=delta;
                        heap.root[0].originalIndex=i;
                        s_heap_push_down(&heap,0);
                    }
                }
            }
            float priceSlash=heap.root[0].key;
            if(priceSlash<=0){
                prices[l]+=priceSlash-epsilon;
            }
            for(i=0;i<numMissing;i++){
                int index=heap.root[i].originalIndex;
                currentVolumes[labels[index]]--;
                currentVolumes[l]++;
                labels[index]=l;
            }
        }
        
    }
}

void prepare_reverse_auction_phase(float *weights, float *prices, int *labels, unsigned char *fixedLabels, int *currentVolumes, int *volumes, int pcount, int lcount, int *done){
    int i,j;
    memset(currentVolumes,0,lcount*sizeof(int));
    for(i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            float max=-FLT_MAX;
            int ml=0;
            for(j=0;j<lcount;j++){
                if(weights[lcount*i+j]-prices[j]>max){
                    max=weights[lcount*i+j]-prices[j];
                    ml=j;
                }
            }
            labels[i]=ml;
            currentVolumes[ml]++;
        }
    }
    int dd=1;
    for(j=0;j<lcount;j++){
        if(currentVolumes[j]<volumes[j]){
            dd=0;
        }
    }
    *done=dd;
}
//thresholding with lower bounds
void volume_preserving_reverse_auction(float *weights, int *labels, int *volumes, unsigned char *fixedLabels, int pcount, int lcount, float epsilonMin){
    int j;
    float epsilon=fmax(1,epsilonMin*1.1);
    float scalingFactor=4;
    //float prices[lcount];
    float *prices = (float*) calloc(lcount, sizeof(float)); 
    s_heap heap=s_heap_create_empty_heap(pcount);
    //memset(prices,0,lcount*sizeof(float));
    //int currentVolumes[lcount];
    int *currentVolumes = (int*) calloc(lcount, sizeof(int)); 
    while(epsilon>epsilonMin){
        int done=0;
        prepare_reverse_auction_phase(weights,prices,labels,fixedLabels,currentVolumes,volumes, pcount,lcount,&done);
        while(!done){
            run_reverse_auction_phase(heap, weights, fixedLabels, prices, labels, volumes, currentVolumes, pcount, lcount, epsilon);
            done=1;
            for(j=0;j<lcount;j++){
                if(currentVolumes[j]<volumes[j]){
                    done=0;
                }
            }
        }
        epsilon/=scalingFactor;
    }
    s_heap_free_heap(&heap);
    
}

void forward_to_reverse(int *labels, unsigned char *fixedLabels, int *currentVolumes, int *lowerVolumes, int pcount, int lcount, int *done){
    int i,j;
    memset(currentVolumes,0,lcount*sizeof(int));
    for(i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            currentVolumes[labels[i]]++;
        }
    }
    int dd=1;
    for(j=0;j<lcount;j++){
        if(currentVolumes[j]<lowerVolumes[j]){
            dd=0;
        }
    }
    *done=dd;
}

//thresholding with upper and lower volume bounds
void volume_preserving_forward_reverse_auction(float *weights, int *labels, int *lowerVolumes, int *upperVolumes, unsigned char *fixedLabels, int pcount, int lcount, float epsilonMin){
    int j,l;
    float scalingFactor=4;
    float epsilon=fmax(1,epsilonMin*1.01*scalingFactor);
    //float prices[lcount];
    float *prices = (float*) calloc(lcount, sizeof(float)); 
    s_heap reverseHeap=s_heap_create_empty_heap(pcount);
    //memset(prices,0,lcount*sizeof(float));
    //int currentVolumes[lcount];
    int *currentVolumes = (int*) calloc(lcount, sizeof(int)); 
    //s_heap forwardHeapHolder[lcount];
    s_heap *forwardHeapHolder = (s_heap*) malloc(lcount*sizeof(s_heap));
    for(l=0;l<lcount;l++){
        forwardHeapHolder[l]=s_heap_create_empty_heap(upperVolumes[l]);
    }
    int *unassigned = (int*)calloc(pcount,sizeof(int));
    while(epsilon>epsilonMin){
        int feasible=0;
        while(!feasible){
            feasible=1;
            int unassignedCount=0;
            prepare_auction_phase(forwardHeapHolder, unassigned, fixedLabels, &unassignedCount,  pcount,lcount);
            int numBids=0;
            while(unassignedCount>0){
                //printf("%d\n",unassignedCount);
                run_auction_phase(forwardHeapHolder, weights, prices, labels, unassigned, upperVolumes, pcount, lcount, &unassignedCount, epsilon);
                numBids+=unassignedCount;
            }
            epsilon=fmax(epsilon/scalingFactor,epsilonMin);
            int done=0;
            for(l=0;l<lcount;l++){
                //printf("%d %d %d\n", lowerVolumes[l], currentVolumes[l], upperVolumes[l]);
                if(currentVolumes[l]>upperVolumes[l]||currentVolumes[l]<lowerVolumes[l]){
                    feasible=0;
                }
            }
            if(feasible){
                continue;
            }
            //forward_to_reverse(labels, fixedLabels, currentVolumes, lowerVolumes, pcount, lcount, &done);
            prepare_reverse_auction_phase(weights, prices, labels, fixedLabels, currentVolumes, lowerVolumes, pcount,lcount, &done);
            while(!done){
                run_reverse_auction_phase(reverseHeap, weights, fixedLabels, prices, labels, lowerVolumes, currentVolumes, pcount, lcount, epsilon);
                done=1;
                for(j=0;j<lcount;j++){
                    if(currentVolumes[j]<lowerVolumes[j]){
                        done=0;
                    }
                }
            }
            for(l=0;l<lcount;l++){
                //printf("%d %d %d\n", lowerVolumes[l], currentVolumes[l], upperVolumes[l]);
                if(currentVolumes[l]>upperVolumes[l]||currentVolumes[l]<lowerVolumes[l]){
                    feasible=0;
                }
            }
            //printf("%d %d\n", forwardOptimal,reverseOptimal);
            
        }
        
        epsilon/=scalingFactor;
    }
    
    free(unassigned);
    for(l=0;l<lcount;l++){
        s_heap_free_heap(&forwardHeapHolder[l]);
    }
    s_heap_free_heap(&reverseHeap);
}



generalConvolutionStructure create_symmetric_adjacency_matrix(indexedFloat *neighborData, int pcount, int maxNeighbors, int k){
    
    int i,j;
    
    float *medians=(float*)calloc(pcount,sizeof(float));
    for(i=0;i<pcount;i++){
        medians[i]=neighborData[i*maxNeighbors+k/2].dist;
    }
    
    
    int *tcounts=(int*)calloc(pcount+1,sizeof(int));
    for(i=0;i<pcount;i++){
        for(j=0;j<k;j++){
            int index=neighborData[maxNeighbors*i+j].index;
            tcounts[index]++;
            tcounts[i]++;
            
        }
    }
    for(i=0;i<pcount;i++){
        tcounts[i+1]+=tcounts[i];
    }
    int num=tcounts[pcount-1];
    float *tempWeights=(float*)calloc(num,sizeof(float));
    int *tempIndicies=(int*)calloc(num,sizeof(int));
    for(i=0;i<pcount;i++){
        for(j=0;j<k;j++){
            int index=neighborData[maxNeighbors*i+j].index;
            float m1=medians[i];
            float m2=medians[index];
            float factor=sqrt(m1*m2);
            tempIndicies[tcounts[i]-1]=index;
            tempWeights[tcounts[i]-1]=neighborData[i*maxNeighbors+j].dist/factor;
            tcounts[i]--;
            tempIndicies[tcounts[index]-1]=i;
            tempWeights[tcounts[index]-1]=neighborData[i*maxNeighbors+j].dist/factor;
            tcounts[index]--;
            
        }
    }
    
    int *counts=(int*)calloc(pcount+1,sizeof(int));
    int *seen=(int*)calloc(pcount,sizeof(int));
    
    for(i=0;i<pcount;i++){
        for(j=tcounts[i];j<tcounts[i+1];j++){
            int index=tempIndicies[j];
            if(seen[index]!=i+1){
                seen[index]=i+1;
                counts[i]++;
            }
        }
    }
    for(i=0;i<pcount;i++){
        counts[i+1]+=counts[i];
    }
    
    float *weights=(float*)calloc(counts[pcount],sizeof(float));
    int *indicies=(int*)calloc(counts[pcount],sizeof(int));
    memset(seen,0,pcount*sizeof(int));
    
    for(i=0;i<pcount;i++){
        for(j=tcounts[i];j<tcounts[i+1];j++){
            int index=tempIndicies[j];
            if(seen[index]!=i+1){
                seen[index]=i+1;
                indicies[counts[i]-1]=index;
                weights[counts[i]-1]=tempWeights[j];
                counts[i]--;
                
            }
        }
    }
    
    
    
    generalConvolutionStructure g;
    g.neighbors=indicies;
    g.connectionStrengths=weights;
    g.counts=counts;
    
    free(tcounts);
    free(medians);
    free(tempIndicies);
    free(tempWeights);
    free(seen);
    return g;
    
}

generalConvolutionStructure create_symmetric_matrix(float (* kernel)(float), indexedFloat *neighborData, int pcount, int maxNeighbors, int k){
    int i,j;
    generalConvolutionStructure g=create_symmetric_adjacency_matrix(neighborData, pcount,  maxNeighbors, k);
    for(i=0;i<pcount;i++){
        for(j=g.counts[i];j<g.counts[i+1];j++){
            g.connectionStrengths[j]=kernel(g.connectionStrengths[j]);
        }
    }
    return g;
    
}

void normalize_matrix(generalConvolutionStructure g, int pcount){
    int i,j;
    float *sums=(float*)calloc(pcount,sizeof(float));
    float *weights=g.connectionStrengths;
    int *indicies=g.neighbors;
    int *counts=g.counts;
    for(i=0;i<pcount;i++){
        for(j=counts[i];j<counts[i+1];j++){
            sums[i]+=weights[j];
        }
        
    }
    for(i=0;i<pcount;i++){
        for(j=counts[i];j<counts[i+1];j++){
            int index=indicies[j];
            weights[j]/=sqrt(sums[i]*sums[index]);
        }
    }
    free(sums);
    
}

generalConvolutionStructure create_dual_convolution_structure(mbo_struct mbos){
    
    int i,j;
    int tot=0;
    int pcount=mbos.pcount;
    
    int *counts=(int*)calloc(pcount+1,sizeof(int)); //store the index range for each point
    int *nncounts=mbos.nncounts;
    for(i=0;i<pcount;i++){
        for(j=nncounts[i];j<nncounts[i+1];j++){
            int neighbor=mbos.indicies[j];
            
            counts[neighbor]++; //count the number of dual neighbors for each point
            
            tot++;
        }
    }
    counts[pcount]=tot;
    for(i=1;i<pcount;i++){
        counts[i]+=counts[i-1]; //count now holds upper index cutoff for each point
    }
    int *neighbors=(int*)calloc(tot,sizeof(int));
    float *connectionStrengths=(float*)calloc(tot,sizeof(float));
    
    for(i=0;i<pcount;i++){//loop to assign neighbors and strengths and set counts to hold the index range for each point
        for(j=nncounts[i];j<nncounts[i+1];j++){
            int neighbor=mbos.indicies[j];
            float str=mbos.weights[j];
            neighbors[counts[neighbor]-1]=i;
            connectionStrengths[counts[neighbor]-1]=str;
            counts[neighbor]--;
            
        }
    }
    generalConvolutionStructure dual;
    dual.neighbors=neighbors;
    dual.connectionStrengths=connectionStrengths;
    dual.counts=counts;
    
    return dual;
    
}


//destroy convolution structure
void free_generalConvolutionStructure(generalConvolutionStructure dual){
    free(dual.neighbors);
    free(dual.connectionStrengths);
    free(dual.counts);
    
}


void calc_linear_from_voronoi_neighbors(mbo_struct *mbos, generalConvolutionStructure g, float *voronoiDistances, int *numbers){
    
    int i,j, l;
    int pcount=mbos->pcount;
    int lcount=mbos->lcount;
    int *labels=mbos->labels;
    float *linear=(float*)calloc(pcount*lcount,sizeof(float));
    int *seen=(int*)calloc(lcount,sizeof(int));
    for(i=0;i<pcount;i++){
        for(j=g.counts[i];j<g.counts[i+1];j++){
            int index=g.neighbors[j];
            float dist=g.connectionStrengths[j];
            if(seen[labels[index]]!=i+1){
                linear[lcount*i+labels[index]]=dist+voronoiDistances[index];
                seen[labels[index]]=i+1;
            }else{
                linear[lcount*i+labels[index]]=fmin(dist+voronoiDistances[index],linear[lcount*i+labels[index]]);
            }
            
        }
        for(l=0;l<lcount;l++){
            if(linear[lcount*i+l]>0){
                linear[lcount*i+l]=.1*exp(-linear[lcount*i+l]);
            }
        }
    }
    mbos->linear=linear;
    free(seen);
    
}

void bellman_ford_voronoi_initialization(mbo_struct *mbos, generalConvolutionStructure g, float distanceExponent, int lin){
    int i,j;
    //clock_t b,e;
    int pcount=mbos->pcount;
    unsigned char *active=(unsigned char*)calloc(pcount,1);
    unsigned char *fixedLabels=mbos->fixedLabels;
    int *labels=mbos->labels;
    memcpy(active,fixedLabels,pcount*1);
    float *voronoiDistances=(float*)calloc(pcount,sizeof(float));
    //b=clock();
    for(i=0;i<pcount;i++){
        for(j=g.counts[i];j<g.counts[i+1];j++){
            g.connectionStrengths[j]=pow(g.connectionStrengths[j],distanceExponent);
        }
        if(!active[i]){
            voronoiDistances[i]=FLT_MAX;
        }
        
        
    }
    int done=0;
    while(!done){
        done=1;
        for(i=0;i<pcount;i++){
            if(active[i]){
                done=0;
                for(j=g.counts[i];j<g.counts[i+1];j++){
                    int index=g.neighbors[j];
                    float dist=g.connectionStrengths[j];
                    float current=voronoiDistances[i];
                    if(current+dist<voronoiDistances[index]){
                        voronoiDistances[index]=current+dist;
                        active[index]=1;
                        labels[index]=labels[i];
                    }
                }
                active[i]=0;
                
            }
        }
        
    }
    //e=clock();
    //printf("voronoi %f\n", (e-b)/(CLOCKS_PER_SEC*1.0));
    //b=clock();
    
    if(lin){
        calc_linear_from_voronoi_neighbors(mbos, g, voronoiDistances, labels);
    }
    //e=clock();
    //printf("linear %f\n", (e-b)/(CLOCKS_PER_SEC*1.0));
    
    free(active);
    free(voronoiDistances);
}





void voronoi_initialization(mbo_struct *mbos, generalConvolutionStructure g, int maxNeighbors, float distanceExponent, int lin){
    int i,j;
    int *labels=mbos->labels;
    unsigned char *fixedLabels=mbos->fixedLabels;
    int pcount=mbos->pcount;
    float *voronoiDistances=(float*)calloc(pcount,sizeof(float));
    
    for(i=0;i<pcount;i++){
        for(j=g.counts[i];j<g.counts[i+1];j++){
            g.connectionStrengths[j]=pow(g.connectionStrengths[j],distanceExponent);
        }
    }
    
    for(i=0;i<pcount;i++){
        if(fixedLabels[i]){
            voronoiDistances[i]=0;
        }else{
            voronoiDistances[i]=FLT_MAX;
        }
    }
    
    labeled_dijkstra(g.neighbors, g.connectionStrengths, g.counts, voronoiDistances,labels, pcount, 1);
    
    
    if(lin){
        calc_linear_from_voronoi_neighbors(mbos, g, voronoiDistances, labels);
    }
    
    free(voronoiDistances);
}




//calculate the convolution values for each label at every point

char calc_first_convolution(mbo_struct mbos, float *labelValues){
    
    int i,j;
    
    int *nncounts=mbos.nncounts;
    char noneNegative=1;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    for(i=0;i<pcount;i++){
        char activate=0;
        
        for(j=nncounts[i];j<nncounts[i+1];j++){
            int neighbor=mbos.indicies[j];
            int id=mbos.labels[neighbor];
            activate|=(id!=-1);
            
            if(id==-1){//only happens when using initialization by percolation.
                noneNegative=0;
                continue; //initialization by percolation is still happening, ignore nodes that haven't been labeled yet
            }
            float str=mbos.weights[j];
            
            labelValues[lcount*i+id]+=str;
        }
        if(activate){
            mbos.fixedLabels[i]&=1;
        }
        
        
        
    }
    return noneNegative;
    
}

char calc_dual_convolution(mbo_struct mbos, generalConvolutionStructure dual, float *convLabels){
    
    int i,j,l;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    
    
    float *labelValues=(float*)calloc(pcount*lcount,sizeof(float));
    char noneNegative=calc_first_convolution(mbos,labelValues);
    for(i=0;i<pcount;i++){
        for(j=dual.counts[i];j<dual.counts[i+1];j++){//counts holds the neighbor and strength array index range for each point
            int neighbor=dual.neighbors[j];
            
            float str=dual.connectionStrengths[j];
            for(l=0;l<lcount;l++){
                convLabels[i*lcount+l]+=str*labelValues[lcount*neighbor+l];
                
            }
        }
        
    }
    
    free(labelValues);
    
    
    return noneNegative;
    
}

void symmetric_matrix_vector_product(float *matrix, float *invector, float *outvector, int length){
    
    int i,j;
    for(i=0;i<length;i++){
        outvector[i]=0;
        for(j=0;j<length;j++){
            outvector[i]+=invector[j]*matrix[i*length+j];
        }
    }
    
}

void recompute_convolution_with_update_list_k(mbo_struct mbos, float *convLabels, int changedCount){
    
    int i,j;
    int *nncounts=mbos.nncounts;
    int lcount=mbos.lcount;
    for(i=0;i<changedCount;i++){
        int index=mbos.updateList[i].index;
        int to=mbos.updateList[i].to;
        int from=mbos.updateList[i].from;
        for(j=nncounts[index];j<nncounts[index+1];j++){
            int newIndex=mbos.indicies[j];
            float str=mbos.weights[j];
            convLabels[newIndex*lcount+from]-=str;
            convLabels[newIndex*lcount+to]+=str;
            if(convLabels[newIndex*lcount+from]<0){
                convLabels[newIndex*lcount+from]=0;
            }
            
        }
    }
    
}


void recompute_convolution_with_update_list_d(mbo_struct mbos, float *convLabels, int changedCount){
    
    int i,j,r;
    int *nncounts=mbos.nncounts;
    int lcount=mbos.lcount;
    for(i=0;i<changedCount;i++){
        int index=mbos.updateList[i].index;
        int to=mbos.updateList[i].to;
        int from=mbos.updateList[i].from;
        for(j=nncounts[index];j<nncounts[index+1];j++){
            int newIndex=mbos.indicies[j];
            float str1=mbos.weights[j];
            for(r=nncounts[newIndex];r<nncounts[newIndex+1];r++){
                int newNewIndex=mbos.indicies[r];
                float str2=mbos.weights[r];
                float combo=str1*str2;
                convLabels[newNewIndex*lcount+from]-=combo;
                convLabels[newNewIndex*lcount+to]+=combo;
                if(convLabels[newNewIndex*lcount+from]<0){
                    convLabels[newNewIndex*lcount+from]=0;
                }
            }
            
        }
    }
    
}


float update_energy(mbo_struct mbos, float *convLabels){
    int i,j;
    float energy=0;
    int *labels=mbos.labels;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    for(i=0;i<pcount;i++){
        for(j=0;j<lcount;j++){
            if(j!=labels[i]){
                energy+=convLabels[i*lcount+j];
            }
        }
    }
    return energy;
}


void monte_carlo_thresholding(mbo_struct mbos, float *convLabels){
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int *labels=mbos.labels;
    float tinv=1/mbos.temperature;
    unsigned char *fixedLabels=mbos.fixedLabels;
    int i,l;
    for(i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            float max=convLabels[lcount*i+labels[i]];
            for(l=0;l<lcount;l++){
                if(convLabels[lcount*i+l]>max){
                    max=convLabels[lcount*i+l];
                }
            }
            //float helper[lcount];
	    float *helper = (float *) malloc(lcount*sizeof(float));
            helper[0]=exp(tinv*(convLabels[lcount*i]-max));
            for(l=1;l<lcount;l++){
                helper[l]=helper[l-1]+exp(tinv*(convLabels[lcount*i+l]-max));
            }
            float prob=helper[lcount-1]*rand()/(RAND_MAX*1.0);
            for(int l=0;l<lcount;l++){
                if(prob<helper[l]){
                    labels[i]=l;
                    break;
                }
            }
        }
    }
    
    
}


void normal_thresholding(mbo_struct mbos, float *convLabels){
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
   // printf("nothing to see here folks\n");
    int *labels=mbos.labels;
    unsigned char *fixedLabels=mbos.fixedLabels;
    int i,l;
    for(i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            float max=convLabels[lcount*i+labels[i]];
            int maxIndex=labels[i];
            for(l=0;l<lcount;l++){
                if(convLabels[lcount*i+l]>max){
                    max=convLabels[lcount*i+l];
                    maxIndex=l;
                }
            }
            labels[i]=maxIndex;
        }
    }
    
}

//updates the labels with volume constraints.  upper and lower bounds can be set here
float update_labels(mbo_struct mbos, float *convLabels, int *volumes, int *cc){
    int i,l;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int *labels=mbos.labels;
    unsigned char *fixedLabels=mbos.fixedLabels;
    float *surfaceTensions=mbos.surfaceTensions;
    float epsilon=mbos.epsilon;
    float energy=0;
    int *oldLabels=(int*)calloc(pcount,sizeof(int));
    memcpy(oldLabels,labels,pcount*sizeof(int));
    float uvm=mbos.upperVolumeMultiplier;
    float lvm=mbos.lowerVolumeMultiplier;
    if(surfaceTensions){
        //float temp[lcount];
	float *temp = (float*)malloc(lcount*sizeof(float));
        for(i=0;i<pcount;i++){
            memcpy(temp,&convLabels[lcount*i],lcount*sizeof(float));
            symmetric_matrix_vector_product(surfaceTensions, temp, &convLabels[lcount*i], lcount);
            energy+=convLabels[lcount*i+labels[i]];
            float max=0;
            for(l=0;l<lcount;l++){
                if(convLabels[lcount*i+l]>max){
                    max=convLabels[lcount*i+l];
                }
            }
            for(l=0;l<lcount;l++){
                convLabels[lcount*i+l]=max-convLabels[lcount*i+l];
            }
            
        }
        //int lowerVolumes[lcount];
        //int upperVolumes[lcount];
        //set upper and lower volume bounds
        //for(l=0;l<lcount;l++){
        //    lowerVolumes[l]=volumes[l]*lvm;
        //    upperVolumes[l]=volumes[l]*uvm;
        //}
        normal_thresholding(mbos, convLabels);
        // use for upper bounds only
        //volume_preserving_auction(convLabels, labels, upperVolumes, fixedLabels, pcount, lcount,epsilon);
        
        //use for lower bounds only
        //volume_preserving_reverse_auction(convLabels, labels, lowerVolumes, fixedLabels, pcount, lcount,epsilon);
        //use for upper and lower bounds together
        //volume_preserving_forward_reverse_auction(convLabels, labels,  lowerVolumes, upperVolumes, fixedLabels, pcount, lcount,epsilon);
    }else{
        for(i=0;i<pcount;i++){
            float tot=0;
            for(l=0;l<lcount;l++){
                tot+=convLabels[lcount*i+l];
            }
            energy+=tot-convLabels[lcount*i+labels[i]];
        }
        //int lowerVolumes[lcount];
        //int upperVolumes[lcount];
	int *lowerVolumes = (int*)malloc(lcount*sizeof(int));
	int *upperVolumes = (int*)malloc(lcount*sizeof(int));
        //set upper and lower volume bounds
        for(l=0;l<lcount;l++){
            lowerVolumes[l]=volumes[l]*lvm;
            upperVolumes[l]=volumes[l]*uvm;
        }
        //normal_thresholding(mbos, convLabels);
        //monte_carlo_thresholding(mbos,convLabels);
        // use for upper bounds only
       //volume_preserving_auction(convLabels, labels, upperVolumes, fixedLabels, pcount, lcount,epsilon);
        
        //use for lower bounds only
        //volume_preserving_reverse_auction(convLabels, labels, lowerVolumes, fixedLabels, pcount, lcount,epsilon);
        //use for upper and lower bounds together
       volume_preserving_forward_reverse_auction(convLabels, labels,  lowerVolumes, upperVolumes, fixedLabels, pcount, lcount,epsilon);
    }
    int changedCount=0;
    for(i=0;i<pcount;i++){
        if(oldLabels[i]!=labels[i]){
            mbos.updateList[changedCount].from=oldLabels[i];
            mbos.updateList[changedCount].to=labels[i];
            mbos.updateList[changedCount].index=i;
            changedCount++;
        }
    }
    free(oldLabels);
    *cc=changedCount;
    return energy;
}



void remove_fixed_labels_from_volumes(mbo_struct mbos){
    int pcount=mbos.pcount;
    int *volumes=mbos.classCounts;
    unsigned char *fixedLabels=mbos.fixedLabels;
    int *labels=mbos.labels;
    int i;
    for(i=0;i<pcount;i++){
        if(fixedLabels[i]){
            volumes[labels[i]]--;
        }
    }
}


//creates two normal random variables
void box_mueller_transform(float *z1, float *z2, float mu, float sigma){
    float r1=rand()/(RAND_MAX*1.0);
    float r2=rand()/(RAND_MAX*1.0);
    float lsq=sqrt(-2*log(r1));
    *z1=mu+sigma*lsq*cos(2*3.14159*r2);
    *z2=mu+sigma*lsq*sin(2*3.14159*r2);
}

//add gaussian noise with standard deviation=temperature to all of the convolution values
void add_temperature(mbo_struct mbos, float *convLabels){
    float temperature=mbos.temperature;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int i;
    for(i=0;i<pcount*lcount/2;i++){
        float noise1,noise2;
        box_mueller_transform(&noise1,&noise2,0,temperature);
        convLabels[2*i]+=noise1;
        convLabels[2*i+1]+=noise2;
    }
}


int convolve_and_threshold(mbo_struct mbos, int *tempLabels){
    int *nncounts=mbos.nncounts;
    int *indicies=mbos.indicies;
    //float *weights=mbos.weights;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int *labels=mbos.labels;
    //int coefficientHolder[lcount];
    int *coefficientHolder = (int*)malloc(lcount*sizeof(int));
    int changedCount=0;
    unsigned char *fixedLabels=mbos.fixedLabels;
    for(int i=0;i<pcount;i++){
        if(!fixedLabels[i]){
            coefficientHolder[labels[i]]=0;
            
            for(int j=nncounts[i];j<nncounts[i+1];j++){
                int index=indicies[j];
                coefficientHolder[labels[index]]=0;
            }
            for(int j=nncounts[i];j<nncounts[i+1];j++){
                int index=indicies[j];
                coefficientHolder[labels[index]]++;
            }
            
            int max=0;
            int maxIndex=labels[i];
            
            for(int j=nncounts[i];j<nncounts[i+1];j++){
                int index=indicies[j];
                if(coefficientHolder[labels[index]]>max){
                    maxIndex=labels[index];
                    max=coefficientHolder[labels[index]];
                }
            }
            tempLabels[i]=maxIndex;
            if(tempLabels[i]!=labels[i]){
                changedCount++;
            }
        }else{
            tempLabels[i]=labels[i];
        }
    }
    
    memcpy(labels,tempLabels,pcount*sizeof(int));
    //printf("%d\n",changedCount);
    
    return changedCount;
    
}




void run_mbo_lfr(mbo_struct mbos){
    int maxIters=mbos.maxIters;
    int pcount=mbos.pcount;
    int *tempLabels=(int*)calloc(pcount,sizeof(int));
    for(int i=0;i<maxIters;i++){
        int changedCount=convolve_and_threshold(mbos,tempLabels);
        if(changedCount<mbos.stoppingCriterion){
            // printf("%d %d\n",i, changedCount);
            break;
        }
       
        
    }
    free(tempLabels);
}





float run_mbo(mbo_struct mbos, char mode){
    
    int i=0;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int maxIters=mbos.maxIters;
    float oldEnergy=FLT_MAX;
    float stoppingCriterion=mbos.stoppingCriterion;
    int degree=mbos.k;
    int d2=degree*degree;
    generalConvolutionStructure dual;
    if(mode!='k'){
        dual=create_dual_convolution_structure(mbos);
    }
    //volume preserving algorithms don't see the fixed labels
    remove_fixed_labels_from_volumes(mbos);
    //char noneNegative=0;
    
    //int volumes[lcount];
    int *volumes = (int*)malloc(lcount*sizeof(int));
    memcpy(volumes,mbos.classCounts,lcount*sizeof(int));
    int changedCount=pcount*lcount;
    float *convLabels=(float*)calloc(pcount*lcount,sizeof(float));
    for(i=0;i<maxIters;i++){
        //factor=sqrt(factor);
        if(mode=='k'&& changedCount*degree>pcount*lcount){
            memset(convLabels,0,pcount*lcount*sizeof(float));
            calc_first_convolution(mbos, convLabels);
        }else if(changedCount*d2>pcount*lcount){
            memset(convLabels,0,pcount*lcount*sizeof(float));
            calc_dual_convolution(mbos, dual,convLabels);
        }else{
            if(mode!='k'){
                recompute_convolution_with_update_list_d(mbos, convLabels, changedCount);
            }else{
                recompute_convolution_with_update_list_k(mbos, convLabels, changedCount);
            }
            
        }
        
        //volume constraints are imposed here
        float newEnergy=update_labels(mbos, convLabels,volumes,&changedCount);
        
        if((oldEnergy-newEnergy)/newEnergy<stoppingCriterion){
            break;
            
        }
        oldEnergy=newEnergy;
        
        
    }
    free(convLabels);
    
    if(mode!='k'){
        free_generalConvolutionStructure(dual);
    }
    
    return oldEnergy;
}

//run mbo with temperature
float run_mbo_with_temperature(mbo_struct mbos, char mode){
    
    int i=0;
    int pcount=mbos.pcount;
    int lcount=mbos.lcount;
    int maxIters=mbos.maxIters;
    int degree=mbos.k;
    int d2=degree*degree;
    generalConvolutionStructure dual;
    if(mode!='k'){
        dual=create_dual_convolution_structure(mbos);
    }
    remove_fixed_labels_from_volumes(mbos);
    float best=FLT_MAX;
    int *savedLabels=(int*)calloc(pcount,sizeof(int));
    //int volumes[lcount];
    int *volumes = (int*)malloc(lcount*sizeof(int));
    memcpy(volumes,mbos.classCounts,lcount*sizeof(int));
    float *convLabels=(float*)calloc(pcount*lcount,sizeof(float));
    int changedCount=pcount*lcount;
    for(i=0;i<maxIters;i++){
        //factor=sqrt(factor);
        float *tempLabels=(float*)calloc(pcount*lcount,sizeof(float));
        if(mode=='k'&& changedCount*degree>pcount*lcount){
            memset(convLabels,0,pcount*lcount*sizeof(float));
            calc_first_convolution(mbos, convLabels);
        }else if(changedCount*d2>pcount*lcount){
            memset(convLabels,0,pcount*lcount*sizeof(float));
            calc_dual_convolution(mbos, dual,convLabels);
        }else{
            if(mode!='k'){
                recompute_convolution_with_update_list_d(mbos, convLabels, changedCount);
            }else{
                recompute_convolution_with_update_list_k(mbos, convLabels, changedCount);
            }

        }
        //calculates the energy of the current configuration mbos.labels
        float energy=update_energy(mbos,convLabels);
        
        //check if this configuration is the best so far
        if(energy<best){
            //save the best
            memcpy(savedLabels,mbos.labels,pcount*sizeof(int));
            best=energy;
        }
        
        if(i==maxIters-1){
            //we cannot check the energy of the final configuration so there's no reason to calculate the update
            break;
        }
        memcpy(tempLabels,convLabels,pcount*lcount*sizeof(float));
        //once the energy has been correctly calculated we can add in temperature
        add_temperature( mbos, tempLabels);
        //threshold with volume constraints
        update_labels(mbos, tempLabels,volumes,&changedCount);
        
        //mbos.temperature*=.98;
        
        free(tempLabels);
    }
    free(convLabels);
    
    if(mode!='k'){
        free_generalConvolutionStructure(dual);
    }
    //copy the best configuration
    memcpy(mbos.labels, savedLabels, pcount*sizeof(int));
    free(savedLabels);
    
    return best;
}






