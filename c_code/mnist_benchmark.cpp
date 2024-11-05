#include "mbo_convolution.h"
#include <assert.h>
#include <time.h>
#include "memory_allocation.h"


//reads a file and returns the bytes

void *getFileData(const char *fileName){
    
    FILE *f=fopen(fileName,"r");
    size_t fsize;
    fseek(f,0L,SEEK_END);
    fsize=ftell(f);
    fseek(f,0L,SEEK_SET);
    void *data=malloc(fsize);
    size_t o = fread(data,1,fsize,f);
    if(o==0){
       printf("Data missing");
       assert(0);
    }
    fclose(f);
    return data;
}

int compare_int(const void *p, const void *q){
    int i1=*((int *) p);
    int i2=*((int *) q);
    
    if(i2>i1){
        return -1;
    }else if(i1>i2){
        return 1;
    }else{
        return 0;
    }
}

int compare_float(const void *p, const void *q){
    int i1=*((float *) p);
    int i2=*((float *) q);
    
    if(i2>i1){
        return -1;
    }else if(i1>i2){
        return 1;
    }else{
        return 0;
    }
}

int compare_indexed_floats(const void *p, const void *q){
    
    indexedFloat f1=*((indexedFloat *) p);
    indexedFloat f2=*((indexedFloat *) q);
    
    if(f2.dist>f1.dist){
        return -1;
    }else if(f1.dist>f2.dist){
        return 1;
    }else{
        return 0;
    }
    
}

void sort_indexed_floats(indexedFloat *distances, int num){
    
    qsort(distances, num, sizeof(indexedFloat), &compare_indexed_floats);
    
}
void sort_int(int *array, int n){
    qsort(array,n,sizeof(int),&compare_int);
}
void sort_float(float *array, int n){
    qsort(array,n,sizeof(float),&compare_float);
}
/*
indexedFloat *readAndConvertDistanceDataToIndexedFloat(const char *distanceFilename, const char *knnFilename, int maxNeighbors, int pcount){
    int i,j;
    float *distances=getFileData(distanceFilename);
    int *indicies=getFileData(knnFilename);
    indexedFloat *comboData=calloc(maxNeighbors*pcount,sizeof(indexedFloat));
    for(i=0;i<pcount;i++){
        
        for(j=0;j<maxNeighbors;j++){
            comboData[i*maxNeighbors+j].index=indicies[i*maxNeighbors+j];
            comboData[i*maxNeighbors+j].dist=distances[i*maxNeighbors+j];
            

        }
        sort_indexed_floats(&comboData[i*maxNeighbors],maxNeighbors);
        
    }
    free(distances);
    free(indicies);
    return comboData;
    
}*/


float kernel(float t){
    
    return exp(-t);
}



double factorial(int n){
    
    int i;
    double result=0;
    for(i=1;i<n;i++){
        result+=log(i+1);
    }
    
    return exp(result);
    
}


//Main MBO routine
void mbo_main(int *labels, int *I, int *J, float *W, int *ind, int *val, int *classCounts, bool prog, int pcount, int M, int m, int lcount, int maxIters, float stoppingCriterion, float temperature, float upperVolumeMultiplier, float lowerVolumeMultiplier){

   int i,j;
   //srand48(time(NULL));


   float volumeEpsilon=1e-7;
   char *mode="dvn";

   //Set vector indicating label locations and label values
   unsigned char *fixedLabels=(unsigned char*)calloc(pcount,1); //array that tracks which points have a fixed correct label
   for(j=0;j<m;j++){
      fixedLabels[ind[j]] = 1;
      labels[ind[j]]=val[j];
   }

   //float realRatio[lcount];
   //float sampleRatio[lcount];
   //memset(realRatio,0,lcount*sizeof(float));
   //memset(sampleRatio,0,lcount*sizeof(float));
   //float *realRatio = (float *) calloc(lcount, sizeof(float));
   //float *sampleRatioRatio = (float *) calloc(lcount, sizeof(float));

   //Compute number of neighbors of each vertex and vertex degrees
   int *nncounts = vector_int(pcount+1,0);    //number of neighbors
   j=0;
   for(i=0;i<pcount;i++){
      nncounts[i]=j;
      while((J[j]==i) & (j < M)){
         j++;
      }
   }
   nncounts[pcount] = j;


   if(prog){
      //Assign initial labels
      for(i=0;i<pcount;i++){
          if(fixedLabels[i] == 0){
             if(mode[1]=='p'){//initialization methods.
                 labels[i]=-1;//initialize by percolation
             }else{
                 labels[i]=rand()%lcount;//random initialization.
             }
          }
      }
   }

   generalConvolutionStructure g;
   g.neighbors = I;
   g.counts = nncounts;
   g.connectionStrengths = W;


   //MBO method constructions
   mbo_struct mbos; //object used by mbo.c to do convolutions
   memset(&mbos,0,sizeof(mbo_struct)); //clear memory
   mbos.nncounts=g.counts; //increasing integer array of length pcount.  nncounts[i+1]-nncounts[i] gives the number of nearest neighbors of node i.
   mbos.indicies=g.neighbors; // entries nncounts[i] through nncounts[i+1]-1 hold the indicies of the nearest neighbors of node i
   mbos.weights=g.connectionStrengths; // entries nncounts[i] through nncounts[i+1]-1 hold the weights of the nearest neighbors of node i

   mbos.updateList=(nodeChanged*)calloc(pcount,sizeof(nodeChanged)); //allocate memory for internal workspace
   mbos.surfaceTensions=NULL;//surfaceTensions; // create a lcount*lcount symmetric matrix if surface tensions are used.
   mbos.fixedLabels=fixedLabels; //binary array of length pcount recording the nodes whose label is known
   mbos.labels=labels; //integer array of length pcount holding label of each node
   mbos.pcount=pcount; //number of nodes
   mbos.lcount=lcount; //number of labels
   mbos.stoppingCriterion=stoppingCriterion; //algorithm stops if fewer than pcount*stoppingCriterion nodes change
   mbos.maxIters=maxIters; //max number of iterations
   mbos.singleGrowth=0;//use single growth or not
   //run mbo dynamics. see mbo_convolution.c
   mbos.epsilon=volumeEpsilon;
   mbos.temperature=temperature;
   mbos.upperVolumeMultiplier=upperVolumeMultiplier;
   mbos.lowerVolumeMultiplier=lowerVolumeMultiplier;

   mbos.classCounts=classCounts;
   mbos.k=(int) (M/pcount);  
   int lin=0;

   
   if(prog){
      bellman_ford_voronoi_initialization(&mbos, g, 1,lin);
   }
   normalize_matrix(g,pcount); //make the matrix have row sum 1

   
   if(temperature>0){
      run_mbo_with_temperature(mbos, mode[0]);
   }else{
      run_mbo(mbos, mode[0]);
   }

    
}




