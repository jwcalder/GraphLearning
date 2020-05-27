#include "mbo_convolution.h"
#include <assert.h>
#include <time.h>
#include "memory_allocation.h"


//randomly shuffles an array of integers

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
            size_t j = (unsigned int) (drand48()*(i+1));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
    
   
}


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
    
}


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
   srand48(time(NULL));


   float volumeEpsilon=1e-7;
   char *mode="dvn";

   //Set vector indicating label locations and label values
   unsigned char *fixedLabels=calloc(pcount,1); //array that tracks which points have a fixed correct label
   for(j=0;j<m;j++){
      fixedLabels[ind[j]] = 1;
      labels[ind[j]]=val[j];
   }

   float realRatio[lcount];
   float sampleRatio[lcount];
   memset(realRatio,0,lcount*sizeof(float));
   memset(sampleRatio,0,lcount*sizeof(float));

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

   mbos.updateList=calloc(pcount,sizeof(nodeChanged)); //allocate memory for internal workspace
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

void mbo_main_original(int k, int numTrials, float trainingFraction, int maxIters, float stoppingCriterion, float temperature, float upperVolumeMultiplier, float lowerVolumeMultiplier){

    int i,j;
    int maxNeighbors=15;
    srand48(time(NULL));

    if(k>maxNeighbors){
        printf("Number of neighbors: %d must be smaller than the number of max neighbors: %d saved in mnist_vl_neighbor_data \n.", k, maxNeighbors);
        assert(0);
    }
    
    
    
    float volumeEpsilon=1e-7;
    char *mode="dvn";
    
   
    int lcount=10; //number of distinct classes
    int testCount=10000; //10,000 objects in the test set
    int trainCount=60000; //60,000 objects in the training set
    int pcount=testCount+trainCount; //we just combine the two sets together
   //int pcount=10000;
    
    
    
    indexedFloat *nearestNeighborData=getFileData("mnist_vl_neighbor_data"); //precomputed nearest neighbor data containing the 15 nearest neighbors of each image and the distance between them.  The data is stored in the following format nearestNeighborData[15*i+j].index= index of the jth neighbor of i, nearestNeighborData[15*i+j].dist= euclidean distance between i and its jth neighbor.
    
    unsigned char *dataTstLbl=getFileData("mnist_test_set_labels"); //read out the labels from the test file
    unsigned char *dataTrLbl=getFileData("mnist_training_set_labels");  //read out the labels from the training file
    
    int confusionMatrix[lcount*lcount];
    memset(confusionMatrix,0,lcount*lcount*sizeof(int));
    
    
    int incorrect=0;
    float time=0;
    unsigned char *correctLabels=calloc(70000,1);
    for(i=0;i<testCount;i++){
        correctLabels[i]=dataTstLbl[i];
    }
    for(i=0;i<trainCount;i++){
        correctLabels[i+10000]=dataTrLbl[i];
    }
    float etot=0;
    for(j=0;j<numTrials;j++){
        float eThisRound=0;
        int wrongThisRound=0;
        int *labels=calloc(pcount,sizeof(int)); //array that will hold all of the labels
        unsigned char *fixedLabels=calloc(pcount,1); //array that tracks which points have a fixed correct label
        int *numbers=calloc(pcount,sizeof(int)); //used for shuffling
        int counter[lcount];
        memset(counter,0,lcount*sizeof(int));
        float realRatio[lcount];
        float sampleRatio[lcount];
         memset(realRatio,0,lcount*sizeof(float));
         memset(sampleRatio,0,lcount*sizeof(float));
        for(i=0;i<pcount;i++){
            numbers[i]=i; //prepare numbers to be shuffled
        }
        int seen[lcount];
        memset(seen,0,lcount*sizeof(int));
        shuffle(numbers,pcount); //shuffle numbers
        for(i=0;i<pcount;i++){
            realRatio[correctLabels[i]]++;
            if(i/(pcount*1.0)<trainingFraction){
                fixedLabels[numbers[i]]=1;
                sampleRatio[correctLabels[numbers[i]]]++;
            }
             //randomly choose which images will have fixed labels
        }
        int count=0;
        for(i=0;i<pcount;i++){
           count+=fixedLabels[i];
        }
        printf("numlabels=%d\n",count);fflush(stdout);
       
        for(i=0;i<pcount;i++){
            
            //if a label is fixed then we assign it the correct value from the label data.....
            
            if(fixedLabels[i]){
                
                labels[i]=correctLabels[i];
                
                
            }else{
                
                if(mode[1]=='p'){//initialization methods.
                    labels[i]=-1;//initialize by percolation
                }else{
                    labels[i]=rand()%lcount;//random initialization.
                }
            }
            
            
        }
        
       
        generalConvolutionStructure g=create_symmetric_adjacency_matrix(nearestNeighborData, pcount, maxNeighbors, k);
        
        
        
        mbo_struct mbos; //object used by mbo.c to do convolutions
        memset(&mbos,0,sizeof(mbo_struct)); //clear memory
        mbos.nncounts=g.counts; //increasing integer array of length pcount.  nncounts[i+1]-nncounts[i] gives the number of nearest neighbors of node i.
        mbos.indicies=g.neighbors; // entries nncounts[i] through nncounts[i+1]-1 hold the indicies of the nearest neighbors of node i
        mbos.weights=g.connectionStrengths; // entries nncounts[i] through nncounts[i+1]-1 hold the weights of the nearest neighbors of node i
        
        mbos.updateList=calloc(pcount,sizeof(nodeChanged)); //allocate memory for internal workspace
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
        int classCounts[lcount];
        memset(classCounts,0,lcount*sizeof(int));
        for(i=0;i<pcount;i++){
            classCounts[correctLabels[i]]++;
        }
        for(i=0;i<lcount;i++){
            realRatio[i]/=pcount;
            sampleRatio[i]/=(pcount*trainingFraction);
           // printf("%d %f %f %f \n",i,realRatio[i],sampleRatio[i], realRatio[i]/sampleRatio[i]);
        }
        mbos.classCounts=classCounts;
        mbos.k=k;
         int ix,jx;
        clock_t b,e;
        b=clock();
        int lin=0;
        
        if(mode[1]=='v'){
            
            bellman_ford_voronoi_initialization(&mbos, g, 2,lin);
        }else if(mode[1]=='c'){
            for(ix=0;ix<pcount;ix++){
                labels[ix]=correctLabels[ix];
            }
        }
       
        for(ix=0;ix<pcount;ix++){
            for(jx=g.counts[ix];jx<g.counts[ix+1];jx++){
                g.connectionStrengths[jx]=kernel(g.connectionStrengths[jx]);
            }
        }
        

         normalize_matrix(g,pcount); //make the matrix have row sum 1
        
        

        
        
        if(temperature>0){
            eThisRound=run_mbo_with_temperature(mbos, mode[0]);
        }else{
            eThisRound=run_mbo(mbos, mode[0]);
        }
        
        
        
        e=clock();
        etot+=eThisRound;
        int wrongLabels[lcount];
        memset(wrongLabels,0,lcount*sizeof(int));
        
        
        
        
        //compare results to known data
        for(i=0;i<pcount;i++){
            if(correctLabels[i]!=labels[i]){
                wrongLabels[correctLabels[i]]++;
                incorrect++;
                wrongThisRound++;
            }
            confusionMatrix[correctLabels[i]*lcount+labels[i]]++;
            
        }
        
        time+=(e-b)/(CLOCKS_PER_SEC*1.0);
        free(labels);
        free(fixedLabels);
        free(numbers);
        printf("%f %f\n", wrongThisRound*100/(pcount*1.0), eThisRound);
        
    }
    printf("\n");
    for(i=0;i<lcount;i++){
        printf("digit %d confusion:\n",i);
        for(j=0;j<lcount;j++){
            printf("%d ", confusionMatrix[i*lcount+j]);
        }
        printf("\n \n");
    }
    printf("%f %f \n", 100-100*(incorrect)/(pcount*numTrials*1.0), time/numTrials);
    free(dataTrLbl);
    free(dataTstLbl);
    
    
}




