#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
//#include <sys/time.h>
#include <string.h>




typedef struct{
    
    int *neighbors;
    float *connectionStrengths;
}knnConvolutionStructure;




typedef struct{
    int *neighbors;
    float *connectionStrengths;
    int *counts;
    
}generalConvolutionStructure;

typedef struct{
    int index;
    float dist;
}indexedFloat;

typedef struct{
    float x;
    float y;
}mpoint;

typedef struct{
    int to;
    int from;
    int index;
    
}changedClass;

typedef struct{
    int index;
    int to;
    int from;
}nodeChanged;

typedef struct{
    nodeChanged *updateList;
    int *indicies;
    float *weights;
    int *nncounts;
    float *surfaceTensions;
    unsigned char *fixedLabels;
    int *labels;
    float *linear;
    int *linearBest;
    int *classCounts;
    float stoppingCriterion;
    float epsilon;
    float temperature;
    float convexityParameter;
    float upperVolumeMultiplier;
    float lowerVolumeMultiplier;
    int k;
    int pcount;
    int lcount;
    int maxIters;
    char singleGrowth;
}mbo_struct;

void normalize_matrix_with_linear(generalConvolutionStructure g, float *linear, int pcount, int lcount);

void free_generalConvolutionStructure(generalConvolutionStructure g);

generalConvolutionStructure create_symmetric_adjacency_matrix(indexedFloat *neighborData, int pcount, int maxNeighbors, int k);

generalConvolutionStructure create_symmetric_matrix(float (*kernel)(float), indexedFloat *neighborData, int pcount, int maxNeighbors, int k);
void normalize_matrix(generalConvolutionStructure g, int pcount);

void voronoi_initialization(mbo_struct *mbos, generalConvolutionStructure g, int maxNeighbors, float distanceExponent, int lin);
void bellman_ford_voronoi_initialization(mbo_struct *mbos, generalConvolutionStructure g, float distanceExponent, int lin);


void reweight_fidelity_nodes(mbo_struct mbos);

generalConvolutionStructure create_dual_convolution_structure(mbo_struct mbos);

float run_mbo(mbo_struct mbos, char mode);
float run_mbo_distance(mbo_struct mbos, float *graphDistances, char mode);

float run_mbo_efficient(mbo_struct mbos, char mode);
float run_mbo_single_growth_efficient(mbo_struct mbos, char mode);
float run_mbo_with_temperature(mbo_struct mbos, char mode);
float run_mbo_convexity(mbo_struct mbos);
void run_mbo_sgd(mbo_struct mbos);

void run_k_means_mbo(mbo_struct mbos, float *rawData, int dataDim, char mode);
