#include "mbo_convolution.h"
#include <assert.h>
#include <time.h>
#include <assert.h>


void mbo_main(int *labels, int *II, int *J, float *W, int *ind, int *val, int *classCounts, bool prog, int pcount, int M, int m, int lcount, int maxIters, float stoppingCriterion, float temperature, float upperVolumeMultiplier, float lowerVolumeMultiplier);

void mbo_main_original(int k, int numTrials, float trainingFraction, int maxIters, float StoppingCriterion, float temperature, float upperVolumeMultiplier, float lowerVolumeMultiplier);
