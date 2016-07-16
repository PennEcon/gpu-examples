// ---------------------------------------------------------------------------
// Kernel for simple value function iteration example. This is the C code that
// will run on each core on the GPU. It solves a basic Bellman equation
// example with consumption and capital.
// ---------------------------------------------------------------------------

// Include CUDA's math library
#include "math.h"

__global__ void upd(float *v, float *p, float *kd, float *dist) {
        float bestValue = -INFINITY;
        float currentValue;

	// loop through the possible capital choices and find the best one
	for (int i = 0; i < 1024l; ++i) {
                currentValue = kd[threadIdx.x] > kd[i]/1.02 ? log(kd[threadIdx.x] - kd[i]/1.02) + 0.99*v[i] : -INFINITY;
                if (currentValue > bestValue) {
			bestValue = currentValue;
                        p[threadIdx.x] = i;
                }
        }

	// perform a bunch of calculations to slow down
	int now = 1;
	for (int i =0; i < 1000000; ++i) {
		now *= i;
	}
	p[threadIdx.x] = now;

	// update the distance criterion
        dist[threadIdx.x] = abs(bestValue - v[threadIdx.x]);

	// assign the best value to memory
	v[threadIdx.x] = bestValue;
}

