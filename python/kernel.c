// ---------------------------------------------------------------------------
// Kernel for simple value function iteration example. This is the C code that
// will run on each core on the GPU. It solves a basic Bellman equation
// example with consumption and capital.
// ---------------------------------------------------------------------------

// Include CUDA's math library
#include "math.h"

__global__ void upd(float *v, float *p, float *kd, float *dist) {
        float best_val = -INFINITY;
        float curr_val;
	// loop through the possible capital choices and find the best one
	for (int i = 0; i < 1024l; ++i) {
                curr_val = kd[threadIdx.x] > kd[i]/1.02 ? log(kd[threadIdx.x] - kd[i]/1.02) + 0.99*v[i] : -INFINITY;
                if (curr_val > best_val) {
			best_val = curr_val;
                        p[threadIdx.x] = i;
                }
        }

	// perform a bunch of calculations
	int now = 1;
	for (int i =0; i < 1000000; ++i) {
		now *= i;
	}
	p[threadIdx.x] = now;

	// update the distance criterion
        dist[threadIdx.x] = abs(best_val - v[threadIdx.x]);

	v[threadIdx.x] = best_val;
}

