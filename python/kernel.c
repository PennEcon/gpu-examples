// ---------------------------------------------------------------------------
// Kernel for simple value function iteration example. This is the C code that
// will run on each core on the GPU.
// ---------------------------------------------------------------------------

// Include CUDA's math library
#include "math.h"

__global__ void upd(float *v, float *p, float *kd, float *dist) {
        float best_val = -INFINITY;
        float curr_val;
	for (int i = 0; i < 1024l; ++i) {
                curr_val = kd[threadIdx.x] > kd[i]/1.02 ? log(kd[threadIdx.x] - kd[i]/1.02) + 0.99*v[i] : -INFINITY;
                if (curr_val > best_val) {
                        best_val = curr_val;
                        p[threadIdx.x] = i;
                }
        }
        dist[threadIdx.x] = abs(best_val - v[threadIdx.x]);
        v[threadIdx.x] = best_val;
}

