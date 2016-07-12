import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Example script for GPU computations in python.

N = 1024						# number of threads to run.

print "Compiling kernel..."
mod = SourceModule(open("kernel.c").read())		# compile the kernel, the C code 
							# which will be run on each thread.

## Allocate memory on the GPU.

print "Allocating memory..."
v_gpu = cuda.mem_alloc(N * np.float32(0).nbytes)	# value function
p_gpu = cuda.mem_alloc(N * np.int32(0).nbytes)		# policy function
kd_gpu = cuda.mem_alloc(N * np.float32(0).nbytes)	# capital domain
dist_gpu = cuda.mem_alloc(N * np.float32(0).nbytes)	# distance between value functions

# Allocate the corresponding objects on the CPU.
v = np.zeros(N).astype(np.float32)			# value function
p = np.zeros(N).astype(np.int32)			# policy function
kd = np.linspace(1, 100, num = N).astype(np.float32)	# capital domain
dist = np.zeros(N).astype(np.float32)			# distance between value functions

# Copy the CPU objects over to the GPU objects
cuda.memcpy_htod(v_gpu, v)				# value function
cuda.memcpy_htod(p_gpu, p)				# policy function
cuda.memcpy_htod(kd_gpu, kd)				# capital domain
cuda.memcpy_htod(dist_gpu, dist)			# distance between value functions

func = mod.get_function("upd")				# get a	handle for the kernel function to run.

## Run value function iteration

print "Value function iteration..."
dist_out = np.empty_like(dist)  # Store the distance between successive
				# value functions here on the CPU.
while(True):
	# Execute the kernel function. The results are written into the memory
	# pointed at by v_gpu.
	func(v_gpu, p_gpu, kd_gpu, dist_gpu, block = (N, 1, 1), grid = (1, 1))
	cuda.memcpy_dtoh(dist_out, dist_gpu)		# copy the 'distance' vector from 
							# the GPU to the CPU.
	print "\rDistance: " + str(max(dist_out)),	# check	the distance criterion.
	if max(dist_out) < 0.1:
		break


v_out = np.empty_like(v)	# store the value function on the CPU here.
cuda.memcpy_dtoh(v_out, v_gpu)	# copy the value function to memory.
				# result is in v_out.
