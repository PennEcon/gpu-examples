//
//  Created by David Zarruk Valencia on June, 2016.
//  Copyright (c) 2016 David Zarruk Valencia. All rights reserved.
//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

#include <algorithm>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <unistd.h>
#include <nlopt.hpp>
#include <typeinfo>
#include <ctime>
using std::vector;
using namespace std;


//======================================
//         Utility function
//======================================

__device__ float u(const float c, const float ssigma){
   
  float utility = 0.0; 
   
  utility = pow(c, 1-ssigma) / (1-ssigma);
 
  if(c <= 0){
    utility = pow(-10.0, 5.0);
  }
   
  return(utility);
}




//======================================
//         Grids
//======================================

void gridx(const int nx, const float xmin, const float xmax, float* xgrid){
  
  const float size = nx;
  const float xstep = (xmax - xmin) /(size - 1);
  float it = 0;
  
  for(int i = 0; i < nx; i++){
    xgrid[i] = xmin + it*xstep;
    it++;
  }
}


void gride(const int ne, const float ssigma_eps, const float llambda_eps, const float m, float* egrid){
  
  // This grid is made with Tauchen (1986)
  const float size = ne;
  const float ssigma_y = sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2)));
  const float estep = 2*ssigma_y*m / (size-1);
  float it = 0;
  
  for(int i = 0; i < ne; i++){
    egrid[i] = exp(-m*sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2))) + it*estep);
    it++;
  }
}

float normCDF(const float value){
  return 0.5 * erfc(-value * M_SQRT1_2);
}



void eprob(const int ne, const float ssigma_eps, const float llambda_eps, const float m, const float* egrid, float* P){
  
  // This grid is made with Tauchen (1986)
  // P is: first ne elements are transition from e_0 to e_i,
  //       second ne elementrs are from e_1 to e_i, ...
  const float w = egrid[1] - egrid[0];
  
  for(int j = 0; j < ne; j++){
    for(int k = 0; k < ne; k++){
      if(k == 0){
        P[j*ne + k] = normCDF((egrid[k] - llambda_eps*egrid[j] + (w/2))/ssigma_eps);
      } else if(k == ne-1){
        P[j*ne + k] = 1 - normCDF((egrid[k] - llambda_eps*egrid[j] - (w/2))/ssigma_eps);
      } else{
        P[j*ne + k] = normCDF((egrid[k] - llambda_eps*egrid[j] + (w/2))/ssigma_eps) - normCDF((egrid[k] - llambda_eps*egrid[j] - (w/2))/ssigma_eps);
      }
    }
  }
}


//======================================
//         Survival probabilities
//======================================


float pi(int age){
  
  float probability = 1;
  
  return(probability);
}



//======================================
//         Auxiliary
//======================================

float maximum(float a, float b){
  
  float max = a;
  if(b>=a){
    max = b;
  }
  return(max);
}

float minimum(float a, float b){
  
  float min = a;
  if(b<=a){
    min = b;
  }
  return(min);
}


//======================================
//         Parameter structure
//======================================

class parameters{
 public:
  int nx; 
  float xmin; 
  float xmax;
  int ne; 
  float ssigma_eps; 
  float llambda_eps; 
  float m; 

  float ssigma; 
  float eeta; 
  float ppsi; 
  float rrho; 
  float llambda; 
  float bbeta;
  int T;
  float r;
  float w;

  void load(const char*);
};



//======================================
//         MAIN  MAIN  MAIN
//======================================

__global__ void Vmaximization(const parameters params, const float* xgrid, const float* egrid, const float* P, const int age, float* V, int* PolX){
  
  // Recover the parameters
  const int nx              = params.nx; 
  const int ne              = params.ne; 
  const float ssigma        = params.ssigma; 
  const float bbeta         = params.bbeta;
  const int T               = params.T;
  const float r             = params.r;
  const float w             = params.w;

  // Recover state variables from indices
  const int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  const int ie  = threadIdx.y;

  float expected;
  float utility;
  float cons;
  float VV = pow(-10.0,5.0);
  int ixpopt = 0;
  
  for(int ixp = 0; ixp < nx; ixp++){

    expected = 0.0;
    if(age < T-1){
      for(int iep = 0; iep < ne; iep++){
        expected = expected + P[ie*ne + iep]*V[(age+1)*nx*ne + ixp*ne + iep];
      }
    }

    cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];
 
    utility = u(cons, ssigma) + bbeta*expected;

    if(utility >= VV){
      VV = utility;
      ixpopt = ixp;
    }

    utility = 0.0;
  }

  V[age*nx*ne + ix*ne + ie] = VV;
  PolX[age*nx*ne + ix*ne + ie] = ixpopt;

}



int main()
{ 
  // Grids
  const int nx              = 960; 
  const float xmin          = 0.1; 
  const float xmax          = 4.0;
  const int ne              = 16; 
  const float ssigma_eps    = 0.02058; 
  const float llambda_eps   = 0.99; 
  const float m             = 1.5; 

  // Parameters
  const float ssigma        = 2; 
  const float eeta          = 0.36; 
  const float ppsi          = 0.89; 
  const float rrho          = 0.5; 
  const float llambda       = 1; 
  const float bbeta         = 0.97;
  const int T             = 3;

  // Prices
  const float r             = 0.07;
  const float w             = 5;

  parameters params = {nx, xmin, xmax, ne, ssigma_eps, llambda_eps, m, ssigma, eeta, ppsi, rrho, llambda, bbeta, T, r, w};

  // Pointers to variables in the DEVICE memory
  float *V, *X, *E, *P;
  int *PolX;
  size_t sizeX = nx*sizeof(float);
  size_t sizeE = ne*sizeof(float);
  size_t sizeP = ne*ne*sizeof(float);
  size_t sizeV = T*ne*nx*sizeof(float);
  size_t sizePolX = T*ne*nx*sizeof(int);

  cudaMalloc((void**)&X, sizeX);
  cudaMalloc((void**)&E, sizeE);
  cudaMalloc((void**)&P, sizeP);
  cudaMalloc((void**)&V, sizeV);
  cudaMalloc((void**)&PolX, sizePolX);

  // Parameters for CUDA: cada block tiene ne columnas, y una fila que representa un valor de x
  //                      Hay nx blocks 
  //                      Cada layer es una edad >= hay 80 layers
  
  const int block_size = 512/ne;
  dim3 dimBlock(block_size, ne);
  dim3 dimGrid(nx/block_size, 1);


  // Variables in the host have "h" prefix
  // I create the grid for X
  float hxgrid[nx];
  gridx(nx, xmin, xmax, hxgrid);

  // I create the grid for E and the probability matrix
  float hegrid[ne];  
  float hP[ne*ne];
  gride(ne, ssigma_eps, llambda_eps, m, hegrid);
  eprob(ne, ssigma_eps, llambda_eps, m, hegrid, hP);



  float *hV;
  int *hPolX;
  hV = (float *)malloc(sizeV);
  hPolX = (int *)malloc(sizePolX);

  // Copy matrices from host (CPU) to device (GPU) memory
  cudaMemcpy(X, hxgrid, sizeX, cudaMemcpyHostToDevice);
  cudaMemcpy(E, hegrid, sizeE, cudaMemcpyHostToDevice);
  cudaMemcpy(P, hP, sizeP, cudaMemcpyHostToDevice);
  cudaMemcpy(V, hV, sizeV, cudaMemcpyHostToDevice);
  cudaMemcpy(PolX, hPolX, sizePolX, cudaMemcpyHostToDevice);


  // Time the GPU startup overhead
  clock_t t;
  t = clock();

  for(int age=T-1; age>=0; age--){
    Vmaximization<<<dimGrid,dimBlock>>>(params, X, E, P, age, V, PolX); 
    cudaDeviceSynchronize();
  }

  t = clock() - t;
  std::cout << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

  cudaMemcpy(hV, V, sizeV, cudaMemcpyDeviceToHost);
  cudaMemcpy(hPolX, PolX, sizePolX, cudaMemcpyDeviceToHost);

  // Free variables in device memory
  cudaFree(V);
  cudaFree(PolX);
  cudaFree(X);
  cudaFree(E);
  cudaFree(P);

  // I print out some values of the value function
  for(int i = 20; i<40; i++){
    std::cout << hV[i] << std::endl;
  }



  // For the matter of comparison, I do it without GPU parallelization

  float *hV2;
  int *hPolX2;
  hV2 = (float *)malloc(sizeV);
  hPolX2 = (int *)malloc(sizePolX);

  float expected;
  float utility;
  float cons;
  float VV = pow(-10.0,5.0);
  int ixpopt = 0;
  
  t = clock();

  for(int age=T-1; age>=0; age--){
    for(int ix = 0; ix<nx; ix++){
      for(int ie = 0; ie<ne; ie++){
        for(int ixp = 0; ixp < nx; ixp++){

          expected = 0.0;
          if(age < T-1){
            for(int iep = 0; iep < ne; iep++){
              expected = expected + hP[ie*ne + iep]*hV2[(age+1)*nx*ne + ixp*ne + iep];
            }
          }

          cons  = (1 + r)*hxgrid[ix] + hegrid[ie]*w - hxgrid[ixp];

          utility = std::pow(cons, 1-ssigma) / (1-ssigma) + bbeta*expected;

          if(cons <= 0){
            utility = std::pow(-10.0, 5.0);
          }

          if(utility >= VV){
            VV = utility;
            ixpopt = ixp;
          }

          utility = 0.0;
        }

        hV2[age*nx*ne + ix*ne + ie] = VV;
        hPolX2[age*nx*ne + ix*ne + ie] = ixpopt;

        VV = pow(-10.0,5.0);
        ixpopt = 0;

      }
    }
  }
  
  t = clock() - t;
  std::cout << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;


  std::cout << "When manually..." << std::endl;

  // I print out some values of the value function
  for(int i = 20; i<40; i++){
    std::cout << hV2[i] << std::endl;
  }

  return 0;
}
