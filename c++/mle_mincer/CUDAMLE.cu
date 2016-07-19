//  Created by Rodrigo Azuero on 2/24/16.
//  Copyright (c) 2016 Rodrigo Azuero Melo. All rights reserved.
//


#include <algorithm>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <string>
#include <stdlib.h>
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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <unistd.h>
#include <nlopt.hpp>
using std::vector;
using namespace std;


__device__ __host__ float normpdfESTANDAR(float D){
        float term=D*D;
        float exp=-term;
        float num=expf(exp);
        float den=powf(2*3.141592,0.5);
        float ans=num/den;
        return(ans);
}
//----------
//0.0. Wages
//------------
//===================================================================================//


__device__ __host__  float F_predwageD(float bbeta0, float bbeta1, float bbeta2, float bbeta3,  float Schooling, float Age){
    //1. Defining the output
    float wage=bbeta0+bbeta1*Schooling+bbeta2*Age+bbeta3*Age*Age;
    return(wage);
}



//=================================================================================
//1.0. Likelihood of Wages
//=================================================================================
__host__ __device__ float F_likelihood_wageD(float bbeta0, float bbeta1, float bbeta2, float bbeta3, float stdwage, float Schooling, float Age, float Wage){
    
    //1. Obtaining the predicted Wage:
    float predwage=F_predwageD(bbeta0,bbeta1,bbeta2,bbeta3,Schooling,Age);
    float likelihood_wage=Wage-predwage;
    likelihood_wage=likelihood_wage/stdwage;
    likelihood_wage=normpdfESTANDAR(likelihood_wage);
    
    likelihood_wage=likelihood_wage/stdwage;
    if(likelihood_wage==0){
        likelihood_wage=0.000001;
    }
    likelihood_wage=logf(likelihood_wage);
    return(likelihood_wage);
}



//----------------------------------------------------
//2. Block of defining likelihood for all the dataset
//---------------------------------------------------
 
__device__ __host__ float F_likelihoodSCALARD(const float* par_likelihood,
                                                float  Schooling,
                                                float Age,
                                                float Wage){


     //float aalpha1m=FT_expD(par_likelihood[0]);
    float bbeta0=(par_likelihood[0]);
    float bbeta1=(par_likelihood[1]);
    float bbeta2=(par_likelihood[2]);
    float bbeta3=(par_likelihood[3]);
    float std=exp(par_likelihood[4]);

    float iilike=F_likelihood_wageD(bbeta0, bbeta1,  bbeta2,  bbeta3,  std,  Schooling,  Age,  Wage);
    return(iilike);
}
    

//Functions of loading all the likelihoods. Looping over individuals for the function F_likelihoodSCALARD
__global__ void KernelLikelihood(
    const float* par_likelihood, 
    const float *V_Schooling,
    const float *V_Age,
    const float * V_Wage, float *output){

    //Indexing
    int ii=threadIdx.x+blockIdx.x*blockDim.x;
    if(ii<1000){
        //Loading
        float Schooling=V_Schooling[ii];
        float Age=V_Age[ii];
        float Wage=V_Wage[ii];
        float iilik=F_likelihoodSCALARD(par_likelihood,Schooling,Age,Wage);
        output[ii]=iilik;
    }
    if(ii>=1000){
        output[ii]=0;
    }
}






//Attempt of likelihood
float F_likelihood_FIN(const double *PARDOUBLE){ //In this function I will call the kernel
    //0. Define the current directory


    int THREADSPERBLOCK=200;
    int TOTALTHREADS=800;
    int SIZEOBS=800;
    int NVAR=3;
    int NPAR=5;
    //Passing copies of the parameters to the device
    float *Schooling, *Age, *Wage, *output, *PAR;
    float *d_Schooling, *d_Age, *d_Wage, *d_output, *d_PAR;
    
    size_t sizePARFLOAT=NPAR*sizeof(float);
    size_t dimobsFLOAT=TOTALTHREADS*sizeof(float);


    //Allocate space for device copies of Parameters
    cudaMalloc((void **)&d_PAR,sizePARFLOAT);
    cudaMalloc((void **)&d_Schooling,dimobsFLOAT);
    cudaMalloc((void **)&d_Age,dimobsFLOAT);
    cudaMalloc((void **)&d_Wage,dimobsFLOAT);
    cudaMalloc((void **)&d_output,dimobsFLOAT);
    
    //*Cliveswithfather12, *Cliveswithmother12, *Hhchores12, *Meffort12, *Feffort12, *Mage12, *Fage12, *Cedad_meses12, *Ctestsfactorsss2012, *Ctestsfactor2ss_10, *Myrschool12, *Fyrschool12, *Ffraclabor12, *Mfraclabor12, *Mwage12, *Fwage12, *Mnly12, *Fnly12, *Cfactorinv12, *Cchildcare12, *Ccareskills12, *Hbarg, *Feffort10, *Meffort10, *Cfactorinv10, *Cbirthfactor, *Mwage10, *Fwage10, *Mnly10, *Fnly10, *Cedad_meses10, *Cliveswithfather10, *Cliveswithmother10, *Mage10, *Fage10, *Mfraclabor10, *Ffraclabor10, *Cchildcare10, *Hchildcareobs, *PG, *MTJH,*MRATIO,*Unemployment, *Wageratio,*Distance,*Magegroup10,*Magegroup12,*Hmemberstotal10,*Hmemberstotal12,*CSTDtepsi_pb_coo10,*CSTDtepsi_pb_len10,*CSTDtepsi_pb_mot10,*CSTDtvip_pb10,*CSTDcbcl1_pb_110,*CSTDcbcl1_pb_210,*CSTDcbcl1_pb_310,*CSTDcbcl1_pb_410,*CSTDcbcl1_pb_510,*CSTDcbcl1_pb_610,*CSTDcbcl1_pb_710,*CSTDtadi_pb_cog12, *CSTDtadi_pb_mot12, *CSTDtadi_pb_len12, *CSTDtadi_pb_se12, *CSTDbt_112, *CSTDbt_212, *CSTDbt_312, *CSTDbt_412, *CSTDbt_512, *CSTDbt_t12,  *CSTDhtks_st12, *CSTDbdst_st12, *CSTDppvt_t12,  *Ccondpregg7b, *Ccondpregg8b, *Ccondpregg9, *Ccondpregg11b, *Ccondpregg24, *Ccondpregg23,*Cwais_pb_num, *Cwais_pb_vo, *Cbfi_pb_ama,  *Cbfi_pb_ape, *Cbfi_pb_ext, *Cbfi_pb_neu, *Cbfi_pb_res, *Cpsi_pb_total,*Hbargg2a, *Hbargg2b, *Hbargg2c, *Hbargg2d, *Hbargg2e, *Hbargg2f, *Hbargg2g, *Hbargg2h, *Hbargg2i, *Hbargg2j, *Hcaresacwom, *Hcaresacman, *Cinvf11a, *Cinvf11b,  *Cinvf11c, *Cinvf11d, *Cinvf11e, *Cinvf11f, *Cinvf11g, *Cinvf11h, *Cinvf11i, *Cinvf11j, *Cinvf11k,*Csharesbedroomhowmany12,  *Csharesbedhowmany12, *g42_a2, *g42_b2, *g42_c2, *g42_d2, *g42_e2, *g42_f2,*g42_a1, *g42_b1, *g42_c1, *g42_d1, *g42_e1, *g42_f1,*f21a_p_t, *f21b_p_t, *f21c_p_t, *f21d_p_t, *f21e_p_t, *f21f_p_t, *f21g_p_t, *f21h_p_t, *f21i_p_t, *f21j_p_t, *f21k_p_t, *f21l_p_t, *f21m_p_t, *f21n_p_t,*f21a_m_t, *f21b_m_t, *f21c_m_t, *f21d_m_t, *f21e_m_t, *f21f_m_t, *f21g_m_t, *f21h_m_t, *f21i_m_t, *f21j_m_t, *f21k_m_t, *f21l_m_t, *f21m_m_t, *f21n_m_t;
    
    
    Schooling=(float *)malloc(dimobsFLOAT);
    Age=(float *)malloc(dimobsFLOAT);
    Wage=(float *)malloc(dimobsFLOAT);
    output=(float *)malloc(dimobsFLOAT); 
    PAR=(float *)malloc(sizePARFLOAT);
    


    //Loading the dataset
    std::ifstream theFile ("DATACUDA.csv");
    double MYARRAY[SIZEOBS+1][NVAR+1];

    std::string line;
    std::vector<std::vector<std::string> > values;
    int it=0;
    int it2=0;
    std::string line_value;
    std::vector<std::string> line_values;
    std::stringstream ss;
    while(std::getline(theFile, line))
    {
        
        ss<<line;
        //std::stringstream ss(line);
        //std::string item;
        //cout <<  << "linevalprev"<<endl;
        while(std::getline(ss, line_value, ','))
        {
            line_values.push_back(line_value);
            MYARRAY[it][it2] = ::atof(line_value.c_str());
            
            //MYARRAY[it][it2]=std::stod (line_value); only for c++11 compi
            it2=it2+1;
            if (it2==NVAR){ //later change 4 for
                it2=0;
            }
        }
        values.push_back(line_values);
        
        //For c++11 used values.emplace_back(line_values);
        
        //cout << line_value<< "line_value2"<< endl;
        it=it+1;
        //Free the string types
        line_value.clear();
        line_values.clear();
        ss.clear();
        
    }

    //Allocate initialization 
    for (int it=0; it<SIZEOBS;it++){
        Age[it]=MYARRAY[it][0];
        Schooling[it]=MYARRAY[it][1];
        Wage[it]=MYARRAY[it][2];
        
    }//Finishmloading data
    cout << " ---- " << endl;
    //Allocate parameters
    for (int it=0; it<NPAR;it++){
        PAR[it]=(float)(PARDOUBLE[it]);
        cout <<PAR[it] << " par[it]"<< endl;
        cout << it  << " it "<< endl;
    }
    cout << " ---- " << endl;
    //Cuda copies double
    cudaMemcpy(d_Schooling,Schooling,dimobsFLOAT,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Age,Age,dimobsFLOAT,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_Wage,Wage,dimobsFLOAT,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_PAR,PAR,sizePARFLOAT,cudaMemcpyHostToDevice);

    //Run kernel
    KernelLikelihood<<<TOTALTHREADS/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_PAR,d_Schooling,d_Age,d_Wage,d_output);

    cudaMemcpy(output,d_output,dimobsFLOAT,cudaMemcpyDeviceToHost);
    free(Wage);
    free(output);
    free(Age);
    free(Schooling);
    free(PAR);
    cudaFree(d_Wage);
    cudaFree(d_output);
    cudaFree(Age);
    cudaFree(d_Schooling);
    cudaFree(d_PAR);
    float sumA=0;
    for(int ii=0; ii<SIZEOBS;ii=ii+1){
        sumA+=output[ii];
        //cout << output[ii] << " output[ii] " << endl;

        //cout << Fwagepred[ii] << "wagepredii"<<endl;
    }
    cout << sumA << " FLIK" << endl;
    sumA=-sumA;
    return(sumA);
}


int iterat=0;
double FLIKMINIMIZED(unsigned n, const double *x, double *grad, void *FLIKMNIMZED_data){
    double result=F_likelihood_FIN(x);
    cout << result << " evalF"<< endl;
    iterat=iterat+1;
    printf("Iteration=(%d); Feval=%0.10g\n", iterat, result);
    return(result);
}


int  main (const int           argc,
           const char * const  argv[])

{
	

        std::string line;
        std::vector<std::vector<std::string> > values;
        vector<double> PARRA;
        PARRA.resize(5);
        std::ifstream PARAMETERS ("INITIALGUESS.csv");
        
        std::string linePARAM;
        int itpar=0;
        while(std::getline(PARAMETERS, line))
        {
            PARRA[itpar]=::atof (line.c_str());
            //PAR[itpar]=std::stod (line); only C++11s
            
            
            
            itpar=itpar+1;
        }
        //And the ones that were not stored
        
        //Loading The parameters into PARDOUBLE
        double PARDOUBLE[5]={};
        for (int ii=0;ii<5;ii=ii+1){
            PARDOUBLE[ii]=PARRA[ii];
        }
        
        
        cout << " before evaluating likelihood  " << endl;
        
        cout << F_likelihood_FIN(PARDOUBLE) << " LIKELIHOOD " << endl;
        int optimize=1;
        if(optimize==1){
            nlopt_opt opt3;
            opt3 = nlopt_create(NLOPT_LN_SBPLX, 5); /* algorithm and dimensionality */
            nlopt_set_min_objective(opt3, FLIKMINIMIZED, NULL);
            //nlopt_set_xtol_rel(opt3, 1.0e-4);
            //nlopt_set_maxtime(opt3,  10000);
            nlopt_set_maxeval(opt3,2000);
            double x3[5]={0};
            for(int it=0;it<5;it=it+1){
                x3[it]=PARDOUBLE[it];
            }
            
            double minf3; /* the minimum objective value, upon return */
            nlopt_optimize(opt3, x3, &minf3);
            if (nlopt_optimize(opt3, x3, &minf3) < 0) {
                printf("nlopt failed!\n");
            }
            else {
                printf("found minimum at f(%g) = %0.10g\n", x3[0], minf3);
            }
            
            //Saving the vector of optimal parameters in csv file
            ofstream optparam("PARAMETERSFOUND.csv");
            for (int it=0;it<5;it=it+1){
                optparam<<x3[it] << endl;
            }
            optparam.close();
            
            
            nlopt_destroy(opt3);
        }
    

    



	return 0;
    
}