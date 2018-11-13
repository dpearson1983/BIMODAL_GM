#ifndef _BISPECMOD_H_
#define _BISPECMOD_H_

#include <vector>
#include <cmath>
#include <cuda.h>
#include <vector_types.h>
#include "gpuerrchk.h"

#define TWOSEVENTHS 0.285714285714286
#define THREESEVENTHS 0.428571428571429
#define FOURSEVENTHS 0.571428571428571
#define FIVESEVENTHS 0.714285714285714
#define ONETHIRD 0.333333333333333
#define PI 3.1415926535897932384626433832795

__constant__ double4 d_Pk[128]; //  4096 bytes
__constant__ double4 d_n[877];  // 22528 bytes
__constant__ double4 d_Q3[877]; // 22528 bytes
__constant__ double d_w[32];    //   256 bytes
__constant__ double d_x[32];    //   256 bytes
__constant__ double d_p[15];    //   120 bytes
__constant__ double d_aF[9];    //    72 bytes
__constant__ double d_aG[9];    //    72 bytes
__constant__ double d_knl;      //     8 bytes
// Total constant memory usage:    49936 bytes

#define b1 d_p[0]
#define b2 d_p[1]
#define f d_p[2]
#define sigma_8 d_p[3]
#define a_para d_p[4]
#define a_perp d_p[5]
#define sigma_v d_p[6]
#define a0 d_p[7]
#define a1 d_p[8]
#define a2 d_p[9]
#define a3 d_p[10]
#define c0 d_p[11]
#define c1 d_p[12]
#define c2 d_p[13]
#define c3 d_p[14]

const double w_i[] = {0.096540088514728, 0.096540088514728, 0.095638720079275, 0.095638720079275,
                     0.093844399080805, 0.093844399080805, 0.091173878695764, 0.091173878695764,
                     0.087652093004404, 0.087652093004404, 0.083311924226947, 0.083311924226947,
                     0.078193895787070, 0.078193895787070, 0.072345794108849, 0.072345794108849,
                     0.065822222776362, 0.065822222776362, 0.058684093478536, 0.058684093478536,
                     0.050998059262376, 0.050998059262376, 0.042835898022227, 0.042835898022227,
                     0.034273862913021, 0.034273862913021, 0.025392065309262, 0.025392065309262,
                     0.016274394730906, 0.016274394730906, 0.007018610009470, 0.007018610009470};

const double x_i[] = {-0.048307665687738, 0.048307665687738, -0.144471961582796, 0.144471961582796,
                     -0.239287362252137, 0.239287362252137, -0.331868602282128, 0.331868602282128,
                     -0.421351276130635, 0.421351276130635, -0.506899908932229, 0.506899908932229,
                     -0.587715757240762, 0.587715757240762, -0.663044266930215, 0.663044266930215,
                     -0.732182118740290, 0.732182118740290, -0.794483795967942, 0.794483795967942,
                     -0.849367613732570, 0.849367613732570, -0.896321155766052, 0.896321155766052,
                     -0.934906075937739, 0.934906075937739, -0.964762255587506, 0.964762255587506,
                     -0.985611511545268, 0.985611511545268, -0.997263861849481, 0.997263861849481};
                     
const double a_F[] = {0.484, 3.740, -0.849, 0.392, 1.013, -0.575, 0.128, -0.722, -0.926};

const double a_G[] = {3.599, -3.879, 0.518, -3.588, 0.336, 7.431, 5.022, -3.104, -0.484};
                     
__device__ double pk_spline_eval(double &k) {
    int i = (k - d_Pk[0].x)/(d_Pk[1].x - d_Pk[0].x);
    
    double Pk = (d_Pk[i + 1].z*(k - d_Pk[i].x)*(k - d_Pk[i].x)*(k - d_Pk[i].x))/(6.0*d_Pk[i].w)
              + (d_Pk[i].z*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k))/(6.0*d_Pk[i].w)
              + (d_Pk[i + 1].y/d_Pk[i].w - (d_Pk[i + 1].z*d_Pk[i].w)/6.0)*(k - d_Pk[i].x)
              + (d_Pk[i].y/d_Pk[i].w - (d_Pk[i].w*d_Pk[i].z)/6.0)*(d_Pk[i + 1].x - k);
              
    return Pk;
}

__device__ double n_spline_eval(double &k) {
    int i = (k - d_n[0].x)/(d_n[1].x - d_n[0].x);
    
    double n = (d_n[i + 1].z*(k - d_n[i].x)*(k - d_n[i].x)*(k - d_n[i].x))/(6.0*d_n[i].w)
              + (d_n[i].z*(d_n[i + 1].x - k)*(d_n[i + 1].x - k)*(d_n[i + 1].x - k))/(6.0*d_n[i].w)
              + (d_n[i + 1].y/d_n[i].w - (d_n[i + 1].z*d_n[i].w)/6.0)*(k - d_n[i].x)
              + (d_n[i].y/d_n[i].w - (d_n[i].w*d_n[i].z)/6.0)*(d_n[i + 1].x - k);
              
    return n;
}

__device__ double Q3_spline_eval(double &k) {
    int i = (k - d_Q3[0].x)/(d_Q3[1].x - d_Q3[0].x);
    
    double Q3 = (d_Q3[i + 1].z*(k - d_Q3[i].x)*(k - d_Q3[i].x)*(k - d_Q3[i].x))/(6.0*d_Q3[i].w)
              + (d_Q3[i].z*(d_Q3[i + 1].x - k)*(d_Q3[i + 1].x - k)*(d_Q3[i + 1].x - k))/(6.0*d_Q3[i].w)
              + (d_Q3[i + 1].y/d_Q3[i].w - (d_Q3[i + 1].z*d_Q3[i].w)/6.0)*(k - d_Q3[i].x)
              + (d_Q3[i].y/d_Q3[i].w - (d_Q3[i].w*d_Q3[i].z)/6.0)*(d_Q3[i + 1].x - k);
              
    return Q3;
}

__device__ double FoG(double &k_1, double &k_2, double &k_3, double &mu_1, double &mu_2, double &mu_3) {
    double kmus = k_1*k_1*mu_1*mu_1 + k_2*k_2*mu_2*mu_2 + k_3*k_3*mu_3*mu_3;
    double denom = 1.0 + 0.5*kmus*kmus*sigma_v*sigma_v;
    return 1.0/(denom*denom);
}

__device__ double Z_1(double &mu_i) {
    return b1 + mu_i*mu_i*f;
}

__device__ double a_nkaF(double &n, double &k) {
    double q = k/d_knl;
    double Q3 = Q3_spline_eval(k);
    double powqa = __powf(q*d_aF[0], n + d_aF[1]);
    return (1.0 + __powf(sigma_8, d_aF[5])*sqrt(0.7*Q3)*powqa)/(1.0 + powqa);
}

__device__ double b_nkaF(double &n, double &k) {
    double q = k/d_knl;
    return (1.0 + 0.2*d_aF[2]*(n + 3.0)*__powf(q*d_aF[6],n + 3.0 + d_aF[7]))/(1.0 + __powf(q*d_aF[6],n + 3.5 + d_aF[7]));
}

__device__ double c_nkaF(double &n, double &k) {
    double q = k/d_knl;
    double denom = 1.5 + (n + 3.0)*(n + 3.0)*(n + 3.0)*(n + 3.0);
    return (1.0 + ((4.5*d_aF[3])/denom)*__powf(q*d_aF[4], n + 3 + d_aF[8]))/(1.0 + __powf(q*d_aF[4], n + 3.5 + d_aF[8]));
}

__device__ double a_nkaG(double &n, double &k) {
    double q = k/d_knl;
    double Q3 = Q3_spline_eval(k);
    double powqa = __powf(q*d_aG[0], n + d_aG[1]);
    return (1.0 + __powf(sigma_8, d_aG[5])*sqrt(0.7*Q3)*powqa)/(1.0 + powqa);
}

__device__ double b_nkaG(double &n, double &k) {
    double q = k/d_knl;
    return (1.0 + 0.2*d_aG[2]*(n + 3.0)*__powf(q*d_aG[6],n + 3.0 + d_aG[7]))/(1.0 + __powf(q*d_aG[6],n + 3.5 + d_aG[7]));
}

__device__ double c_nkaG(double &n, double &k) {
    double q = k/d_knl;
    double denom = 1.5 + (n + 3.0)*(n + 3.0)*(n + 3.0)*(n + 3.0);
    return (1.0 + ((4.5*d_aG[3])/denom)*__powf(q*d_aG[4], n + 3 + d_aG[8]))/(1.0 + __powf(q*d_aG[4], n + 3.5 + d_aG[8]));
}

// __device__ double a_nkaF(double &n, double &k) {
//     return 1.0;
// }
// 
// __device__ double b_nkaF(double &n, double &k) {
//     return 1.0;
// }
// 
// __device__ double c_nkaF(double &n, double &k) {
//     return 1.0;
// }
// 
// __device__ double a_nkaG(double &n, double &k) {
//     return 1.0;
// }
// 
// __device__ double b_nkaG(double &n, double &k) {
//     return 1.0;
// }
// 
// __device__ double c_nkaG(double &n, double &k) {
//     return 1.0;
// }

__device__ double F_2eff(double &k_i, double &k_j, double &mu_ij) {
    double n_i = n_spline_eval(k_i);
    double n_j = n_spline_eval(k_j);
    double a_i = a_nkaF(n_i, k_i);
    double a_j = a_nkaF(n_j, k_j);
    double b_i = b_nkaF(n_i, k_i);
    double b_j = b_nkaF(n_j, k_j);
    double c_i = c_nkaF(n_i, k_i);
    double c_j = c_nkaF(n_j, k_j);
    return FIVESEVENTHS*a_i*a_j + 0.5*mu_ij*((k_i/k_j) + (k_j/k_i))*b_i*b_j + TWOSEVENTHS*mu_ij*mu_ij*c_i*c_j;
}

__device__ double G_2eff(double &k_i, double &k_j, double &mu_ij) {
    double n_i = n_spline_eval(k_i);
    double n_j = n_spline_eval(k_j);
    double a_i = a_nkaG(n_i, k_i);
    double a_j = a_nkaG(n_j, k_j);
    double b_i = b_nkaG(n_i, k_i);
    double b_j = b_nkaG(n_j, k_j);
    double c_i = c_nkaG(n_i, k_i);
    double c_j = c_nkaG(n_j, k_j);
    return THREESEVENTHS*a_i*a_j + 0.5*mu_ij*((k_i/k_j) + (k_j/k_i))*b_i*b_j + FOURSEVENTHS*mu_ij*mu_ij*c_i*c_j;
}

__device__ double Z_2eff(double &k_i, double &k_j, double &k_ij, double &mu_i, double &mu_j, double &mu_ij, double &mu_ijp) {
    double b_s2 = -FOURSEVENTHS*(b1 - 1.0);
    double S_2 = mu_ij*mu_ij - ONETHIRD;
    double F_2 = F_2eff(k_i, k_j, mu_ij);
    double G_2 = G_2eff(k_i, k_j, mu_ij);
    return b1*(F_2 + 0.5*f*mu_ijp*k_ij*((mu_i/k_i) + (mu_j/k_j))) + f*mu_ijp*mu_ijp*G_2 + 0.5*f*f*mu_ijp*k_ij*mu_i*mu_j*((mu_j/k_i) + (mu_i/k_j)) + 0.5*b2 + 0.5*b_s2*S_2;
}

__device__ double get_Legendre(double &mu, int &l) {
    if (l == 0) {
        return 1.0;
    } else {
        return 0.5*(3.0*mu*mu - 1.0);
    }
}

__device__ double get_shape_correction(double4 &k, double &BkNW) {
    if ((int)k.w == 0) {
        return BkNW*(a0 + a1*k.x + a2*k.y + a3*k.z);
    } else {
        return BkNW*(c0 + c1*k.x + c2*k.y + c3*k.z);
    }
}

__device__ double get_grid_value(double &mu, double &phi, double4 &k, int l, double &BkNW) {
    float z = (k.x*k.x + k.y*k.y - k.z*k.z)/(2.0*k.x*k.y);
    double mu_1 = mu;
    double mu_2 = -mu_1*z + sqrt(1.0 - mu_1*mu_1)*sqrt(1.0 - z*z)*cos(phi);
    double mu_3 = -(mu_1*k.x + mu_2*k.y)/k.z;
    
    double P_L = get_Legendre(mu_1, l);
    
    double sq_ratio = (a_perp*a_perp)/(a_para*a_para) - 1.0;
    double mu1bar = 1.0 + mu_1*mu_1*sq_ratio;
    double mu2bar = 1.0 + mu_2*mu_2*sq_ratio;
    double mu3bar = 1.0 + mu_3*mu_3*sq_ratio;
    
    double k_1 = (k.x*sqrt(mu1bar))/a_perp;
    double k_2 = (k.y*sqrt(mu2bar))/a_perp;
    double k_3 = (k.z*sqrt(mu3bar))/a_perp;
    
    mu_1 = (mu_1*a_perp)/(a_para*sqrt(mu1bar));
    mu_2 = (mu_2*a_perp)/(a_para*sqrt(mu2bar));
    mu_3 = (mu_3*a_perp)/(a_para*sqrt(mu3bar));
    
    double P_1 = pk_spline_eval(k_1)/(a_perp*a_perp*a_para);
    double P_2 = pk_spline_eval(k_2)/(a_perp*a_perp*a_para);
    double P_3 = pk_spline_eval(k_3)/(a_perp*a_perp*a_para);
    
    double mu_12 = -(k_1*k_1 + k_2*k_2 - k_3*k_3)/(2.0*k_1*k_2);
    double mu_23 = -(k_2*k_2 + k_3*k_3 - k_1*k_1)/(2.0*k_2*k_3);
    double mu_31 = -(k_3*k_3 + k_1*k_1 - k_2*k_2)/(2.0*k_3*k_1);
    
    double k_12 = sqrt(k_1*k_1 + k_2*k_2 + 2.0*k_1*k_2*mu_12);
    double k_23 = sqrt(k_2*k_2 + k_3*k_3 + 2.0*k_2*k_3*mu_23);
    double k_31 = sqrt(k_3*k_3 + k_1*k_1 + 2.0*k_3*k_1*mu_31);
    
    double mu_12p = (k_1*mu_1 + k_2*mu_2)/k_12;
    double mu_23p = (k_2*mu_2 + k_3*mu_3)/k_23;
    double mu_31p = (k_3*mu_3 + k_1*mu_1)/k_31;
    
    double Z1k1 = Z_1(mu_1);
    double Z1k2 = Z_1(mu_2);
    double Z1k3 = Z_1(mu_3);
    
    double Z2k12 = Z_2eff(k_1, k_2, k_12, mu_1, mu_2, mu_12, mu_12p);
    double Z2k23 = Z_2eff(k_2, k_3, k_23, mu_2, mu_3, mu_23, mu_23p);
    double Z2k31 = Z_2eff(k_3, k_1, k_31, mu_3, mu_1, mu_31, mu_31p);
    
    double shape_cor = get_shape_correction(k, BkNW);
    
    return 2.0*(Z1k1*Z1k2*Z2k12*P_1*P_2 + Z1k2*Z1k3*Z2k23*P_2*P_3 + Z1k3*Z1k1*Z2k31*P_3*P_1)*FoG(k_1, k_2, k_3, mu_1, mu_2, mu_3)*P_L + shape_cor;
}


__global__ void calc_model_bispectrum(double4 *ks, double *Bk, double *BkNW) {
    int tid = threadIdx.y + blockDim.x*threadIdx.x; // Block local thread ID
    
    __shared__ double integration_grid[1024];
    
    // Calculate the value for this thread
    double phi = PI*d_x[threadIdx.y] + PI;
    integration_grid[tid] = d_w[threadIdx.x]*d_w[threadIdx.y]*get_grid_value(d_x[threadIdx.x], phi, ks[blockIdx.x], (int)ks[blockIdx.x].w, BkNW[blockIdx.x]);
    __syncthreads();
    
    // First step of reduction done by 32 threads
    if (threadIdx.y == 0) {
        for (int i = 1; i < 32; ++i)
            integration_grid[tid] += integration_grid[tid + i];
    }
    __syncthreads();
    
    // Final reduction and writing result to global memory done only by the first thread
    if (tid == 0) {
        for (int i = 1; i < 32; ++i)
            integration_grid[0] += integration_grid[blockDim.x*i];
        Bk[blockIdx.x] = (integration_grid[0]/4.0)*sqrt((2.0*ks[blockIdx.x].w + 1.0)/PI);
//         Bk[blockIdx.x] += get_shape_correction(ks[blockIdx.x], BkNW[blockIdx.x]);
    }
}

void model_calc(std::vector<double> &pars, double4 *d_ks, double *d_Bk, double *d_BkNW, 
                std::vector<double> &Bk) {
    // Move current parameters to device constant memory
    std::vector<double> theta(pars.size());
    for (int i = 0; i < pars.size(); ++i)
        theta[i] = pars[i];
    theta[0] *= theta[3];
    theta[1] *= theta[3];
    theta[2] *= theta[3];
    gpuErrchk(cudaMemcpyToSymbol(d_p, theta.data(), pars.size()*sizeof(double)));
    
    // Set up 2D thread grid to do the 2D integration
    dim3 num_threads(32,32);
    
    size_t num_data = Bk.size();
    
    // Call the CUDA kernel to launch num_data blocks (i.e. each block calculates one point of the model)
    calc_model_bispectrum<<<num_data, num_threads>>>(d_ks, d_Bk, d_BkNW);

    gpuErrchk(cudaMemcpy(Bk.data(), d_Bk, num_data*sizeof(double), cudaMemcpyDeviceToHost));
}

// __global__ void get_slope(double *d_k, double *d_nk, int N) {
//     int tid = threadIdx.x + blockIdx.x*blockDim.x;
//     
//     if (tid < N) d_nk[tid] = n_spline_eval((double)d_k[tid]);
// }
// 
// __global__ void get___powfer(double *d_k, double *d_Power, int N) {
//     int tid = threadIdx.x + blockIdx.x*blockDim.x;
//     
//     if (tid < N) d_Power[tid] = pk_spline_eval((double)d_k[tid]);
// }
// 
// __global__ void get_Q3(double *d_k, double *d_Q3check, int N) {
//     int tid = threadIdx.x + blockIdx.x*blockDim.x;
//     
//     if (tid < N) d_Q3check[tid] = Q3_spline_eval((double)d_k[tid]);
// }
    
#endif
