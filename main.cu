/* BIMODAL v1
 * David W. Pearson
 * September 28, 2017
 * 
 * This version of the code will implement some improvements to make the model better fit non-linear
 * features present in the data. The algorithm is effectively that of Gil-Marin 2012/2015.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <gsl/gsl_spline.h>
#include "include/gpuerrchk.h"
#include "include/mcmc.h"
#include "include/harppi.h"
#include "include/make_spline.h"
#include "include/dewiggle.h"

double get_k_nl(std::string P_lin_file);

int main(int argc, char *argv[]) {
    // Use HARPPI hidden in an object file to parse parameters
    parameters p(argv[1]);
    
    // Generate cubic splines of the input BAO and NW power spectra
    std::vector<double4> Pk_spline = make_spline(p.gets("input_power"));
    std::vector<double> k;
    std::vector<double> n;
    get_dewiggled_slope(p.gets("in_pk_lin_file"), k, n);
    std::vector<double4> nk_spline = make_uniform_spline(k, n, 877, p.getd("k_min"), p.getd("k_max"));
    
    std::vector<double> Q_3;
    Q_3.reserve(n.size());
    for (size_t i = 0; i < n.size(); ++i) {
        Q_3.push_back((4.0 - pow(2.0, n[i]))/(1.0 + pow(2.0, n[i] + 1.0)));
    }
    
    std::vector<double4> Q3_spline = make_uniform_spline(k, Q_3, 877, p.getd("k_min"), p.getd("k_max"));
    std::cout << "Pk_spline.size() = " << Pk_spline.size() << std::endl;
    std::cout << "nk_spline.size() = " << nk_spline.size() << std::endl;
    std::cout << "Q3_spline.size() = " << Q3_spline.size() << std::endl;    
    
    // Copy the splines to the allocated GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_Pk, Pk_spline.data(), 128*sizeof(double4)));
    gpuErrchk(cudaMemcpyToSymbol(d_n, nk_spline.data(), 877*sizeof(double4)));
    gpuErrchk(cudaMemcpyToSymbol(d_Q3, Q3_spline.data(), 877*sizeof(double4)));
    
    // Copy Gaussian Quadrature weights and evaluation point to GPU constant memory
    gpuErrchk(cudaMemcpyToSymbol(d_w, &w_i[0], 32*sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(d_x, &x_i[0], 32*sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(d_aF, a_F, 9*sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(d_aG, a_G, 9*sizeof(double)));
    
    double k_nl = get_k_nl(p.gets("in_pk_lin_file"));
    gpuErrchk(cudaMemcpyToSymbol(d_knl, &k_nl, sizeof(double)));
    
    // Declare a pointer for the integration workspace and allocate memory on the GPU
    double *d_Bk;
    double4 *d_ks;
    
    gpuErrchk(cudaMalloc((void **)&d_Bk, p.geti("num_data")*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_ks, p.geti("num_data")*sizeof(float4)));
    
    std::vector<double> start_params;
    std::vector<bool> limit_params;
    std::vector<double> var_i;
    std::vector<double> min;
    std::vector<double> max;
    for (int i = 0; i < p.geti("num_params"); ++i) {
        start_params.push_back(p.getd("start_params", i));
        limit_params.push_back(p.getb("limit_params", i));
        var_i.push_back(p.getd("vars", i));
        min.push_back(p.getd("min_params", i));
        max.push_back(p.getd("max_params", i));
    }
    
    // Initialize bkmcmc object
    bkmcmc bk_fit(p.gets("data_file"), p.gets("cov_file"), start_params, var_i, d_ks, d_Bk);
    
    // Check that the initialization worked
    bk_fit.check_init();
    
    // Set any limits on the parameters
    bk_fit.set_param_limits(limit_params, min, max);
    
    // Run the MCMC chain
    bk_fit.run_chain(p.geti("num_draws"), p.geti("num_burn"), p.gets("reals_file"), d_ks, d_Bk,
                     p.getb("new_chain"));
    
    // Free device pointers
    gpuErrchk(cudaFree(d_Bk));
    gpuErrchk(cudaFree(d_ks));
    
    return 0;
}

double get_k_nl(std::string P_lin_file) {
    double k_nl;
    std::vector<double> prod;
    std::vector<double> k;
    std::ifstream fin(P_lin_file);
    while (!fin.eof()) {
        double kt, P;
        fin >> kt >> P;
        if (!fin.eof()) {
            k.push_back(kt);
            prod.push_back(kt*kt*kt*P/(2.0*PI*PI));
        }
    }
    fin.close();
    
    gsl_spline *Getknl = gsl_spline_alloc(gsl_interp_cspline, k.size());
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(Getknl, prod.data(), k.data(), k.size());
    
    k_nl = gsl_spline_eval(Getknl, 1.0, acc);
    std::cout << "k_nl = " << k_nl << std::endl;
    
    gsl_spline_free(Getknl);
    gsl_interp_accel_free(acc);
    
    return k_nl;
}
