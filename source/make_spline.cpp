#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <gsl/gsl_spline.h>
#include "../include/cspline.h"
#include <vector_types.h>

std::vector<double4> make_spline(std::string in_pk_file) {
    std::ifstream fin;
    
    std::vector<double> kin;
    std::vector<double> pin;
    
    fin.open(in_pk_file.c_str(), std::ios::in);
    while (!fin.eof()) {
        double kt, pt;
        fin >> kt >> pt;
        if (!fin.eof()) {
            kin.push_back(kt);
            pin.push_back(pt);
        }
    }
    fin.close();
    
    cspline<double> Pk_spline(kin, pin);
    
    std::cout << "Setting up the spline..." << std::endl;
    std::vector<double4> Pk;
    Pk_spline.set_pointer_for_device(Pk);

    return Pk;
}

std::vector<double4> make_uniform_spline(std::string in_pk_file, int num_points, double k_min, double k_max) {
    std::vector<double> kin;
    std::vector<double> pin;
    
    std::ifstream fin(in_pk_file);
    while (!fin.eof()) {
        double kt, pt;
        fin >> kt >> pt;
        if (!fin.eof()) {
            kin.push_back(kt);
            pin.push_back(pt);
        }
    }
    fin.close();
    
    double Delta_k = (k_max - k_min)/(num_points - 1.0);
    
    gsl_spline *P_nu = gsl_spline_alloc(gsl_interp_cspline, pin.size());
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(P_nu, kin.data(), pin.data(), pin.size());
    
    std::vector<double> k;
    std::vector<double> P;
    for (size_t i = 0; i < num_points; ++i) {
        double k_i = k_min + i*Delta_k;
        k.push_back(k_i);
        P.push_back(gsl_spline_eval(P_nu, k_i, acc));
    }
    
    cspline<double> P_u(k, P);
    std::vector<double4> P_spline;
    P_u.set_pointer_for_device(P_spline);
    
    gsl_spline_free(P_nu);
    gsl_interp_accel_free(acc);
    
    return P_spline;
}

std::vector<double4> make_uniform_spline(std::vector<double> &x, std::vector<double> &y, int num_points,
                                        double x_min, double x_max) {
    gsl_spline *y_nu = gsl_spline_alloc(gsl_interp_cspline, y.size());
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(y_nu, x.data(), y.data(), y.size());
    
    double Delta_x = (x_max - x_min)/(num_points - 1.0);
    
    std::vector<double> x_u;
    std::vector<double> y_u;
    for (size_t i = 0; i < num_points; ++i) {
        double x_i = x_min + i*Delta_x;
        x_u.push_back(x_i);
        y_u.push_back(gsl_spline_eval(y_nu, x_i, acc));
    }
    
    cspline<double> y_spline(x_u, y_u);
    std::vector<double4> spline;
    y_spline.set_pointer_for_device(spline);
    
    gsl_spline_free(y_nu);
    gsl_interp_accel_free(acc);
    
    return spline;
}
