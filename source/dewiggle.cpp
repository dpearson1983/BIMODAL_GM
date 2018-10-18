#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <gsl/gsl_spline.h>
#include "../include/dewiggle.h"

std::vector<double> derivative(const std::vector<double> &x, std::vector<double> &y) {
    std::vector<double> deriv;
    
    for (size_t i = 0; i < x.size(); ++i) {
        double result;
        if (i != 0 && i != x.size() - 1) {
            result = (y[i + 1] - y[i - 1])/(x[i + 1] - x[i - 1]);
        } else if (i == 0) {
            result = (y[i + 1] - y[i])/(x[i + 1] - x[i]);
        } else if (i == x.size() - 1) {
            result = (y[i] - y[i - 1])/(x[i] - x[i - 1]);
        }
        deriv.push_back(result);
    }
    
    return deriv;
}

std::vector<double> log_derivative(const std::vector<double> &x, std::vector<double> &y) {
    std::vector<double> deriv;
    
    for (size_t i = 0; i < x.size(); ++i) {
        double result;
        if (i != 0 && i != x.size() - 1) {
            result = (y[i + 1] - y[i - 1])/(x[i + 1] - x[i - 1]);
        } else if (i == 0) {
            result = (y[i + 1] - y[i])/(x[i + 1] - x[i]);
        } else if (i == x.size() - 1) {
            result = (y[i] - y[i - 1])/(x[i] - x[i - 1]);
        }
        result *= x[i]/y[i];
        deriv.push_back(result);
    }
    
    return deriv;
}

std::vector<double> dewiggle(const std::vector<double> &k, std::vector<double> &n) {
    std::vector<double> n_nw;
    n_nw.reserve(n.size());
    for (size_t i = 0; i < n.size(); ++i)
        n_nw.push_back(n[i]);
    
    for (int iter = 0; iter < 2; ++iter)  {
        std::vector<double> dn = derivative(k, n_nw);
        std::vector<double> d2n = derivative(k, dn);
        gsl_spline *nk = gsl_spline_alloc(gsl_interp_cspline, n_nw.size());
        gsl_interp_accel *acc = gsl_interp_accel_alloc();
        
        gsl_spline_init(nk, k.data(), n_nw.data(), n_nw.size());
        
        std::vector<double> ks;
        std::vector<double> vals;
        for (size_t i = 0; i < k.size(); ++i) {
            if (k[i] < 0.01 || k[i] > 0.8) {
                ks.push_back(k[i]);
                vals.push_back(n[i]);
            } else {
                if (d2n[i + 1]/d2n[i] < 0) {
                    double kt = (k[i + 1] + k[i])/2.0;
                    double valt = gsl_spline_eval(nk, kt, acc);
                    ks.push_back(kt);
                    vals.push_back(valt);
                }
            }
        }
        
        gsl_spline *nk_nw = gsl_spline_alloc(gsl_interp_cspline, vals.size());
        gsl_interp_accel *acc_nw = gsl_interp_accel_alloc();
        
        gsl_spline_init(nk_nw, ks.data(), vals.data(), vals.size());
        
        for (size_t i = 0; i < k.size(); ++i) {
            n_nw[i] = gsl_spline_eval(nk_nw, k[i], acc_nw);
        }
        
        gsl_spline_free(nk);
        gsl_spline_free(nk_nw);
        gsl_interp_accel_free(acc);
        gsl_interp_accel_free(acc_nw);
    }
    
    return n_nw;
}

void get_dewiggled_slope(std::string file, std::vector<double> &x, std::vector<double> &y) {
    std::vector<double> pin;
    
    std::ifstream fin(file);
    while(!fin.eof()) {
        double k, p;
        fin >> k >> p;
        if (!fin.eof()) {
            x.push_back(k);
            pin.push_back(p);
        }
    }
    fin.close();
    
    std::vector<double> dP = log_derivative(x, pin);
    
    y = dewiggle(x, dP);
}
