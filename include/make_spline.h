#ifndef _MAKE_SPLINE_H_
#define _MAKE_SPLINE_H_

#include <vector>
#include <string>
#include <vector_types.h>

std::vector<double4> make_spline(std::string in_pk_file);

std::vector<double4> make_uniform_spline(std::string in_pk_file, int num_points, double k_min, double k_max);

std::vector<double4> make_uniform_spline(std::vector<double> &x, std::vector<double> &y, int num_points, 
                                        double k_min, double k_max);

#endif
