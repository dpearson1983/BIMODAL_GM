#ifndef _DEWIGGLE_H_
#define _DEWIGGLE_H_

#include <vector>
#include <string>

std::vector<double> derivative(const std::vector<double> &x, std::vector<double> &y);

std::vector<double> log_derivative(const std::vector<double> &x, std::vector<double> &y);

std::vector<double> dewiggle(const std::vector<double> &k, std::vector<double> &n);

void get_dewiggled_slope(std::string file, std::vector<double> &x, std::vector<double> &y);

#endif
