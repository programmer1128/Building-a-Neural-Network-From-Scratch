#ifndef functions_hpp
#define functions_hpp
#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>


double relu(double x);

double relu_derivative(double x);

std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& z) ;

std::vector<std::vector<double>> relu_derivative_of_matrix(const std::vector<std::vector<double>> &matrix);

#endif