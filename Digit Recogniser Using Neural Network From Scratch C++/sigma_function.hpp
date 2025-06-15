#ifndef sigma_function_hpp
#define sigma_function_hpp
#include<vector>

//sigma function
double sigma(const double &x);

//sigma derivative function
double sigma_derivative(const double & x);

//sigma function applied to matrices
std::vector<std::vector<double>> sigma_of_matrix(const std::vector<std::vector<double>>& x);

//sigma deriavtive function aplied to matrices
std::vector<std::vector<double>> sigma_derivative_of_matrix(const std::vector<std::vector<double>>& x);


#endif
