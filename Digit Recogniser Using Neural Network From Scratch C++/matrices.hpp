#ifndef matrices_hpp
#define matrices_hpp

#include<vector>


//function for multiplying the matrices
std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y);

//function to add two matrices
std::vector<std::vector<double>> add_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y);

//function to subtracy two matrices
std::vector<std::vector<double>> subtract_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y);

//function to find the transpose of a matrix
std::vector<std::vector<double>> transpose_matrix(const std::vector<std::vector<double>>& x);



#endif