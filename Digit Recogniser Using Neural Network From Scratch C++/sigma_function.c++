#include<iostream>
#include <cmath>
#include<vector>

double sigma(const double &x)
{
     double d= 1+(std::exp(x));
     double result = (double)(std::exp(x)/(d));
     return result;
}

double sigma_derivative(const double & x)
{
     double d = 1+(std::exp(x));
     double sigma = (double)(std::exp(x)/(d));
     double derivative_value = (double)(1-sigma);
     
     return (derivative_value)*(sigma);
}

std::vector<std::vector<double>> sigma_of_matrix(const std::vector<std::vector<double>>& x)
{
     int x_row=x.size(); int x_col=x[0].size();
     std::vector<std::vector<double>> result (x_row,std::vector<double> (x_col));
     for(int i=0;i<x_row;i++)
     {
         for(int k=0;k<x_col;k++)
         {
             result[i][k]= sigma(x[i][k]);
         }
     }
     return result;
}


std::vector<std::vector<double>> sigma_derivative_of_matrix(const std::vector<std::vector<double>>& x)
{
     int x_row=x.size(); int x_col=x[0].size();
     std::vector<std::vector<double>> result (x_row,std::vector<double> (x_col));
     for(int i=0;i<x_row;i++)
     {
         for(int k=0;k<x_col;k++)
         {
             result[i][k]= sigma_derivative(x[i][k]);
         }
     }
     return result;
}


/*
int main()
{
     std::cout<<sigma_derivative(1);
     return 0;
}
*/
