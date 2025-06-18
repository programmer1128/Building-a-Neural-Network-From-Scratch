#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<cmath>
double relu(double x)
{
     return std::max(0.0,x);
}

double relu_derivative(double x)
{
     return  x > 0 ? 1.0 : 0.0;
}


std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& z) 
{
    std::vector<std::vector<double>> output = z;

    // Find max element for numerical stability
    double max_val = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < z.size(); i++)
    {
        for (int k = 0; k < z[0].size(); k++)
        {
            if (z[i][k] > max_val)
                max_val = z[i][k];
        }
    }

    // Compute sum of exp(x - max_val)
    double sum = 0.0;
    for (int i = 0; i < z.size(); i++)
    {
        for (int k = 0; k < z[0].size(); k++)
        {
            output[i][k] = std::exp(z[i][k] - max_val);
            sum += output[i][k];
        }
    }

    // Normalize
    for (int i = 0; i < z.size(); i++)
    {
        for (int k = 0; k < z[0].size(); k++)
        {
            output[i][k] /= sum;
        }
    }

    return output;
}

std::vector<std::vector<double>> relu_derivative_of_matrix(const std::vector<std::vector<double>> &matrix)
{
     std::vector<std::vector<double>> result = matrix;

     for (int i = 0; i < matrix.size(); i++) 
     {
         for (int j = 0; j < matrix[0].size(); j++) 
         {
             result[i][j] = (matrix[i][j] > 0.0) ? 1.0 : 0.0;
         }
     }

    return result;
}

//implementing the cross entropy function 
