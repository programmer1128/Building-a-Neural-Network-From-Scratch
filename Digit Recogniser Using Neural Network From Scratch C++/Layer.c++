#include<iostream>
#include<vector>
#include"matrices.hpp"
#include"sigma_function.hpp"
#include<cmath>
#include<random>
#include"Layer.hpp"



     Layer:: Layer(int input_size, int output_size)
     {
         //the weight matrix has rows= number of output neurons because
         //from one node of current layer there will be m links to the 
         //next layer where m is the number of neurons in next layer
         this->input_size = input_size;
         //weight matrix has columns = number of input neurons 
         //each column in the weight matrix contains links from one 
         //node of current layer to other nodes of next layer so there 
         //will n columns where n is number of neurons in current layer
         this->output_size=output_size;
         
         //initialsing the weights of this layer 
         weights.resize(output_size,std::vector<double> (input_size));

         //using values from probability distribution of mean =0
         //and standard deviation = 1/sqrt(input_size)
         //this provides better results
         std::default_random_engine generator;
         std::normal_distribution<double> distribution(0.0, 1.0 / std::sqrt(input_size));


         for(int i=0;i<output_size;i++)
         {
             for(int k=0;k<input_size;k++)
             {
                 weights[i][k]=distribution(generator);
             }
         }

         //output matrix will be a row matrix of columns size = output_size
         //as it will be the input to the new layer

     }

             

     //this function calculate the input for the next layer 
     std::vector<std::vector<double>> Layer::forward(const std::vector<std::vector<double>> &input)
     {
         std::cout<<"forward method inside Layer is called"<<std::endl;
         std::vector<std::vector<double>> result;
         result= multiply_matrix(weights,input);
                 
         result = sigma_of_matrix(result);
         output=result;
         inputs=input;
         return result;
     }

     //now we have to implement the function for computing the rate of error function
     void Layer::compute_error(const std::vector<std::vector<double>>& target)
     {
         //compute the sigma derivative of the output matrix 
         std::vector<std::vector<double>> result = sigma_derivative_of_matrix(output);
         int row = result.size(); int col = result[0].size();
         for(int i=0;i<row;i++)
         {
             for(int k=0;k<col;k++)
             {
                 result[i][k]*=output[i][k]-target[i][k];
             }
         }

         delta = result;

         std::vector<std::vector<double>> input_transpose= transpose_matrix(inputs);
         std::vector<std::vector<double>> error_gradient = multiply_matrix(result,input_transpose);
         update_weights(0.01,error_gradient);
     }

     //after the rate is found we now implement the updating weights function using the value
     //of the rate Wnew = Wold - (learning_rate*rate of error)
     //we use the minus sign because we have to minimise the error

     void Layer::update_weights(double learning_rate,const std::vector<std::vector<double>> &error)
     {
         int row = weights.size(); int col = weights[0].size();
         for(int i=0;i<row;i++)
         {
             for(int k=0;k<col;k++)
             {
                 weights[i][k]-= learning_rate*(error[i][k]);
             }
         }
     }

     //compute the errors for the hidden layers
     void Layer::compute_error_hidden_layers(const std::vector<std::vector<double>> &next_weights, std::vector<std::vector<double>> &next_delta)
     {
         //we have the weights of the next layer and the delta of the next layer
         std::vector<std::vector<double>> next_layer_weight_transpose= transpose_matrix(next_weights);

         std::vector<std::vector<double>> propagated_error = multiply_matrix(next_layer_weight_transpose,next_delta);

         std::vector<std::vector<double>> sigma_deriv= sigma_derivative_of_matrix(output);

         int row = sigma_deriv.size(); int col = sigma_deriv[0].size();

         for(int i=0;i<row;i++)
         {
             for(int k=0;k<col;k++)
             {
                 propagated_error[i][k]*=sigma_deriv[i][k];
             }
         }

         delta = propagated_error;

         std::vector<std::vector<double>> input_transpose= transpose_matrix(inputs);
         std::vector<std::vector<double>> error_gradient = multiply_matrix(delta,input_transpose);

         //update the weight matrix 
         update_weights(0.01,error_gradient);
     }

     std::vector<std::vector<double>> Layer::get_weights()
     {
         return weights;
     }

     std::vector<std::vector<double>> Layer::get_delta()
     {
         return  delta;
     }

