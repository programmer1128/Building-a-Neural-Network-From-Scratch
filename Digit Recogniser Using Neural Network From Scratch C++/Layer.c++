#include<iostream>
#include<vector>
#include"matrices.hpp"
#include"sigma_function.hpp"
#include<cmath>
#include<random>
#include"Layer.hpp"
#include<fstream>
#include"functions.hpp"


     Layer::Layer()
     : input_size(0), output_size(0)
     {
         // Empty constructor used for loading later
     }

     Layer::Layer(int input_size, int output_size)
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
         std::normal_distribution<double> distribution(0.0, std::sqrt(2.0/input_size));


         for(int i=0;i<output_size;i++)
         {
             for(int k=0;k<input_size;k++)
             {
                 weights[i][k]=distribution(generator);
             }
         }

         biases.resize(output_size, std::vector<double>(1));

         for(int i=0;i<output_size;i++)
         {
             for(int k=0;k<1;k++)
             {
                 biases[i][k]=0.01;
             }
         }

         //output matrix will be a row matrix of columns size = output_size
         //as it will be the input to the new layer

     }

             

     //this function calculate the input for the next layer 
     std::vector<std::vector<double>> Layer::forward(const std::vector<std::vector<double>> &input,bool is_ouput_layer)
     {
         //std::cout<<"forward method inside Layer is called"<<std::endl;
         std::vector<std::vector<double>> result(weights.size(),std::vector<double> (input[0].size()));
         //std::cout<<"weight matrix row "<<weights.size()<<" weight maatrixx column "<<weights[0].size()<<std::endl;
         result= multiply_matrix(weights,input);

         output.resize(result.size(),std::vector<double>(result[0].size()));       
         
         inputs.resize(input.size(),std::vector<double> (input[0].size()));

         //adding the biases to the result before sigmoid function
         for(int i=0;i<result.size();i++)
         {
             for(int k=0;k<result[0].size();k++)
             {
                 result[i][k]+=(double)biases[i][k];
             }
         }
         //result = sigma_of_matrix(result);


         if(is_ouput_layer)
         {
             result = softmax(result);
         }
         else
         {
             for(int i=0;i<result.size();i++)
             {
                 for(int k=0;k<result[0].size();k++)
                 {
                     result[i][k]= relu(result[i][k]);
                 }
             }
         }
         

         output=result;
         
         inputs=input;
         return result;
     }

     //now we have to implement the function for computing the rate of error function
     void Layer::compute_error(const std::vector<std::vector<double>>& target)
     {
         //compute the sigma derivative of the output matrix 
         
         //std::vector<std::vector<double>> result = softmax(output);
         
         std::vector<std::vector<double>> result = output;

         
         int row = result.size(); int col = result[0].size();

         delta = result;
         
         for(int i=0;i<row;i++)
         {
             for(int k=0;k<col;k++)
             {
                 
                 delta[i][k]-=target[i][k];
             }
         }
         

         

         std::vector<std::vector<double>> bias_gradient(delta.size(), std::vector<double>(1, 0.0));

         // Accumulate gradient across batch dimension (columns)
         for (int i = 0; i < delta.size(); ++i) 
         {
             for (int k = 0; k < delta[0].size(); ++k) 
             {
                 bias_gradient[i][0] += delta[i][k];
             }
         }

         

         std::vector<std::vector<double>> input_transpose= transpose_matrix(inputs);
         
         std::vector<std::vector<double>> error_gradient = multiply_matrix(delta,input_transpose);
         
         update_weights(0.0001,error_gradient);
         update_biases(0.0001,bias_gradient);
         
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
                 if (std::isnan(weights[i][k]) || std::isinf(weights[i][k])) 
                 {
                     std::cerr << "NaN or INF detected in weights at [" << i << "][" << k << "]\n";
                 }
             }
         }
     }

     void Layer::update_biases(double learning_rate,const std::vector<std::vector<double>> &error)
     {

         for(int i=0;i<biases.size();i++)
         {
             for(int k=0;k<biases[0].size();k++)
             {
                 
                 biases[i][k]-= learning_rate*error[i][k];
                 if (std::isnan(biases[i][k]) || std::isinf(biases[i][k])) 
                 {
                     std::cerr << "NaN or INF detected in weights at [" << i << "][" << k << "]\n";
                 }
             }
         }
     }
     //compute the errors for the hidden layers
     void Layer::compute_error_hidden_layers(const std::vector<std::vector<double>> &next_weights, std::vector<std::vector<double>> &next_delta)
     {
         //we have the weights of the next layer and the delta of the next layer
         std::vector<std::vector<double>> next_layer_weight_transpose= transpose_matrix(next_weights);

         std::vector<std::vector<double>> propagated_error = multiply_matrix(next_layer_weight_transpose,next_delta);

         std::vector<std::vector<double>> relu_deriv= relu_derivative_of_matrix(output);

         int row = relu_deriv.size(); int col = relu_deriv[0].size();

         for(int i=0;i<row;i++)
         {
             for(int k=0;k<col;k++)
             {
                 propagated_error[i][k]*=relu_deriv[i][k];
             }
         }

         delta = propagated_error;

         std::vector<std::vector<double>> bias_gradient(delta.size(), std::vector<double>(1, 0.0));

         for(int i=0;i<delta.size();i++)
         {
             for(int k=0;k<delta[0].size();k++)
             {
                 bias_gradient[i][k]+=delta[i][k];
             }
         }
         std::vector<std::vector<double>> input_transpose= transpose_matrix(inputs);
         std::vector<std::vector<double>> error_gradient = multiply_matrix(delta,input_transpose);

         //update the weight matrix 
         update_weights(0.0001,error_gradient);
         update_biases(0.0001,bias_gradient);
     }

     std::vector<std::vector<double>> Layer::get_weights()
     {
         return weights;
     }

     std::vector<std::vector<double>> Layer::get_delta()
     {
         return  delta;
     }

     std::vector<std::vector<double>> Layer::get_biases()
     {
         return biases;
     }
     
     //now after our model is trained we need to store the trained model
     //so that we do not have to train the neural network for every input
     void Layer::save(std::ofstream& out) const 
     {
         out.write(reinterpret_cast<const char*>(&input_size), sizeof(int));
         out.write(reinterpret_cast<const char*>(&output_size), sizeof(int));

         // Save weights
         int weight_rows = weights.size();
         int weight_cols = weights[0].size();
         out.write(reinterpret_cast<const char*>(&weight_rows), sizeof(int));
         out.write(reinterpret_cast<const char*>(&weight_cols), sizeof(int));
         for (const auto& row : weights) 
         {
             out.write(reinterpret_cast<const char*>(row.data()), sizeof(double) * weight_cols);
         }

         // Save biases
         int bias_rows = biases.size();
         int bias_cols = biases[0].size();
         out.write(reinterpret_cast<const char*>(&bias_rows), sizeof(int));
         out.write(reinterpret_cast<const char*>(&bias_cols), sizeof(int));
         for (const auto& row : biases) 
         {
             out.write(reinterpret_cast<const char*>(row.data()), sizeof(double) * bias_cols);
         }
            
     }
    
     void Layer::load(std::ifstream& in) 
     {
         in.read(reinterpret_cast<char*>(&input_size), sizeof(int));
         in.read(reinterpret_cast<char*>(&output_size), sizeof(int));

         int weight_rows, weight_cols;
         in.read(reinterpret_cast<char*>(&weight_rows), sizeof(int));
         in.read(reinterpret_cast<char*>(&weight_cols), sizeof(int));
         weights.resize(weight_rows, std::vector<double>(weight_cols));
         for (auto& row : weights) 
         {
             in.read(reinterpret_cast<char*>(row.data()), sizeof(double) * weight_cols);
         }

         int bias_rows, bias_cols;
         in.read(reinterpret_cast<char*>(&bias_rows), sizeof(int));
         in.read(reinterpret_cast<char*>(&bias_cols), sizeof(int));
         biases.resize(bias_rows, std::vector<double>(bias_cols));
         for (auto& row : biases) 
         {
             in.read(reinterpret_cast<char*>(row.data()), sizeof(double) * bias_cols);
         }
            
    
     }
    
