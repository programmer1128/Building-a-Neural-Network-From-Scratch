#ifndef Layer_hpp
#define Layer_hpp

#include<iostream>
#include<vector>
#include"matrices.hpp"
#include"sigma_function.hpp"
#include<cmath>
#include<random>
#include<fstream>

class Layer
{
     int input_size;
     int output_size;

     
             
     //store the input from the previous layer
     std::vector<std::vector<double>> inputs;

     //store the weights of links of the current layer to next layer
     std::vector<std::vector<double>> weights;

     //biases
     std::vector<std::vector<double>> biases;

     //store the output of this layer which will serve as input 
     //for next layer
     std::vector<std::vector<double>> output;

     std::vector<std::vector<double>> delta;
     
     
     public:
                
             Layer();
             Layer(int input_size, int output_size);

             std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input,bool is_ouput_layer);
 
             void compute_error(const std::vector<std::vector<double>>& target);

             void update_weights(double learning_rate,const std::vector<std::vector<double>> &error);

             void update_biases(double learning_rate,const std::vector<std::vector<double>> &error);

             void compute_error_hidden_layers(const std::vector<std::vector<double>> &next_weights, std::vector<std::vector<double>> &next_delta);

             std::vector<std::vector<double>> get_weights();

             std::vector<std::vector<double>> get_delta();

             std::vector<std::vector<double>> get_biases();

             void save(std::ofstream& out) const;
             
             void load(std::ifstream& in);


};

#endif
