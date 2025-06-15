#include<iostream>
#include"Layer.hpp"
#include"Network.hpp"



     //constructor of class

   Network:: Network(std::vector<std::pair<int,int>> sizes)
             {
                 int len = sizes.size();
                 std::cout<<"Network created"<<std::endl;
                 for(int i=0;i<len;i++)
                 {
                     //creating a single layer
                     Layer layer(sizes[i].first,sizes[i].second);

                     //linking the layers together by putting them in a vector
                     layers.push_back(layer);
                 }
             }


             //Declaring the function that takes the output of one layer and puts
             //it into the input of the next layer till we reach the final layer
             std::vector<std::vector<double>> Network::forward_propagation(const std::vector<std::vector<double>> &input)
             {
                 std::cout<<"entering forward_propagation"<<std::endl;
                 std::vector<std::vector<double>> result = input;
                 
                 int len = layers.size();
                 //create a object of Layer class to access the function
                
                 //now performing the forward propagation of input from input 
                 //layer to output layer
                 for(auto& layer :layers)
                 {
                     //this will store the output of the current layer. Then 
                     //It will send this output as the input to the next layer 
                     //and  this will continue till the last layer finally getiing our
                     //entire output from the nueral network
                     std::cout<<"forward passing through layer "<<std::endl;
                     result = layer.forward(result); 
                 }

                 return result;
             }


             //now we add the method to train the network using the train function

             void Network::train_network(const std::vector<std::vector<double>> &X,
                    const std::vector<std::vector<double>> &Y,
                    int epochs,
                    double learning_rate)
             { 
                 std::cout<<"training the network"<<std::endl;
                 int sample_size = X.size();

                 for(int i=0;i<epochs;i++)
                 {
                     for(int k=0;k<sample_size;k++)
                     {
                         //input matrix
                         std::vector<std::vector<double>> input= {X[k]};
                         std::vector<std::vector<double>> target = {Y[k]};

                         std::vector<std::vector<double>> output = forward_propagation(input);

                         //computing the loss for output layer to the hidden layer behind
                         std::cout<<"Getting the loss of the output and last hidden layer"<<std::endl;
                         layers.back().compute_error(target);

                         //now backpropagation through the hidden layers
                         for(int l=layers.size()-2;l>=0;l--)
                         {
                             std::vector<std::vector<double>> next_weights = layers[l + 1].get_weights();
                             std::vector<std::vector<double>> next_delta = layers[l + 1].get_delta();

                             layers[l].compute_error_hidden_layers(next_weights, next_delta);
                         }
                     }  

                     std::cout<<"epoch stage "<<i+1<< "completed"<<std::endl; 
                 }

             }//end of training method
