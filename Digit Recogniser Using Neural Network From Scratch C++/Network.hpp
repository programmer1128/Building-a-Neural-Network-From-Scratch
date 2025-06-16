#ifndef Network_hpp
#define Network_hpp

#include<iostream>
#include"Layer.hpp"

class Network
{
     std::vector<Layer> layers;

     public :
             Network(std::vector<std::pair<int,int>> sizes);

             std::vector<std::vector<double>> forward_propagation(const std::vector<std::vector<double>> &input);

             void train_network(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y,
                    int epochs,
                    double learning_rate);


            void save_model(const std::string& filename) const;
            void load_model(const std::string& filename);
                    
            
             
};


#endif
