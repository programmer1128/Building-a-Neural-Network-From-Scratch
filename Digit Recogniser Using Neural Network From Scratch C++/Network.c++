#include<iostream>
#include"Layer.hpp"
#include"Network.hpp"
#include<fstream>

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
         //std::cout<<"entering forward_propagation"<<std::endl;

         //std::cout<<"Input matrix row "<<input.size()<<" input matrix column "<<input[0].size()<<std::endl;
         std::vector<std::vector<double>> result = input;
                 
         int len = layers.size();
         //create a object of Layer class to access the function
                
         //now performing the forward propagation of input from input 
         //layer to output layer

         for(int i=0;i<layers.size();i++)
         {
             //this will store the output of the current layer. Then 
             //It will send this output as the input to the next layer 
             //and  this will continue till the last layer finally getiing our
             //entire output from the nueral network
             bool is_output_layer = (i==layers.size()-1);
             result = layers[i].forward(result,is_output_layer); 
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
         std::cout<<"sample size is "<<sample_size<<std::endl;
                
         for (int i = 0; i < epochs; i++) 
         {
             double epoch_loss = 0;
             int correct = 0;
                
             for (int k = 0; k < sample_size; k++) 
             {
                 // input and target setup
                 std::vector<std::vector<double>> input(784, std::vector<double>(1));
                 for (int j = 0; j < 784; ++j)
                 {
                     input[j][0] = X[k][j];
                 }
                     
                
                 std::vector<std::vector<double>> target(10, std::vector<double>(1));
                 for (int j = 0; j < 10; ++j)
                 {
                     target[j][0] = Y[k][j];
                 }
                           
                
                 std::vector<std::vector<double>> output = forward_propagation(input);
                
                 // ===== Accuracy Count =====
                 int predicted = -1, actual = -1;
                 double max_pred = -1.0;
                 for (int j = 0; j < 10; j++) 
                 {
                     if (output[j][0] > max_pred) 
                     {
                         max_pred = output[j][0];
                         predicted = j;
                     }
                     if (target[j][0] == 1.0)
                     {
                         actual = j;
                     }
                                 
                 }
                 if (predicted == actual) correct++;
                
                 // ===== Loss Calculation =====
                 double sample_loss = 0.0;
                 for (int j = 0; j < 10; ++j) 
                 {
                     // Avoid log(0) by adding a small epsilon
                     double p = std::max(output[j][0], 1e-15);
                     sample_loss += -target[j][0] * std::log(p);
                 }
                 epoch_loss+=sample_loss;
                 // ===== Backpropagation =====
                 layers.back().compute_error(target);
                 for (int l = layers.size() - 2; l >= 0; l--) 
                 {
                     std::vector<std::vector<double>> next_weights = layers[l + 1].get_weights();
                     std::vector<std::vector<double>> next_delta = layers[l + 1].get_delta();
                     layers[l].compute_error_hidden_layers(next_weights, next_delta);
                 }

             }
                
             double average_epoch_loss = epoch_loss / sample_size;
             double accuracy = (double)correct / sample_size * 100.0;
                 
             if (std::isnan(average_epoch_loss)) 
             {
                 std::cerr << "Loss is NaN at epoch " << i << std::endl;
                 break;
             }
                
             std::cout << "Epoch " << i + 1 << " completed. Loss = " 
                         << average_epoch_loss << ", Accuracy = " << accuracy << "%" << std::endl;
         }



         std::cout << "Evaluating predictions on test data..." << std::endl;
         int correct = 0;
         int total = X.size();

         for (int i = 0; i < total; ++i) 
         {
             // Prepare input vector
             std::vector<std::vector<double>> input(784, std::vector<double>(1));
             for (int j = 0; j < 784; ++j) 
             {
                 input[j][0] = X[i][j];
             }

             // Run forward propagation
             std::vector<std::vector<double>> output = forward_propagation(input);

             // Get predicted label
             int predicted = 0;
             double max_pred = output[0][0];
             for (int j = 1; j < 10; ++j) 
             {
                 if (output[j][0] > max_pred) 
                 {
                     max_pred = output[j][0];
                     predicted = j;
                 }
             }

                     // Get actual label (from one-hot encoded Y_test)
             int actual = 0;
             for (int j = 0; j < 10; ++j) 
             {
                 if (Y[i][j] == 1.0) 
                 {
                     actual = j;
                     break;
                 }
             }

             if (predicted == actual) correct++;

             // Print comparison
             if(i<100)
             {
                         std::cout << "Image " << i << " | Predicted: " << predicted << " | Actual: " << actual << std::endl;
             }
    
         }

                 double accuracy = (double)correct / total * 100.0;
                 std::cout << "Final Test Accuracy: " << accuracy << "%" << std::endl;
                
                
     }//end of training method



     void Network::save_model(const std::string& filename) const 
     {
         std::ofstream out(filename, std::ios::binary);
         if (!out.is_open()) 
         {
             throw std::runtime_error("Could not open file to save model.");
         }

         // Write header for validation
         out.write("MLP", 3);  // File header

         // Write number of layers
         int num_layers = layers.size();
         out.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));

         // Serialize each layer
         for (const auto& layer : layers) 
         {
             layer.save(out);
         }

         out.close();
     }
            
     void Network::load_model(const std::string& filename) 
     {
         std::ifstream in(filename, std::ios::binary);
         if (!in.is_open()) 
         {
             throw std::runtime_error("Could not open file to load model.");
         }

         // Validate header
         char header[3];
         in.read(header, 3);
         if (std::string(header, 3) != "MLP") 
         {
             throw std::runtime_error("Invalid model file format.");
         }

         // Load number of layers
         int num_layers;
         in.read(reinterpret_cast<char*>(&num_layers), sizeof(int));

         layers.clear();
         layers.resize(num_layers);

         // Deserialize each layer
         for (auto& layer : layers) 
         {
             layer.load(in);
         }

         in.close();
     }
            
