#include<iostream>
#include"matrices.hpp"
#include<vector>
#include "sigma_function.hpp"
#include"Layer.hpp"
#include"Network.hpp"
#include <fstream>
#include<opencv2/opencv.hpp>
#include<filesystem>

uint32_t bswap32(uint32_t x) 
{
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}

std::vector<std::vector<double>> load_mnist_images(const std::string& file_path) 
{
     std::ifstream file(file_path, std::ios::binary);
     if (!file) 
     {
        throw std::runtime_error("Unable to open file " + file_path);
     }

     int magic_number = 0;
     int number_of_images = 0;
     int rows = 0;
     int cols = 0;

     file.read((char*)&magic_number, 4);
     file.read((char*)&number_of_images, 4);
     file.read((char*)&rows, 4);
     file.read((char*)&cols, 4);

     // Convert from big endian to little endian
     magic_number = bswap32(magic_number);
     number_of_images = bswap32(number_of_images);
     rows = bswap32(rows);
     cols = bswap32(cols);

     if (magic_number != 2051) 
     {
         throw std::runtime_error("Invalid MNIST image file!");
     }

     int image_size = rows * cols;
     std::vector<std::vector<double>> images;

     for (int i = 0; i < number_of_images; ++i) 
     {
         std::vector<unsigned char> buffer(image_size);
         file.read((char*)buffer.data(), image_size);

         std::vector<double> image(image_size);
         for (int j = 0; j < image_size; ++j) 
         {
             image[j] = buffer[j] / 255.0;  // Normalize pixel values
         }

         images.push_back(image);
     }

     return images;
}

std::vector<std::vector<double>> load_mnist_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file " + file_path);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, 4);
    file.read((char*)&number_of_labels, 4);

    magic_number = bswap32(magic_number);
    number_of_labels = bswap32(number_of_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file!");
    }

    std::vector<std::vector<double>> labels;

    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, 1);

        // One-hot encode label
        std::vector<double> one_hot(10, 0.0);
        one_hot[label] = 1.0;

        labels.push_back(one_hot);
    }

    return labels;
}


int main()
{
     // Define architecture: vector of (input_size, output_size) pairs for each layer
     std::vector<std::pair<int, int>> architecture = 
     {
        {784, 16},  
        {16, 16},
        {16,10}
     };

     Network net(architecture);  // Instantiate the network    

     std::cout<<"Input T if you want to train the model or U if you want to use the model "<<std::endl;
     char c;
     std::cin>>c;
     if(c=='T')
     {
         std::string image_file = "/home/aritra/Downloads/train-images-idx3-ubyte";
         std::string label_file = "/home/aritra/Downloads/train-labels-idx1-ubyte";
   
         try 
         {
             std::cout << "Inside the try catch" << std::endl;
             auto X = load_mnist_images(image_file);
             auto Y = load_mnist_labels(label_file);
             std::cout << "MNIST data loaded successfully." << std::endl;
   
             std::cout<<"calling the train function"<<std::endl;
             net.train_network(X, Y, 10, 0.0001); // epochs = 10, learning_rate = 0.01
         } 
         catch (const std::exception& ex) 
         {
             std::cerr << "Error occurred: " << ex.what() << std::endl;
         }
   
   
         net.save_model("trained_model.mlp");
     }
     else if(c=='U')
     {
         std::string path="/home/aritra/Desktop/Neural_Network_From_Scratch/digit_6.png";
         
         
         cv::Mat image = cv::imread(path,cv::IMREAD_GRAYSCALE);

         if(image.empty())
         {
             std::cout<<"No image found"<<std::endl;
             return -1;
         }

         //resizing the image to 28*28 pixels as that is the neuron settings
         cv::resize(image,image,cv::Size(28,28));
         image.convertTo(image,CV_64F,1.0/255.0);

         std::vector<std::vector<double>> input(784, std::vector<double>(1));
         int index = 0;
         for (int i = 0; i < image.rows; ++i) {
             for (int j = 0; j < image.cols; ++j) {
                 input[index++][0] = image.at<double>(i, j);
                 }
         }
         net.load_model("trained_model.mlp");
         std::cout << "Model loaded from 'model.txt'." << std::endl; // or whatever your file is
     
         std::cout << "Input vector shape: " << input.size() << " x " << input[0].size() << std::endl;
         std::vector<std::vector<double>> output = net.forward_propagation(input);

// Find the index of the highest value in output[0] – that’s your digit!
         double max=INT_MIN; int pos=0;
         for(int i=0;i<output.size();i++)
         {
             if(output[i][0]>max)
             {
                 max=output[i][0];
                 pos=i;
             }
             //std::cout<<output[i][0]<<" ";
         }
         std::filesystem::path filepath = path;

         std::string digit= filepath.filename().string();

         std::cout << "Predicted Digit: " <<pos<< std::endl;

     }
     


     //net.train_network(X, Y, 10, 0.01); // epochs = 10, learning_rate = 0.01

     return 0;
}
