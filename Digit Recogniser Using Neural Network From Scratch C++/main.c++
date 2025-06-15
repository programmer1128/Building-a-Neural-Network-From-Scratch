#include<iostream>
#include"matrices.hpp"
#include<vector>
#include "sigma_function.hpp"
#include"Layer.hpp"
#include"Network.hpp"
#include <fstream>


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

     std::string image_file = "/home/aritra/Downloads/train-images-idx3-ubyte";
     std::string label_file = "/home/aritra/Downloads/train-labels-idx1-ubyte";

     try {
        std::cout << "Inside the try catch" << std::endl;
     auto X = load_mnist_images(image_file);
     auto Y = load_mnist_labels(label_file);
     std::cout << "MNIST data loaded successfully." << std::endl;

     std::cout<<"calling the train function"<<std::endl;
     net.train_network(X, Y, 10, 0.01); // epochs = 10, learning_rate = 0.01
     } 
     catch (const std::exception& ex) 
     {
     std::cerr << "Error occurred: " << ex.what() << std::endl;
     }


     //net.train_network(X, Y, 10, 0.01); // epochs = 10, learning_rate = 0.01

     return 0;
}
