#pragma once

#include <vector>
#include <string>
#include <stdexcept>

// Enhanced Image structure to support multiple channels
struct Image {
    int width, height;
    int channels;  // Number of color channels (1 for grayscale, 3 for RGB)
    std::vector<float> data;

    // Access element at position (y, x) in specified channel
    float& at(int y, int x, int c = 0) {
        return data[(y * width + x) * channels + c];
    }

    float at(int y, int x, int c = 0) const {
        return data[(y * width + x) * channels + c];
    }

    // Create a new image with same dimensions but single channel
    Image extractChannel(int channel) const {
        if (channel >= channels) {
            throw std::runtime_error("Channel index out of bounds");
        }

        Image result;
        result.width = width;
        result.height = height;
        result.channels = 1;
        result.data.resize(width * height);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                result.at(y, x) = at(y, x, channel);
            }
        }

        return result;
    }

    // Set channel data from single channel image
    void setChannel(const Image& channelData, int channel) {
        if (channel >= channels) {
            throw std::runtime_error("Channel index out of bounds");
        }

        if (channelData.width != width || channelData.height != height || channelData.channels != 1) {
            throw std::runtime_error("Channel dimensions must match");
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                at(y, x, channel) = channelData.at(y, x);
            }
        }
    }
};

// CRITICAL: Function declarations for loading/saving images
Image load_image(const std::string& path, bool force_grayscale = false);
void save_image(const Image& img, const std::string& path);

// For backward compatibility
Image load_grayscale(const std::string& path);
void save_grayscale(const Image& img, const std::string& path);

// Denoising function declarations
Image denoise_sequential(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta);

Image denoise_omp(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta, int num_threads);

void denoise_mpi(const char* input_path, const char* clean_path, const char* output_path,
    float lambda, float tau, float sigma, float theta, int maxIter);