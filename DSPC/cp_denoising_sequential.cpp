#include "denoising.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>

// Only define STB implementation in one file - choose this one
#define STB_IMAGE_IMPLEMENTATION
#include "stb_images.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_images_write.h"

// IMPORTANT: Updated image loading function to support color channels
Image load_image(const std::string& path, bool force_grayscale) {
    int w, h, c;
    // CRITICAL CHANGE: Use 0 to maintain original channels, use 1 to force grayscale
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, force_grayscale ? 1 : 0);
    if (!img) {
        throw std::runtime_error("Could not load image: " + path);
    }

    // If forcing grayscale, ensure c = 1
    if (force_grayscale) {
        c = 1;
    }

    // Create image with appropriate number of channels
    Image image{ w, h, c, std::vector<float>(w * h * c) };

    // Debug output
    std::cout << "Loaded image with " << c << " channels" << std::endl;

    // Copy and normalize pixel values
    for (int i = 0; i < w * h * c; ++i) {
        image.data[i] = img[i] / 255.0f;
    }

    stbi_image_free(img);
    return image;
}

// Backward compatibility
Image load_grayscale(const std::string& path) {
    return load_image(path, true);
}

// IMPORTANT: Updated save function to handle color images properly
void save_image(const Image& img, const std::string& path) {
    std::vector<unsigned char> output(img.width * img.height * img.channels);

    // Debug output
    std::cout << "Saving image with " << img.channels << " channels" << std::endl;

    // Convert floating point normalized values back to 0-255 range
    for (int i = 0; i < img.width * img.height * img.channels; ++i) {
        output[i] = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, img.data[i])) * 255.0f);
    }

    int result;
    // CRITICAL: Use img.channels when saving the image
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
        result = stbi_write_png(path.c_str(), img.width, img.height, img.channels, output.data(), img.width * img.channels);
    }
    else {
        result = stbi_write_jpg(path.c_str(), img.width, img.height, img.channels, output.data(), 95);
    }

    if (!result) {
        std::cerr << "Error writing image: " << path << std::endl;
    }
}

// Backward compatibility
void save_grayscale(const Image& img, const std::string& path) {
    // If image is already grayscale, save directly
    if (img.channels == 1) {
        save_image(img, path);
        return;
    }

    // Otherwise, convert to grayscale
    Image gray{ img.width, img.height, 1, std::vector<float>(img.width * img.height) };

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            // Use standard RGB to grayscale conversion formula
            gray.at(y, x) = 0.299f * img.at(y, x, 0) + 0.587f * img.at(y, x, 1) + 0.114f * img.at(y, x, 2);
        }
    }

    save_image(gray, path);
}

// Single channel denoising implementation (used as a helper function)
Image denoise_channel(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta) {

    int w = input.width, h = input.height;

    Image x = input;                        // Primal variable
    Image x_bar = input;                   // Extrapolated variable
    Image x_old = input;

    Image y1{ w, h, 1, std::vector<float>(w * h) };  // Dual variable (horizontal gradient)
    Image y2{ w, h, 1, std::vector<float>(w * h) };  // Dual variable (vertical gradient)
    Image div{ w, h, 1, std::vector<float>(w * h) }; // Divergence image

    auto grad = [](const Image& img, int x, int y, int dir, int w, int h) {
        if (dir == 0) // horizontal
            return img.at(y, (x + 1 < w) ? x + 1 : x) - img.at(y, x);
        else // vertical
            return img.at((y + 1 < h) ? y + 1 : y, x) - img.at(y, x);
        };

    auto divergence = [&](const Image& y1, const Image& y2, Image& out) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float dx = y1.at(y, x) - (x > 0 ? y1.at(y, x - 1) : 0.0f);
                float dy = y2.at(y, x) - (y > 0 ? y2.at(y - 1, x) : 0.0f);
                out.at(y, x) = dx + dy;
            }
        };

    for (int iter = 0; iter < maxIter; ++iter) {
        // Store previous x
        x_old = x;

        // ---- Dual Update ----
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                float grad_x = grad(x_bar, x_pos, y, 0, w, h);
                float grad_y = grad(x_bar, x_pos, y, 1, w, h);
                float grad_mag = sqrt(grad_x * grad_x + grad_y * grad_y);

                float alpha = 1.0f;
                float edge_weight = 1.0f / (1.0f + alpha * grad_mag);

                grad_x *= edge_weight;
                grad_y *= edge_weight;

                float new_y1 = y1.at(y, x_pos) + sigma * grad_x;
                float new_y2 = y2.at(y, x_pos) + sigma * grad_y;

                float norm = sqrt(new_y1 * new_y1 + new_y2 * new_y2);
                float scale = std::max(1.0f, norm);

                y1.at(y, x_pos) = new_y1 / scale;
                y2.at(y, x_pos) = new_y2 / scale;
            }
        }

        // ---- Primal Update ----
        divergence(y1, y2, div);
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                float numerator = x.at(y, x_pos) + tau * div.at(y, x_pos) + tau * lambda * input.at(y, x_pos);
                float denominator = 1.0f + tau * lambda;
                x.at(y, x_pos) = numerator / denominator;
            }
        }

        // ---- Extrapolation ----
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                x_bar.at(y, x_pos) = x.at(y, x_pos) + theta * (x.at(y, x_pos) - x_old.at(y, x_pos));
            }
        }
    }

    return x;
}

// CRITICAL: Main sequential denoising function for multi-channel images
Image denoise_sequential(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta) {

    // If single channel, use the standard implementation
    if (input.channels == 1) {
        return denoise_channel(input, maxIter, lambda, tau, sigma, theta);
    }

    // CRITICAL: For multi-channel (color) images, create a result with the SAME number of channels
    Image result{ input.width, input.height, input.channels, std::vector<float>(input.width * input.height * input.channels) };

    std::cout << "Processing color image with " << input.channels << " channels" << std::endl;

    // Process each channel separately
    for (int c = 0; c < input.channels; ++c) {
        std::cout << "Processing channel " << c + 1 << " of " << input.channels << std::endl;

        // Extract single channel
        Image channel = input.extractChannel(c);

        // Denoise the channel
        Image denoised_channel = denoise_channel(channel, maxIter, lambda, tau, sigma, theta);

        // Store the denoised channel in the result
        result.setChannel(denoised_channel, c);
    }

    // CRITICAL: Verify the result has the correct number of channels
    std::cout << "Completed denoising. Result has " << result.channels << " channels" << std::endl;

    return result;
}