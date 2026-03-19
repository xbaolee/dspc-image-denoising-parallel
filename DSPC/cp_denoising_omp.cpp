#include "denoising.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// Single channel denoising with OpenMP
Image denoise_channel_omp(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta, int num_threads) {

    int w = input.width, h = input.height;

    // Set number of threads explicitly
    omp_set_num_threads(num_threads);

    // Show thread count for verification
#pragma omp parallel
    {
#pragma omp master
        {
            std::cout << "Running with " << omp_get_num_threads() << " OpenMP threads" << std::endl;
        }
    }

    // Create all required Images - use block allocation for better cache performance
    Image x = input;                       // Primal variable
    Image x_bar = input;                   // Extrapolated variable
    Image x_old = input;                   // Previous iteration
    Image y1{ w, h, 1, std::vector<float>(w * h, 0.0f) };  // Dual variable (horizontal gradient)
    Image y2{ w, h, 1, std::vector<float>(w * h, 0.0f) };  // Dual variable (vertical gradient)
    Image div{ w, h, 1, std::vector<float>(w * h, 0.0f) }; // Divergence image

    // Calculate optimal chunk size based on image size and thread count
    int chunk_size = std::max(16, h / (num_threads * 4));

    // Pre-declare variables to avoid reallocation
    float grad_x, grad_y, grad_mag, edge_weight, new_y1, new_y2, norm, scale;
    float numerator, denominator;
    float dx, dy;

    // For timing the algorithm only (not file I/O)
    double start_time = omp_get_wtime();

    // Main iteration loop
    for (int iter = 0; iter < maxIter; ++iter) {
        // Copy x to x_old (no need to parallelize this small operation)
        x_old = x;

        // ---- Dual Update ----
        // Use collapsed loops with explicit chunk size for better load balancing
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size) private(grad_x, grad_y, grad_mag, edge_weight, new_y1, new_y2, norm, scale)
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                // Calculate gradients
                grad_x = x_bar.at(y, (x_pos + 1 < w) ? x_pos + 1 : x_pos) - x_bar.at(y, x_pos);
                grad_y = x_bar.at((y + 1 < h) ? y + 1 : y, x_pos) - x_bar.at(y, x_pos);

                // Calculate edge weight
                grad_mag = std::sqrt(grad_x * grad_x + grad_y * grad_y);
                edge_weight = 1.0f / (1.0f + grad_mag);

                // Apply edge weight
                grad_x *= edge_weight;
                grad_y *= edge_weight;

                // Update dual variables
                new_y1 = y1.at(y, x_pos) + sigma * grad_x;
                new_y2 = y2.at(y, x_pos) + sigma * grad_y;

                // Normalize
                norm = std::sqrt(new_y1 * new_y1 + new_y2 * new_y2);
                scale = std::max(1.0f, norm);

                // Store updated values
                y1.at(y, x_pos) = new_y1 / scale;
                y2.at(y, x_pos) = new_y2 / scale;
            }
        }

        // ---- Calculate Divergence ----
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size) private(dx, dy)
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                dx = y1.at(y, x_pos) - (x_pos > 0 ? y1.at(y, x_pos - 1) : 0.0f);
                dy = y2.at(y, x_pos) - (y > 0 ? y2.at(y - 1, x_pos) : 0.0f);
                div.at(y, x_pos) = dx + dy;
            }
        }

        // ---- Primal Update ----
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size) private(numerator, denominator)
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                numerator = x.at(y, x_pos) + tau * div.at(y, x_pos) + tau * lambda * input.at(y, x_pos);
                denominator = 1.0f + tau * lambda;
                x.at(y, x_pos) = numerator / denominator;
            }
        }

        // ---- Extrapolation ----
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                x_bar.at(y, x_pos) = x.at(y, x_pos) + theta * (x.at(y, x_pos) - x_old.at(y, x_pos));
            }
        }
    }

    // Calculate execution time of algorithm only
    double exec_time = omp_get_wtime() - start_time;
    std::cout << "Channel denoising time: " << exec_time << " seconds" << std::endl;

    return x;
}

// Optimized OpenMP implementation for multi-channel images
Image denoise_omp(const Image& input, int maxIter, float lambda,
    float tau, float sigma, float theta, int num_threads) {

    // If single channel, use the channel implementation directly
    if (input.channels == 1) {
        return denoise_channel_omp(input, maxIter, lambda, tau, sigma, theta, num_threads);
    }

    // For multi-channel (color) images, create a result with the SAME number of channels
    Image result{ input.width, input.height, input.channels, std::vector<float>(input.width * input.height * input.channels) };

    // Start timing for the whole operation
    double start_time = omp_get_wtime();

    std::cout << "Processing color image with " << input.channels << " channels" << std::endl;

    // IMPROVED: Process channels in parallel using OpenMP
    // Distribute available threads among channels and within each channel's processing
    int threads_per_channel = std::max(1, num_threads / input.channels);
    int remaining_threads = num_threads % input.channels;

    std::cout << "Parallelizing across channels with " << threads_per_channel
        << " threads per channel" << std::endl;

    // Use OpenMP to process channels in parallel
#pragma omp parallel for num_threads(input.channels)
    for (int c = 0; c < input.channels; ++c) {
        // Calculate threads for this channel (distribute remaining threads)
        int channel_threads = threads_per_channel + (c < remaining_threads ? 1 : 0);

        // Extract single channel
        Image channel = input.extractChannel(c);

#pragma omp critical
        std::cout << "Processing channel " << c + 1 << " of " << input.channels
            << " with " << channel_threads << " threads" << std::endl;

        // Denoise the channel - pass the appropriate thread count for this channel
        Image denoised_channel = denoise_channel_omp(channel, maxIter, lambda, tau, sigma, theta, channel_threads);

        // Store the denoised channel in the result
#pragma omp critical
        {
            result.setChannel(denoised_channel, c);
            std::cout << "Completed channel " << c + 1 << " of " << input.channels << std::endl;
        }
    }

    // Calculate total execution time
    double exec_time = omp_get_wtime() - start_time;
    std::cout << "Total execution time: " << exec_time << " seconds" << std::endl;

    // Verify the result has the correct number of channels
    std::cout << "Completed denoising. Result has " << result.channels << " channels" << std::endl;

    return result;
}