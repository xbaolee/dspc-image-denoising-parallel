#include "denoising.h"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <algorithm>

// MPI denoising function for multi-channel images
void denoise_mpi(const char* input_path, const char* clean_path, const char* output_path,
    float lambda, float tau, float sigma, float theta, int maxIter) {

    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 1) {
        if (rank == 0) {
            std::cout << "Error: Need at least 1 MPI process\n";
        }
        MPI_Finalize();
        return;
    }

    Image full_img;
    int width = 0, height = 0, channels = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        try {
            // Load image with original color channels
            full_img = load_image(input_path, false);
            width = full_img.width;
            height = full_img.height;
            channels = full_img.channels;
            std::cout << "Loaded input image: " << width << "x" << height << " with "
                << channels << " channels" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading image: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }

    // Broadcast image dimensions and channels to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Processing with " << size << " MPI processes" << std::endl;
        std::cout << "Parameters: lambda=" << lambda << ", tau=" << tau
            << ", sigma=" << sigma << ", theta=" << theta
            << ", iterations=" << maxIter << std::endl;
    }

    // Create result image container on rank 0
    std::vector<float> full_result;
    if (rank == 0) {
        full_result.resize(width * height * channels);
    }

    // Process each channel separately
    for (int c = 0; c < channels; ++c) {
        if (rank == 0) {
            std::cout << "Processing channel " << c + 1 << " of " << channels << std::endl;
        }

        // Extract channel data on rank 0
        std::vector<float> channel_data;
        if (rank == 0) {
            channel_data.resize(width * height);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    channel_data[y * width + x] = full_img.at(y, x, c);
                }
            }
        }

        // Calculate work distribution
        int block_size = height / size;
        int extra = height % size;
        int local_start = rank * block_size + std::min(rank, extra);
        int local_rows = block_size + (rank < extra ? 1 : 0);

        // Create local image for this process (single channel)
        Image local_img{ width, local_rows, 1, std::vector<float>(width * local_rows) };

        // Calculate scatter counts and displacements
        std::vector<int> sendcounts(size), displs(size);
        for (int i = 0; i < size; ++i) {
            int start = i * block_size + std::min(i, extra);
            int rows = block_size + (i < extra ? 1 : 0);
            sendcounts[i] = rows * width;
            displs[i] = start * width;
        }

        // Distribute channel data to all processes
        MPI_Scatterv(rank == 0 ? channel_data.data() : nullptr,
            sendcounts.data(), displs.data(), MPI_FLOAT,
            local_img.data.data(), local_rows * width, MPI_FLOAT,
            0, MPI_COMM_WORLD);

        // Process the local portion
        auto start = std::chrono::high_resolution_clock::now();

        // Denoise local portion (single channel)
        Image local_result = denoise_sequential(local_img, maxIter, lambda, tau, sigma, theta);

        auto end = std::chrono::high_resolution_clock::now();
        double exec_time = std::chrono::duration<double>(end - start).count();

        // Gather timing information
        double max_time;
        MPI_Reduce(&exec_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Channel " << c + 1 << " processing time: " << max_time << " seconds" << std::endl;
        }

        // Create a buffer for the gathered channel result
        std::vector<float> channel_result;
        if (rank == 0) {
            channel_result.resize(width * height);
        }

        // Gather processed channel data back to rank 0
        MPI_Gatherv(local_result.data.data(), local_rows * width, MPI_FLOAT,
            rank == 0 ? channel_result.data() : nullptr,
            sendcounts.data(), displs.data(), MPI_FLOAT,
            0, MPI_COMM_WORLD);

        // On rank 0, copy the channel result to the correct position in the full result
        if (rank == 0) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Place channel data in the correct channel position in the full result
                    full_result[(y * width + x) * channels + c] = channel_result[y * width + x];
                }
            }
        }
    }

    // Save the result image on rank 0
    if (rank == 0) {
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();

        Image final_img{ width, height, channels, std::move(full_result) };
        save_image(final_img, output_path);

        std::cout << "✅ MPI denoising complete.\n"
            << "Processed " << channels << " channels.\n"
            << "Saved to '" << output_path << "'\n"
            << "Execution Time: " << total_time << " seconds\n";
    }

    MPI_Finalize();
}