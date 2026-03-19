#include <iostream>
#include <chrono>
#include <string>
#include "denoising.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: denoising [sequential|omp|mpi] input.jpg clean.jpg output.jpg [lambda tau sigma theta iter] [num_threads] [force_grayscale]\n";
        std::cout << "  force_grayscale: 0 for color processing (default), 1 to force grayscale\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string input_path = argv[2];
    std::string clean_path = argv[3];
    std::string output_path = argv[4];

    float lambda = (argc > 5) ? atof(argv[5]) : 12.0f;
    float tau = (argc > 6) ? atof(argv[6]) : 0.1f;
    float sigma = (argc > 7) ? atof(argv[7]) : 0.1f;
    float theta = (argc > 8) ? atof(argv[8]) : 1.0f;
    int maxIter = (argc > 9) ? atoi(argv[9]) : 300;
    int num_threads = (argc > 10) ? atoi(argv[10]) : 4;

    // CRITICAL: Default to color processing (force_grayscale = false)
    bool force_grayscale = (argc > 11) ? (atoi(argv[11]) == 1) : false;

    if (mode == "sequential") {
        auto start = std::chrono::high_resolution_clock::now();

        // CRITICAL: Load image with appropriate color mode
        Image input = load_image(input_path, force_grayscale);

        std::cout << "Processing image: " << input.width << "x" << input.height
            << " with " << input.channels << " channels" << std::endl;

        // Run sequential denoising
        Image result = denoise_sequential(input, maxIter, lambda, tau, sigma, theta);

        // CRITICAL: Save the image with its original channels
        save_image(result, output_path);

        auto end = std::chrono::high_resolution_clock::now();
        double exec_time = std::chrono::duration<double>(end - start).count();

        std::cout << "✅ Sequential denoising complete.\n"
            << "Processed " << input.channels << " channels.\n"
            << "Saved to '" << output_path << "'\n"
            << "Execution Time: " << exec_time << " seconds\n";
    }
    else if (mode == "omp") {
        auto start = std::chrono::high_resolution_clock::now();

        // Load image with appropriate color mode
        Image input = load_image(input_path, force_grayscale);

        std::cout << "Processing image: " << input.width << "x" << input.height
            << " with " << input.channels << " channels" << std::endl;

        Image result = denoise_omp(input, maxIter, lambda, tau, sigma, theta, num_threads);
        save_image(result, output_path);

        auto end = std::chrono::high_resolution_clock::now();
        double exec_time = std::chrono::duration<double>(end - start).count();

        std::cout << "✅ OpenMP denoising complete with " << num_threads << " threads.\n"
            << "Processed " << input.channels << " channels.\n"
            << "Saved to '" << output_path << "'\n"
            << "Execution Time: " << exec_time << " seconds\n";
    }
    else if (mode == "mpi") {
        // For MPI, we just call the function that handles MPI initialization/finalization
        denoise_mpi(input_path.c_str(), clean_path.c_str(), output_path.c_str(),
            lambda, tau, sigma, theta, maxIter);
    }
    else {
        std::cout << "Unknown mode: " << mode << "\n";
        std::cout << "Valid modes are: sequential, omp, mpi\n";
        return 1;
    }

    return 0;
}