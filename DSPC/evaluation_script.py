from PIL import Image
import numpy as np
import os
import subprocess
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import pandas as pd
import time as time_module
import sys
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate unique session ID based on timestamp
def generate_session_id():
    """Generate a unique ID based on current date and time"""
    now = datetime.datetime.now()
    # Format: YYYYMMDD_HHMMSS
    return now.strftime("%Y%m%d_%H%M%S")

# Create a unique session ID for this run
SESSION_ID = generate_session_id()

# Define resolutions and tuned parameters
resolutions = [(244, 244), (512, 512), (1024, 1024), (2048, 2048), (3840, 2160), (5120, 2880), (7680, 4320)]

param_map = {
    "244x244":    (10, 0.1,   0.1,   1.0,   300),
    "512x512":    (6, 0.08,  0.08,  1.0,   500),
    "1024x1024":  (2, 0.05,  0.05,  1.0,   900),
    "2048x2048":  (1, 0.025,  0.025,  1.0,  1800),
    "3840x2160":  (0.5, 0.018, 0.018, 1.0,  2500),
    "5120x2880":  (0.5, 0.015, 0.015, 1.0,  3500),
    "7680x4320":  (0.2, 0.012, 0.012, 1.0,  4500),
}

# Process/thread counts for different methods
# For MPI, we use 1,2,4,8,16 processes
mpi_process_counts = [2, 4, 8, 16]
# For OpenMP, we use 1,2,4,8,16 threads
omp_thread_counts = [2, 4, 8, 16]
# For benchmark tests, include all counts for all methods
benchmark_counts = {
    "sequential": [1],
    "omp": [2, 4, 8, 16],
    "mpi": [2, 4, 8, 16]
}

# Create results directory with timestamp
results_base_dir = "benchmark_results"
os.makedirs(results_base_dir, exist_ok=True)

# Function to get consistent output filename
def get_output_filename(method, img_idx, count, resolution, output_dir="output_images"):
    """Generate consistent filename for output images"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    count_str = f"{count}{'proc' if method == 'mpi' else 'thr'}"
    filename = f"{method}_img{img_idx}_{resolution}_{count_str}.jpg"
    return os.path.join(output_dir, filename)

# Path to executables
mpi_exec = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

# Check if Release build exists and use it, otherwise fall back to Debug
if os.path.exists("x64/Release/DSPC.exe"):
    executable_path = r"x64\Release\DSPC.exe"
    print("Using Release build (faster performance)")
else:
    executable_path = r"x64\Debug\DSPC.exe"
    print("WARNING: Using Debug build - performance will be slow")
    print("Consider building in Release mode for better performance")

# Function to organize DataFrame output
def organize_dataframe(df):
    """Organize and clean up the DataFrame before saving to CSV"""
    # Define the order of columns we want
    desired_columns = [
        "Image_ID", "Resolution", "Width", "Height", "Size_MB", 
        "Method", "Process_Thread_Count", "Execution_Time_sec", 
        "Time_per_Megapixel", "PSNR", "SSIM", "MSE", 
        "Throughput_pixels_per_sec", "Baseline_Time", "Speedup"
    ]
    
    # Filter columns to only include those we want (and that exist)
    available_columns = [col for col in desired_columns if col in df.columns]
    
    # Remove "Parameters" column if it exists
    if "Parameters" in df.columns:
        df = df.drop(columns=["Parameters"])
    
    # Create a new DataFrame with just the columns we want, in the right order
    df_cleaned = df[available_columns].copy()
    
    # Sort by Image_ID and Process_Thread_Count
    if "Image_ID" in df_cleaned.columns and "Process_Thread_Count" in df_cleaned.columns:
        df_cleaned = df_cleaned.sort_values(by=["Image_ID", "Process_Thread_Count"])
    
    return df_cleaned

# Function to run denoising with specific parameters
# Function to run denoising with specific parameters
def run_denoising(method, input_path, clean_path, output_path, params, processes_or_threads=1):
    """
    Run the denoising algorithm with specified parameters
    
    method: 'sequential', 'omp', or 'mpi'
    input_path: path to input image
    clean_path: path to clean image (for PSNR calculation)
    output_path: path to save output image
    params: (lambda, tau, sigma, theta, max_iter)
    processes_or_threads: number of processes (MPI) or threads (OMP)
    """
    lam, tau, sigma, theta, max_iter = params
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None
    
    # Handle whether this is a basic or detailed analysis mode
    # IMPORTANT: Only consider it "basic mode" if input and clean are actually the same file
    # (this preserves metrics calculation for options 1 and 2 in the main menu)
    basic_analysis = (clean_path == input_path)
    
    if basic_analysis:
        print(f"Running in basic mode (no separate clean reference image)")
    else:
        if not os.path.exists(clean_path):
            print(f"Warning: Clean reference image not found. Using input as reference.")
            clean_path = input_path
            basic_analysis = True
        else:
            print(f"Running with separate clean reference image for metrics calculation")
    
    if method == "sequential":
            cmd = [
                executable_path, "sequential",
                input_path, clean_path, output_path,
                str(lam), str(tau), str(sigma), str(theta), str(max_iter)
                
            ]

    elif method == "omp":
        cmd = [
            executable_path, "omp",  
            input_path, clean_path, output_path, 
            str(lam), str(tau), str(sigma), str(theta), str(max_iter),  
            str(processes_or_threads)
        ]
    elif method == "mpi":
        cmd = [
            mpi_exec, "-n", str(processes_or_threads),
            executable_path, "mpi",
            input_path, clean_path, output_path,
            str(lam), str(tau), str(sigma), str(theta), str(max_iter)
    
        ]
    else:
        print(f"Unknown method: {method}")
        return None
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time_module.time()  # Keep for fallback timing
    process = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time_module.time()  # Keep for fallback timing
    
    if process.returncode != 0:
        print(f"Error running denoising with {method}")
        print(f"Error output: {process.stderr}")
        return None
    
    # Try to extract execution time from C++ program output
    cpp_exec_time = None
    algorithm_time = None
    
    if process.stdout:
        # Look for execution time in the output
        for line in process.stdout.splitlines():
            # Look for the standard "Execution Time: X.XX seconds" format
            if "Execution Time:" in line:
                try:
                    time_str = line.split("Execution Time:")[1].split("seconds")[0].strip()
                    cpp_exec_time = float(time_str)
                    break
                except (ValueError, IndexError):
                    pass
            
            # Try to find other time reports using regex
            time_match = re.search(r'time:?\s*([\d\.]+)', line.lower())
            if time_match:
                try:
                    algorithm_time = float(time_match.group(1))
                except ValueError:
                    pass
    
    # Use the best available time measurement
    if cpp_exec_time is not None:
        exec_time = cpp_exec_time
        time_source = "C++ explicit"
    elif algorithm_time is not None:
        exec_time = algorithm_time
        time_source = "C++ pattern match"
    else:
        exec_time = end_time - start_time
        time_source = "Python subprocess"
    
    print(f"Completed in {exec_time:.2f} seconds (timing source: {time_source})")
    
    result = {
        "Execution_Time_sec": exec_time,
        "Method": method,
        "Process_Thread_Count": processes_or_threads
    }
    
    # Calculate image quality metrics if:
    # 1. Output exists
    # 2. We're not in basic analysis mode (input != clean) OR
    # 3. We're running from main options 1 or 2 (where clean path should always be used for metrics)
    if os.path.exists(output_path):
        try:
            # Try to load images - check if they're color or grayscale
            clean_img = np.array(Image.open(clean_path))
            denoised_img = np.array(Image.open(output_path))
            
            # Check if images are grayscale or color
            clean_is_grayscale = len(clean_img.shape) == 2 or (len(clean_img.shape) == 3 and clean_img.shape[2] == 1)
            denoised_is_grayscale = len(denoised_img.shape) == 2 or (len(denoised_img.shape) == 3 and denoised_img.shape[2] == 1)
            
            # Convert to grayscale if needed for comparison
            if clean_is_grayscale != denoised_is_grayscale:
                print("Warning: Clean and denoised images have different color formats.")
                if not clean_is_grayscale:  # Clean is color, denoised is grayscale
                    clean_img = np.array(Image.fromarray(clean_img).convert("L"))
                else:  # Clean is grayscale, denoised is color
                    denoised_img = np.array(Image.fromarray(denoised_img).convert("L"))
            
            # For color images, calculate metrics per channel and average
            if len(clean_img.shape) == 3 and clean_img.shape[2] >= 3:
                # RGB image - calculate metrics for each channel
                psnr_values = []
                ssim_values = []
                mse_values = []
                
                for c in range(3):  # RGB channels
                    clean_channel = clean_img[:,:,c]
                    denoised_channel = denoised_img[:,:,c]
                    
                    psnr = peak_signal_noise_ratio(clean_channel, denoised_channel)
                    psnr_values.append(psnr)
                    
                    ssim = structural_similarity(clean_channel, denoised_channel)
                    ssim_values.append(ssim)
                    
                    mse = mean_squared_error(clean_channel, denoised_channel)
                    mse_values.append(mse)
                
                # Average the metrics
                result["PSNR"] = sum(psnr_values) / len(psnr_values)
                result["SSIM"] = sum(ssim_values) / len(ssim_values)
                result["MSE"] = sum(mse_values) / len(mse_values)
                
                print(f"Color image metrics (average across channels):")
                print(f"PSNR: {result['PSNR']:.2f} dB")
                print(f"SSIM: {result['SSIM']:.4f}")
                print(f"MSE: {result['MSE']:.2f}")
            else:
                # Grayscale image - direct calculation
                if clean_img.shape == denoised_img.shape:
                    psnr = peak_signal_noise_ratio(clean_img, denoised_img)
                    result["PSNR"] = psnr
            
                    ssim = structural_similarity(clean_img, denoised_img)
                    result["SSIM"] = ssim
                    
                    # Add MSE
                    mse = mean_squared_error(clean_img, denoised_img)
                    result["MSE"] = mse
                    
                    print(f"Grayscale image metrics:")
                    print(f"PSNR: {psnr:.2f} dB")
                    print(f"SSIM: {ssim:.4f}")
                    print(f"MSE: {mse:.2f}")
                else:
                    print(f"Warning: Clean and denoised images have different dimensions")
        except Exception as e:
            print(f"Error calculating image metrics: {e}")
            import traceback
            traceback.print_exc()
    elif basic_analysis:
        print("Skipping metrics calculation (no separate clean reference image)")
    
    return result

# Function to generate horizontal bar chart for run_full_benchmark
def generate_horizontal_bar_chart(df, output_name, output_dir, prefix=""):
    """Generate horizontal bar chart comparing thread/process performance with bars grouped by resolution"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if DataFrame is not empty
    if df.empty:
        print("No data to visualize")
        return
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Group data by method and resolution
    methods = sorted(df["Method"].unique())
    
    # Sort resolutions by size
    resolution_info = []
    for res in df["Resolution"].unique():
        try:
            if 'x' in res:
                w, h = map(int, res.split('x'))
                pixel_count = w * h
                resolution_info.append((res, pixel_count))
            else:
                resolution_info.append((res, float('inf')))
        except:
            resolution_info.append((res, float('inf')))
    
    # Sort resolutions by pixel count
    resolution_info.sort(key=lambda x: x[1])
    sorted_resolutions = [info[0] for info in resolution_info]
    
    # Create a horizontal bar chart for each method
    for method in methods:
        method_df = df[df["Method"] == method]
        
        # Skip if no data for this method
        if len(method_df) == 0:
            continue
        
        # Get unique thread/process counts for this method
        counts = sorted(method_df["Process_Thread_Count"].unique())
        
        # Skip methods with only one thread/process count (e.g., sequential)
        if len(counts) <= 1:
            print(f"Skipping {method} chart - only one thread/process count available")
            continue
        
        # Create figure - width based on number of resolutions
        height = max(10, 2 + len(counts) * 0.8)
        plt.figure(figsize=(16, height))
        
        # Prepare data for plotting
        plot_data = []
        
        for count in counts:
            count_data = method_df[method_df["Process_Thread_Count"] == count]
            
            for res in sorted_resolutions:
                res_data = count_data[count_data["Resolution"] == res]
                
                if len(res_data) > 0:
                    # Get mean execution time
                    exec_time = res_data["Execution_Time_sec"].mean()
                    
                    plot_data.append({
                        "Resolution": res,
                        "Process_Thread_Count": count,
                        "Execution_Time_sec": exec_time,
                        "Plot_Label": f"{count} {'processes' if method == 'mpi' else 'threads'}"
                    })
        
        if not plot_data:
            print(f"No data to plot for method: {method}")
            continue
            
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Get unique thread/process count labels
        thread_labels = [f"{count} {'processes' if method == 'mpi' else 'threads' if method != 'cuda' else 'blocksize'}" for count in counts]
        
        # Create horizontal bar chart with resolution as hue
        plt.figure(figsize=(16, height))
        
        # Create a horizontal bar chart with resolutions as different color bars
        ax = sns.barplot(
            y="Plot_Label", 
            x="Execution_Time_sec", 
            hue="Resolution",
            data=plot_df,
            orient="h",
            palette="Set2",  # More distinctive color palette
            order=thread_labels,
            dodge=True  # Set to True to place bars side by side instead of stacking
        )
        
        # Add value labels to the bars
        for i, p in enumerate(ax.patches):
            # Only add labels for bars with width > 0
            if p.get_width() > 0:
                ax.annotate(
                    f"{p.get_width():.2f}s",
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    ha="left", va="center",
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=9
                )
        
        # Set chart title and labels
        plt.title(output_name,fontsize=16, pad=20)
        plt.xlabel("Execution Time (seconds)", fontsize=14)
        plt.ylabel(f"{'Processes' if method == 'mpi' else 'Threads' if method != 'cuda' else 'Blocksize'} Count", fontsize=14)
        
        # Move legend to the right side
        plt.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{prefix}{method}_horizontal_comparison.png")
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Horizontal bar chart for {method} saved to: {chart_path}")
        
        # Open the file
        try:
            os.startfile(chart_path)
        except:
            try:
                # For macOS
                subprocess.call(['open', chart_path])
            except:
                try:
                    # For Linux
                    subprocess.call(['xdg-open', chart_path])
                except:
                    print(f"Could not automatically open the chart. Please find it at: {chart_path}")

def generate_cuda_horizontal_bar_chart(session_name):
    """Specific function to generate horizontal bar chart for CUDA results"""
    print("\n=== CUDA Results Visualization ===")
    
    # Prompt for CUDA results CSV
    cuda_csv_path = input("Enter the path to the CUDA results CSV file: ").strip('"')
    
    # Validate file exists
    if not os.path.exists(cuda_csv_path):
        print(f"Error: File not found at {cuda_csv_path}")
        return
    
    # Read the CSV
    try:
        df = pd.read_csv(cuda_csv_path)
        print(f"Successfully loaded CUDA results from {cuda_csv_path}")
        
        # Debug: Print first few rows and column names
        print("\nFirst few rows of the DataFrame:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Ask for output name
    output_name = session_name
    
    # Create visualization output directory
    vis_dir = os.path.join(results_base_dir, output_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    
    # Get unique block sizes and resolutions
    block_sizes = df['Process_Thread_Count'].unique()
    resolutions = df['Resolution'].unique()
    
    # Prepare plot data
    for block_size in block_sizes:
        for resolution in resolutions:
            subset = df[(df['Process_Thread_Count'] == block_size) & 
                        (df['Resolution'] == resolution)]
            
            if not subset.empty:
                plot_data.append({
                    'Resolution': resolution,
                    'Block_Size': block_size,
                    'Execution_Time_sec': subset['Execution_Time_sec'].mean()
                })
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Sort block sizes and resolutions
    block_size_order = sorted(plot_df['Block_Size'].unique())
    resolution_order = sorted(plot_df['Resolution'].unique(), 
                               key=lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
    
    # Generate horizontal bar chart
    plt.figure(figsize=(16, 10))
    
    ax = sns.barplot(
        y='Block_Size', 
        x='Execution_Time_sec', 
        hue='Resolution',
        data=plot_df,
        orient='h',
        palette='Set2',
        order=block_size_order,
        dodge=True
    )
    
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        if p.get_width() > 0:
            ax.annotate(
                f"{p.get_width():.4f}s", 
                (p.get_width(), p.get_y() + p.get_height()/2),
                ha='left', va='center',
                xytext=(5, 0),
                textcoords='offset points'
            )
    
    plt.title(f"{output_name} - CUDA Performance", fontsize=16)
    plt.xlabel("Execution Time (seconds)", fontsize=14)
    plt.ylabel("Block Size", fontsize=14)
    plt.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    chart_path = os.path.join(vis_dir, f"cuda_horizontal_comparison.png")
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"CUDA horizontal bar chart saved to: {chart_path}")
    
    # Attempt to open the chart
    try:
        os.startfile(chart_path)
    except:
        try:
            subprocess.call(['open', chart_path])
        except:
            try:
                subprocess.call(['xdg-open', chart_path])
            except:
                print(f"Could not automatically open the chart. Please find it at: {chart_path}")

def generate_execution_time_chart(df, output_name, output_dir, filename_prefix):
    """Generate bar chart for execution time comparison"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if DataFrame is not empty
    if df.empty:
        print("No data to visualize for execution time chart")
        return None
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Get available resolutions from the data
    available_resolutions = df["Resolution"].unique()
    
    # Create a list of tuples with resolution and pixel count for sorting
    resolution_info = []
    for res in available_resolutions:
        try:
            if 'x' in res:
                w, h = map(int, res.split('x'))
                pixel_count = w * h
                resolution_info.append((res, pixel_count))
            else:
                resolution_info.append((res, float('inf')))
        except:
            resolution_info.append((res, float('inf')))
    
    # Sort resolutions by pixel count
    resolution_info.sort(key=lambda x: x[1])
    sorted_resolutions = [info[0] for info in resolution_info]
    
    # Display sorted resolutions for selection
    print("\nAvailable resolutions for visualization:")
    for i, res in enumerate(sorted_resolutions):
        print(f"{i+1}. {res}")
    
    res_selection = input("Enter resolution numbers to include (comma-separated or 'all'): ").strip()
    
    if res_selection.lower() == 'all':
        selected_resolutions = sorted_resolutions
    else:
        try:
            selection_indices = [int(idx.strip()) - 1 for idx in res_selection.split(',')]
            selected_resolutions = [sorted_resolutions[i] for i in selection_indices 
                                   if 0 <= i < len(sorted_resolutions)]
            
            if not selected_resolutions:
                print("No valid resolutions selected, using all")
                selected_resolutions = sorted_resolutions
                
        except (ValueError, IndexError):
            print("Invalid selection, using all resolutions")
            selected_resolutions = sorted_resolutions
    
    # Create a new column that combines method and thread/process count
    df['Method_Count'] = df.apply(
        lambda row: f"{row['Method']} ({row['Process_Thread_Count']})", 
        axis=1
    )
    
    # Calculate mean execution time per resolution and method+count
    data_to_plot = []
    
    for resolution in selected_resolutions:
        res_df = df[df["Resolution"] == resolution]
        
        for method in sorted(res_df["Method"].unique()):
            method_df = res_df[res_df["Method"] == method]
            
            for count in sorted(method_df["Process_Thread_Count"].unique()):
                count_df = method_df[method_df["Process_Thread_Count"] == count]
                
                if len(count_df) > 0:
                    # Calculate mean execution time
                    exec_time = count_df["Execution_Time_sec"].mean()
                    
                    data_to_plot.append({
                        "Resolution": resolution,
                        "Method_Count": f"{method} ({count})",
                        "Method": method,
                        "Count": count,
                        "Execution_Time_sec": exec_time
                    })
    
    # Create plot DataFrame
    plot_df = pd.DataFrame(data_to_plot)
    
    # Only proceed if we have data
    if not plot_df.empty:
        # Sort the Method_Count values for better visualization
        # First by method name, then by process/thread count
        method_order = {"sequential": 0, "omp": 1, "mpi": 2, "cuda": 3}
        plot_df["Method_Order"] = plot_df["Method"].map(lambda x: method_order.get(x, 99))
        plot_df = plot_df.sort_values(["Method_Order", "Count"])
        
        # Create the bar chart
        plt.figure(figsize=(16, 10))
        
        # Create the bar chart with method_count as hue
        ax = sns.barplot(
            x="Resolution", 
            y="Execution_Time_sec", 
            hue="Method_Count", 
            data=plot_df,
            order=selected_resolutions,
            hue_order=plot_df["Method_Count"].unique()  # Use the sorted order
        )
        
        plt.title(output_name, fontsize=16)
        plt.xlabel("Resolution", fontsize=14)
        plt.ylabel("Execution Time (seconds)", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Method (thread/process/blocksize)", title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on top of bars, but ONLY for actual data points (not origin)
        for p in ax.patches:
            # Only add labels for bars with height > 0
            if p.get_height() > 0:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{filename_prefix}_execution_time.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Execution time chart saved to: {output_path}")
        return output_path
    else:
        print("No data available for plotting execution time")
        return None


# Function to generate performance comparison charts
def generate_comparison_charts(df,output_dir, prefix=""):
    """Generate performance comparison charts from benchmark data"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if DataFrame is not empty
    if df.empty:
        print("No data to visualize")
        return
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Create a comprehensive execution time comparison - all methods and all thread/process counts
    plt.figure(figsize=(16, 10))
    
    # Create a new column that combines method and thread/process count
    df['Method_Count'] = df.apply(lambda row: f"{row['Method']} ({row['Process_Thread_Count']})", axis=1)
    
    # Sort resolutions by size (numerically)
    resolution_order = {}
    for res in df["Resolution"].unique():
        try:
            # Try to extract width and height from the resolution string
            if 'x' in res:
                w, h = map(int, res.split('x'))
                resolution_order[res] = w * h
            else:
                # If it's not in the expected format, use a default large value
                resolution_order[res] = float('inf')
        except:
            resolution_order[res] = float('inf')
    
    # Sort resolutions by pixel count
    sorted_resolutions = sorted(resolution_order.keys(), key=lambda x: resolution_order[x])
    
    # Group data by resolution and method+count
    data_to_plot = []
    
    for resolution in sorted_resolutions:
        res_df = df[df["Resolution"] == resolution]
        
        # Skip if there's no data for this resolution
        if len(res_df) == 0:
            continue
        
        for method in sorted(df["Method"].unique()):
            method_df = res_df[res_df["Method"] == method]
            
            for count in sorted(method_df["Process_Thread_Count"].unique()):
                count_df = method_df[method_df["Process_Thread_Count"] == count]
                
                if len(count_df) == 0:
                    continue
                
                # Calculate mean execution time
                exec_time = count_df["Execution_Time_sec"].mean()
                
                # Create entry for plotting
                data_to_plot.append({
                    "Resolution": resolution,
                    "Method_Count": f"{method} ({count})",
                    "Method": method,
                    "Count": count,
                    "Execution_Time_sec": exec_time
                })
    
    # Create plot DataFrame
    plot_df = pd.DataFrame(data_to_plot)
    
    # Only proceed if we have data
    if not plot_df.empty:
        # Create the bar chart
        plt.figure(figsize=(16, 10))
        
        # Create the bar chart with method_count as hue
        ax = sns.barplot(x="Resolution", y="Execution_Time_sec", hue="Method_Count", data=plot_df, order=sorted_resolutions)
        
        plt.title("Execution Time by Method, Thread/Process/Blocksize, and Resolution", fontsize=16)
        plt.xlabel("Resolution", fontsize=14)
        plt.ylabel("Execution Time (seconds)", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Method (thread/process/blocksize)", title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5),
                       textcoords = 'offset points',
                       fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}execution_time_comparison.png"), bbox_inches='tight')
        plt.close()
        
        print(f"Charts saved to {output_dir}")
    else:
        print("No data available for plotting")

def generate_quality_chart(df, metric_column, metric_title, output_dir, filename_prefix, lower_is_better=False):
    """Generate bar chart for quality metrics (SSIM, PSNR, Throughput, MSE)"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if DataFrame is not empty
    if df.empty:
        print(f"No data to visualize for {metric_title}")
        return None
    
    # Check if metric column exists
    if metric_column not in df.columns:
        print(f"No {metric_column} data available in the CSV")
        return None
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Get available resolutions from the data
    available_resolutions = df["Resolution"].unique()
    
    # Create a list of tuples with resolution and pixel count for sorting
    resolution_info = []
    for res in available_resolutions:
        try:
            if 'x' in res:
                w, h = map(int, res.split('x'))
                pixel_count = w * h
                resolution_info.append((res, pixel_count))
            else:
                resolution_info.append((res, float('inf')))
        except:
            resolution_info.append((res, float('inf')))
    
    # Sort resolutions by pixel count
    resolution_info.sort(key=lambda x: x[1])
    sorted_resolutions = [info[0] for info in resolution_info]
    
    # Display sorted resolutions for selection
    print("\nAvailable resolutions for visualization:")
    for i, res in enumerate(sorted_resolutions):
        print(f"{i+1}. {res}")
    
    res_selection = input("Enter resolution numbers to include (comma-separated or 'all'): ").strip()
    
    if res_selection.lower() == 'all':
        selected_resolutions = sorted_resolutions
    else:
        try:
            selection_indices = [int(idx.strip()) - 1 for idx in res_selection.split(',')]
            selected_resolutions = [sorted_resolutions[i] for i in selection_indices 
                                   if 0 <= i < len(sorted_resolutions)]
            
            if not selected_resolutions:
                print("No valid resolutions selected, using all")
                selected_resolutions = sorted_resolutions
                
        except (ValueError, IndexError):
            print("Invalid selection, using all resolutions")
            selected_resolutions = sorted_resolutions
    
    # Create a new column that combines method and thread/process count
    df['Method_Count'] = df.apply(
        lambda row: f"{row['Method']} ({row['Process_Thread_Count']})", 
        axis=1
    )
    
    # Filter data to only include selected resolutions
    filtered_df = df[df["Resolution"].isin(selected_resolutions)]
    
    # Filter out rows with NaN values in the metric column
    filtered_df = filtered_df.dropna(subset=[metric_column])
    
    # Calculate mean metric per resolution and method+count
    data_to_plot = []
    
    for resolution in selected_resolutions:
        res_df = filtered_df[filtered_df["Resolution"] == resolution]
        
        for method in sorted(res_df["Method"].unique()):
            method_df = res_df[res_df["Method"] == method]
            
            for count in sorted(method_df["Process_Thread_Count"].unique()):
                count_df = method_df[method_df["Process_Thread_Count"] == count]
                
                if len(count_df) > 0:
                    # Calculate mean metric
                    metric_value = count_df[metric_column].mean()
                    
                    data_to_plot.append({
                        "Resolution": resolution,
                        "Method_Count": f"{method} ({count})",
                        "Method": method,
                        "Count": count,
                        "Metric_Value": metric_value
                    })
    
    # Create plot DataFrame
    plot_df = pd.DataFrame(data_to_plot)
    
    # Only proceed if we have data
    if not plot_df.empty:
        # Sort the Method_Count values for better visualization
        # First by method name, then by process/thread count
        method_order = {"sequential": 0, "omp": 1, "mpi": 2, "cuda": 3}
        plot_df["Method_Order"] = plot_df["Method"].map(lambda x: method_order.get(x, 99))
        plot_df = plot_df.sort_values(["Method_Order", "Count"])
        
        # Create the bar chart
        plt.figure(figsize=(16, 10))
        
        # Create the bar chart with method_count as hue
        ax = sns.barplot(
            x="Resolution", 
            y="Metric_Value", 
            hue="Method_Count", 
            data=plot_df,
            order=selected_resolutions,
            hue_order=plot_df["Method_Count"].unique()  # Use the sorted order
        )
        
        plt.title(f"{metric_title} Comparison", fontsize=16)
        plt.xlabel("Resolution", fontsize=14)
        plt.ylabel(metric_title, fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Method (thread/process/blocksize)", title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on top of bars
        for p in ax.patches:
            # Only add labels for bars with height > 0
            if p.get_height() > 0:
                # Format based on the metric
                if metric_column == "PSNR":
                    label_text = f"{p.get_height():.2f} dB"
                elif metric_column == "SSIM":
                    label_text = f"{p.get_height():.3f}"
                elif metric_column == "Throughput_pixels_per_sec":
                    # Format large throughput values with appropriate suffix
                    throughput = p.get_height()
                    if throughput >= 1e6:
                        label_text = f"{throughput/1e6:.2f}M"
                    elif throughput >= 1e3:
                        label_text = f"{throughput/1e3:.2f}K"
                    else:
                        label_text = f"{throughput:.2f}"
                else:
                    label_text = f"{p.get_height():.2f}"
                
                ax.annotate(
                    label_text,
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        plt.tight_layout()
        
        # Create file name based on metric
        metric_name = metric_column.lower()
        if metric_column == "Throughput_pixels_per_sec":
            metric_name = "throughput"
        
        output_path = os.path.join(output_dir, f"{filename_prefix}_{metric_name}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"{metric_title} chart saved to: {output_path}")
        return output_path

# Load sequential execution times from a CSV file
def load_sequential_times(csv_path=None):
    """Load sequential execution times from CSV file"""
    sequential_times = {}
    
    if csv_path is None or csv_path == "-" or not os.path.exists(csv_path):
        print("No baseline CSV file provided or file not found")
        return sequential_times
        
    try:
        df = pd.read_csv(csv_path)
        print("Baseline performance data loaded successfully")
        
        # Create a dictionary that allows lookup by both image ID and resolution
        for _, row in df.iterrows():
            if "Image_ID" in row and "Resolution" in row and "Execution_Time_sec" in row:
                img_id = row["Image_ID"]
                res = row["Resolution"]
                exec_time = row["Execution_Time_sec"]
                
                # Use a tuple of (image_id, resolution) as the key
                key = (img_id, res)
                sequential_times[key] = exec_time
                
        print(f"Loaded baseline times for {len(sequential_times)} image/resolution combinations")
        
    except Exception as e:
        print(f"Warning: Could not load baseline times from CSV: {e}")
        
    return sequential_times

# Function to visualize benchmark results from CSV files
def visualize_from_csv(file_paths):
    """Load data from multiple CSV files and visualize"""
    all_data = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Loaded data from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No valid data files loaded")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create visualization output directory
    vis_dir = os.path.join(results_base_dir, f"visualizations_{SESSION_ID}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate charts
    generate_comparison_charts(combined_df,vis_dir, prefix="combined_")
    print(f"Generated visualizations from {len(file_paths)} files")
  
# Function to run method benchmarks
def run_method_benchmarks(method, img_indices, selected_resolutions, output_dir, sequential_times=None):
    """Run benchmarks for a specific method across selected resolutions"""
    results = []
    
    # Determine thread/process counts to use
    if method == "sequential":
        counts = [1]
    elif method == "omp":
        counts = omp_thread_counts
    elif method == "mpi":
        counts = mpi_process_counts
    else:
        counts = [1]
    
    # Store sequential results for speedup calculation
    sequential_results = {}
    
    # Create output images directory
    images_output_dir = os.path.abspath("output_images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Process each image
    for img_idx in img_indices:
        # Run for each resolution and count
        for (w, h) in selected_resolutions:
            res_str = f"{w}x{h}"
            
            # Get parameters for this resolution
            if res_str in param_map:
                lam, tau, sigma, theta, max_iter = param_map[res_str]
            else:
                print(f"No parameters defined for {res_str}, using defaults")
                lam, tau, sigma, theta, max_iter = 1.0, 0.05, 0.05, 1.0, 500
            
            # Get input and clean paths
            input_path = os.path.abspath(f"input_images/image_{img_idx}_{res_str}.jpg")
            clean_path = os.path.abspath(f"clean_images/image_{img_idx}_{res_str}.jpg")
            
            # Skip if input doesn't exist
            if not os.path.exists(input_path):
                print(f"Skipping {res_str} - input image not found at {input_path}")
                continue
            
            # Check if clean path exists - but always use it for metrics if it exists
            if not os.path.exists(clean_path):
                print(f"Warning: Clean image not found at {clean_path}, using input as reference")
                clean_path = input_path  # Only use input as fallback if clean doesn't exist
            else:
                print(f"Using clean reference image for metrics: {clean_path}")
            
            print(f"\nBenchmarking {method} on {res_str} for image {img_idx}...")
            
            # Prepare to track baseline time for this resolution
            baseline_time = None
            
            # Run for each thread/process count
            for count in counts:
                # Skip displaying "running with 1 thread" for sequential method
                if method == "sequential":
                    print(f"  Running sequential method...")
                else:
                    print(f"  Running with {count} {'processes' if method == 'mpi' else 'threads'}...")
                
                # Define output path with absolute path
                output_path = get_output_filename(method, img_idx, count, res_str, images_output_dir)
                print(f"  Output will be saved to: {output_path}")
                
                # Run test - ALWAYS with clean path for metrics calculation
                result = run_denoising(
                    method, input_path, clean_path, output_path,
                    (lam, tau, sigma, theta, max_iter), count
                )
                
                if not result:
                    print(f"  Failed to run {method}")
                    continue
                
                # Verify that output file was created
                if os.path.exists(output_path):
                    print(f"  ✅ Output file successfully created at: {output_path}")
                else:
                    print(f"  ❌ WARNING: Output file was not created at: {output_path}")
                
                # Add resolution information
                result["Image_ID"] = img_idx
                result["Resolution"] = res_str
                result["Width"] = w
                result["Height"] = h
                            
                # Calculate additional metrics
                size_mb = os.path.getsize(input_path) / (1024 * 1024)
                num_pixels = w * h
                exec_time = result["Execution_Time_sec"]
                time_per_mp = exec_time / (num_pixels / 1e6)
                throughput = num_pixels / exec_time
                
                result["Size_MB"] = round(size_mb, 5)
                result["Time_per_Megapixel"] = round(time_per_mp, 5)
                result["Throughput_pixels_per_sec"] = round(throughput, 2)
                
                # Store sequential result for this image/resolution for baseline comparisons
                key = (img_idx, res_str)
                
                # For sequential method, this becomes our baseline
                if method == "sequential":
                    baseline_time = exec_time
                    sequential_results[key] = exec_time
                    result["Baseline_Time"] = baseline_time
                    result["Speedup"] = 1.0
                
                # Calculate speedup for non-sequential methods
                if method != "sequential":
                    # Use either the baseline from this benchmark or from CSV
                    if baseline_time is not None:
                        speedup = baseline_time / exec_time
                    elif key in sequential_results:
                        baseline_time = sequential_results[key]
                        speedup = baseline_time / exec_time
                    elif sequential_times and key in sequential_times:
                        baseline_time = sequential_times[key]
                        speedup = baseline_time / exec_time
                    else:
                        speedup = None
                    
                    # Add baseline and speedup to result
                    if baseline_time is not None:
                        result["Baseline_Time"] = baseline_time
                    if speedup is not None:
                        result["Speedup"] = round(speedup, 3)
                        print(f"  Speedup vs sequential: {speedup:.2f}x")
                
                # Display quality metrics if available
                if "PSNR" in result:
                    print(f"  PSNR: {result['PSNR']:.2f} dB")
                if "SSIM" in result and result["SSIM"] is not None:
                    print(f"  SSIM: {result['SSIM']:.4f}")
                if "MSE" in result:
                    print(f"  MSE: {result['MSE']:.2f}")
                
                # Add to results
                results.append(result)
                
                # Save progress
                if results:
                    temp_df = organize_dataframe(pd.DataFrame(results))
                    progress_csv = os.path.join(output_dir, f"{method}_progress.csv")
                    temp_df.to_csv(progress_csv, index=False)
    
    # Save method-specific results
    if results:
        df = organize_dataframe(pd.DataFrame(results))
        method_csv = os.path.join(output_dir, f"{method}_results.csv")
        df.to_csv(method_csv, index=False)
        print(f"\n{method.capitalize()} results saved to: {method_csv}")
        
        # Generate method-specific charts
        chart_dir = os.path.join(output_dir, f"{method}_charts")
        generate_comparison_charts(df, chart_dir, prefix=f"{method}_")
    
    return results

# Function to visualize existing results
def visualize_existing_results():
    """Load and visualize results from existing CSV files"""
    print("\n=== Visualize Existing Results ===")
    
    # Get number of files to compare
    try:
        num_files = int(input("How many CSV files would you like to compare? "))
    except ValueError:
        print("Invalid input, defaulting to 1 file")
        num_files = 1
    
    file_paths = []
    
    for i in range(num_files):
        file_path = input(f"Enter path to CSV file #{i+1}: ").strip('"')
        
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"File not found: {file_path}")
    
    if not file_paths:
        print("No valid files provided")
        return
    
    # Ask for output name
    output_name = input("Enter name for output graph files (no extension): ").strip()
    if not output_name:
        output_name = f"comparison_{SESSION_ID}"
    
    # Create visualization output directory
    vis_dir = os.path.join(results_base_dir, output_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data from files
    all_data = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Loaded data from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No valid data loaded")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Choose chart type
    print("\nSelect chart type to generate:")
    print("1. Execution Time comparison (bar chart)")
    print("2. Speedup comparison (line chart)")
    print("3. SSIM comparison (Image quality: higher is better)")
    print("4. Throughput comparison (Pixels/second: higher is better)")
    print("5. PSNR comparison (Image quality in dB: higher is better)")
    print("6. MSE comparison (Error measure: lower is better)")
    
    chart_choice = input("Enter your choice (1-6): ")
    
    # Define file prefix for output
    chart_output = os.path.join(vis_dir, f"{output_name}")
    
    if chart_choice == '1':
        chart_path = generate_execution_time_chart(combined_df, output_name,vis_dir, output_name)
    elif chart_choice == '2':
        if "Speedup" not in combined_df.columns:
            print("Error: Speedup data not found in the CSV files.")
            return
        chart_path = generate_speedup_chart(combined_df, vis_dir, output_name)
    elif chart_choice == '3':
        if "SSIM" not in combined_df.columns:
            print("Error: SSIM data not found in the CSV files.")
            return
        chart_path = generate_quality_chart(combined_df, "SSIM", "Structural Similarity Index (higher is better)", vis_dir, output_name)
    elif chart_choice == '4':
        if "Throughput_pixels_per_sec" not in combined_df.columns:
            print("Error: Throughput data not found in the CSV files.")
            return
        chart_path = generate_quality_chart(combined_df, "Throughput_pixels_per_sec", "Throughput (pixels/second)", vis_dir, output_name)
    elif chart_choice == '5':
        if "PSNR" not in combined_df.columns:
            print("Error: PSNR data not found in the CSV files.")
            return
        chart_path = generate_quality_chart(combined_df, "PSNR", "Peak Signal-to-Noise Ratio (dB)", vis_dir, output_name)
    elif chart_choice == '6':
        if "MSE" not in combined_df.columns:
            print("Error: MSE data not found in the CSV files.")
            return
        chart_path = generate_quality_chart(combined_df, "MSE", "Mean Squared Error (lower is better)", vis_dir, output_name, lower_is_better=True)
    else:
        print("Invalid choice, no charts will be generated.")
        return
    
    # Try to open the chart file
    if chart_path:
        try:
            os.startfile(chart_path)
        except:
            try:
                # For macOS
                subprocess.call(['open', chart_path])
            except:
                try:
                    # For Linux
                    subprocess.call(['xdg-open', chart_path])
                except:
                    print(f"Could not automatically open the chart. Please find it at: {chart_path}")

def generate_speedup_chart(df, output_dir, filename_prefix):
    """Generate line chart for speedup comparison"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if DataFrame is not empty
    if df.empty:
        print("No data to visualize for speedup chart")
        return None
    
    # Check if Speedup column exists
    if "Speedup" not in df.columns:
        print("No speedup data available in the CSV")
        return None
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Get available resolutions from the data
    available_resolutions = df["Resolution"].unique()
    
    # Create a list of tuples with resolution and pixel count for sorting
    resolution_info = []
    for res in available_resolutions:
        try:
            if 'x' in res:
                w, h = map(int, res.split('x'))
                pixel_count = w * h
                resolution_info.append((res, pixel_count))
            else:
                resolution_info.append((res, float('inf')))
        except:
            resolution_info.append((res, float('inf')))
    
    # Sort resolutions by pixel count
    resolution_info.sort(key=lambda x: x[1])
    sorted_resolutions = [info[0] for info in resolution_info]
    
    # Display sorted resolutions for selection
    print("\nAvailable resolutions for visualization:")
    for i, res in enumerate(sorted_resolutions):
        print(f"{i+1}. {res}")
    
    res_selection = input("Enter resolution numbers to include (comma-separated or 'all'): ").strip()
    
    if res_selection.lower() == 'all':
        selected_resolutions = sorted_resolutions
    else:
        try:
            selection_indices = [int(idx.strip()) - 1 for idx in res_selection.split(',')]
            selected_resolutions = [sorted_resolutions[i] for i in selection_indices 
                                   if 0 <= i < len(sorted_resolutions)]
            
            if not selected_resolutions:
                print("No valid resolutions selected, using all")
                selected_resolutions = sorted_resolutions
                
        except (ValueError, IndexError):
            print("Invalid selection, using all resolutions")
            selected_resolutions = sorted_resolutions
    
    # Filter out sequential method as it always has speedup = 1.0
    # Also filter for selected resolutions
    filtered_df = df[(df["Method"] != "sequential") & 
                    (df["Resolution"].isin(selected_resolutions))]
    
    # If no data left after filtering sequential, show a message
    if filtered_df.empty:
        print("No speedup data available after filtering (only sequential data found)")
        return None
    
    # Create figure with adequate spacing
    plt.figure(figsize=(16, 10))
    
    # Get available methods
    methods = filtered_df["Method"].unique()
    
    # Define distinct colors and markers for different thread/process counts
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
    markers = ["o", "s", "^", "D"]  # Circle, Square, Triangle, Diamond
    
    # Create evenly spaced x positions for the resolutions
    x_positions = np.arange(len(selected_resolutions))
    
    # Create a mapping from resolution to x position
    resolution_to_xpos = {res: pos for pos, res in enumerate(selected_resolutions)}
    
    # Track which methods and counts have been plotted for legend
    plotted_methods = {}
    
    # Plot lines for each method and process/thread count
    color_idx = 0
    for method in methods:
        method_df = filtered_df[filtered_df["Method"] == method]
        
        # Group by process/thread count
        for count_idx, count in enumerate(sorted(method_df["Process_Thread_Count"].unique())):
            count_df = method_df[method_df["Process_Thread_Count"] == count]
            
            # Prepare data points for line chart
            x_values = []
            y_values = []
            res_labels = []
            
            for res in selected_resolutions:
                res_data = count_df[count_df["Resolution"] == res]
                
                if not res_data.empty:
                    x_pos = resolution_to_xpos[res]
                    x_values.append(x_pos)
                    y_values.append(res_data["Speedup"].mean())
                    res_labels.append(res)
            
            if x_values:
                # Sort points by x position
                points = sorted(zip(x_values, y_values, res_labels))
                x_sorted, y_sorted, labels_sorted = zip(*points)
                
                # Choose color and marker
                color = colors[color_idx % len(colors)]
                marker = markers[count_idx % len(markers)]
                
                # Make line thicker for higher process counts
                linewidth = 1.5 + (count / 32)
                
                # Plot the line
                label = f"{method} ({count})"
                plt.plot(x_sorted, y_sorted, marker=marker, linestyle='-', 
                         label=label, color=color, 
                         linewidth=linewidth, markersize=8, 
                         alpha=0.9)
                
                # Add data point labels
                for x, y in zip(x_sorted, y_sorted):
                    plt.annotate(f"{y:.2f}x", 
                                (x, y),
                                textcoords="offset points", 
                                xytext=(0, 7), 
                                ha='center',
                                fontsize=9,
                                fontweight='bold')
                
                # Add to plotted methods
                plotted_methods[label] = (color, marker)
            
            # Move to next color for next method/count combination
            color_idx += 1
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, selected_resolutions, rotation=45)
    
    # Set labels and title
    plt.title("Speedup Comparison", fontsize=16)
    plt.xlabel("Resolution", fontsize=14)
    plt.ylabel("Speedup (compared to sequential)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line at y=1 (sequential baseline)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label="Sequential baseline")
    
    # Create custom legend
    legend = plt.legend(title="Method (thread/process/blocksize)", 
                      title_fontsize=12, 
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      fontsize=10)
    
    # Enhance legend look
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)
        handle.set_alpha(1.0)
    
    # Set y-axis to start at 0
    plt.ylim(bottom=0)
    
    # Add more padding to avoid clipping
    plt.subplots_adjust(right=0.8)
    
    # Save the chart with high quality
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{filename_prefix}_speedup.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Speedup chart saved to: {output_path}")
    return output_path


# Function to test with a new image
# Function to test with a new image
def test_new_image():
    """Test denoising with a new image, with options for metrics calculation"""
    # Create a unique session name for this test
    session_name = input("Enter a name for this test session: ")
    if not session_name:
        session_name = f"new_image_{SESSION_ID}"
    
    # Create session directory
    session_dir = os.path.join(results_base_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    print(f"Results will be saved to: {session_dir}")
    
    # Get analysis mode
    print("\nSelect analysis mode:")
    print("1. Denoise only (just performs denoising and records execution time)")
    print("2. Denoise with detailed analysis (calculates PSNR, SSIM, etc. - requires a clean reference image)")
    analysis_mode = input("Enter your choice (1-2): ")
    
    # Get noisy image path
    noisy_image_path = input("Enter the path to your noisy image: ").strip('"')  # Strip quotes if user drags file
    
    if not os.path.exists(noisy_image_path):
        print(f"Error: Image not found at {noisy_image_path}")
        return
    
    # For detailed analysis, get clean reference image
    clean_image_path = noisy_image_path  # Default to same image (for denoise only mode)
    if analysis_mode == "2":
        clean_image_path = input("Enter the path to your clean reference image: ").strip('"')
        if not os.path.exists(clean_image_path):
            print(f"Error: Clean reference image not found at {clean_image_path}")
            return
        print(f"Clean reference image: {clean_image_path}")
    
    # Select methods to test
    print("\nSelect method to test:")
    print("1. Sequential only")
    print("2. OpenMP")
    print("3. MPI")
    method_choice = input("Enter your choice (1-3): ")
    
    if method_choice == '1':
        methods = ["sequential"]
    elif method_choice == '2':
        methods = ["omp"]
    elif method_choice == '3':
        methods = ["mpi"]
    else:
        print("Invalid choice, defaulting to Sequential")
        methods = ["sequential"]
    
    # Load image to get size
    try:
        img = Image.open(noisy_image_path)
        w, h = img.size
        print(f"Image dimensions: {w}x{h}")
        
        # Find the closest resolution in our param_map
        res_str = f"{w}x{h}"
        if res_str not in param_map:
            print("Image resolution not in parameter map. Finding closest match...")
            closest_res = None
            min_diff = float('inf')
            
            for res in param_map.keys():
                res_w, res_h = map(int, res.split('x'))
                diff = abs(w*h - res_w*res_h)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_res = res
            
            print(f"Using parameters for {closest_res}")
            lam, tau, sigma, theta, max_iter = param_map[closest_res]
        else:
            lam, tau, sigma, theta, max_iter = param_map[res_str]
        
        # Allow custom parameters
        print(f"\nDefault parameters (based on image size):")
        print(f"lambda={lam}, tau={tau}, sigma={sigma}, theta={theta}, iterations={max_iter}")
        
        custom_params = input("Use custom parameters? (y/n): ").lower().startswith('y')
        
        if custom_params:
            try:
                lam = float(input(f"Enter lambda value [{lam}]: ") or lam)
                tau = float(input(f"Enter tau value [{tau}]: ") or tau)
                sigma = float(input(f"Enter sigma value [{sigma}]: ") or sigma)
                theta = float(input(f"Enter theta value [{theta}]: ") or theta)
                max_iter = int(input(f"Enter max iterations [{max_iter}]: ") or max_iter)
            except ValueError:
                print("Invalid input, using default parameters")
        
        # Run all methods with appropriate counts
        all_results = []
        
        # Generate file name from path
        file_name = os.path.splitext(os.path.basename(noisy_image_path))[0]
        
        # Create output directory with absolute path
        output_dir = os.path.join(session_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Always run sequential first
        print("\n=== Running Sequential ===")
        
        # Define output path with absolute path
        output_path = os.path.join(output_dir, f"sequential_{file_name}.jpg")
        print(f"Output will be saved to: {output_path}")
        
        seq_result = run_denoising(
            "sequential", noisy_image_path, clean_image_path, output_path,
            (lam, tau, sigma, theta, max_iter), 1
        )
        
        if seq_result:
            # Verify that output file was created
            if os.path.exists(output_path):
                print(f"✅ Output file successfully created at: {output_path}")
            else:
                print(f"❌ WARNING: Output file was not created at: {output_path}")
            
            seq_result["File"] = file_name
            seq_result["Width"] = w
            seq_result["Height"] = h
            seq_result["Resolution"] = f"{w}x{h}"
            seq_result["Analysis_Mode"] = "Full" if analysis_mode == "2" else "Basic"
            
            # Store baseline time for speedup calculations
            baseline_time = seq_result["Execution_Time_sec"]
            seq_result["Baseline_Time"] = baseline_time
            seq_result["Speedup"] = 1.0
            
            all_results.append(seq_result)
            print(f"Sequential completed in {baseline_time:.2f} seconds")
            print(f"Output saved to: {output_path}")
            
            # Display metrics only if in detailed analysis mode with a real reference image
            if analysis_mode == "2" and clean_image_path != noisy_image_path:
                if "PSNR" in seq_result:
                    print(f"PSNR: {seq_result['PSNR']:.2f} dB")
                if "SSIM" in seq_result and seq_result["SSIM"] is not None:
                    print(f"SSIM: {seq_result['SSIM']:.4f}")
                if "MSE" in seq_result:
                    print(f"MSE: {seq_result['MSE']:.2f}")
        else:
            print("Failed to run sequential benchmark.")
            return
        
        # Skip the rest if user only wanted sequential
        if method_choice != '1':
            # Run the selected method with different thread/process counts
            method = methods[0]  # Either "omp" or "mpi"
            
            print(f"\n=== Running {method.upper()} ===")
            
            # Set thread/process counts for this method
            if method == "omp":
                counts = omp_thread_counts
            elif method == "mpi":
                counts = mpi_process_counts
            else:
                counts = [1]
            
            # Run for each thread/process count
            for count in counts:
                print(f"\nRunning... using {method.upper()} method, {count} {'processes' if method == 'mpi' else 'threads'}:")
                
                # Define output path with absolute path 
                output_path = os.path.join(output_dir, f"{method}_{count}_{file_name}.jpg")
                print(f"Output will be saved to: {output_path}")
                
                result = run_denoising(
                    method, noisy_image_path, clean_image_path, output_path,
                    (lam, tau, sigma, theta, max_iter), count
                )
                
                if result:
                    # Verify that output file was created
                    if os.path.exists(output_path):
                        print(f"✅ Output file successfully created at: {output_path}")
                    else:
                        print(f"❌ WARNING: Output file was not created at: {output_path}")
                    
                    result["File"] = file_name
                    result["Width"] = w
                    result["Height"] = h
                    result["Resolution"] = f"{w}x{h}"
                    result["Analysis_Mode"] = "Full" if analysis_mode == "2" else "Basic"
                    
                    # Calculate speedup compared to sequential
                    result["Baseline_Time"] = baseline_time
                    result["Speedup"] = round(baseline_time / result["Execution_Time_sec"], 3)
                    
                    print(f"Time: {result['Execution_Time_sec']:.2f} seconds")
                    
                    # Display metrics only if in detailed analysis mode with a real reference image
                    if analysis_mode == "2" and clean_image_path != noisy_image_path:
                        if "PSNR" in result:
                            print(f"PSNR: {result['PSNR']:.2f} dB")
                        if "SSIM" in result and result["SSIM"] is not None:
                            print(f"SSIM: {result['SSIM']:.4f}")
                        if "MSE" in result:
                            print(f"MSE: {result['MSE']:.2f}")
                    
                    print(f"Performance gain compared with sequential: {result['Speedup']:.2f}x")
                    
                    all_results.append(result)
                    print(f"Output saved to: {output_path}")
        
        # Save all results to CSV
        if all_results:
            df = organize_dataframe(pd.DataFrame(all_results))
            csv_path = os.path.join(session_dir, f"{session_name}_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            
            # Generate charts
            chart_dir = os.path.join(session_dir, "charts")
            
            # Only generate quality charts if we have proper metrics
            if analysis_mode == "2" and clean_image_path != noisy_image_path:
                generate_comparison_charts(df, chart_dir, prefix=f"{file_name}_")
                print(f"Charts saved to: {chart_dir}")
            else:
                print("No quality metrics charts generated (denoise-only mode)")
        else:
            print("No results to save.")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        
# Function to test a single configuration
def test_single_configuration():
    """Test a single configuration from predefined parameters"""
    # Create a unique session name for this test
    session_name = input("Enter a name for this test session: ")
    if not session_name:
        session_name = f"single_test_{SESSION_ID}"
    
    # Create session directory
    session_dir = os.path.join(results_base_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    print(f"Results will be saved to: {session_dir}")
    
    # Ask user if they want to use baseline CSV data for comparison
    print("\nWould you like to use baseline data from a CSV file for speedup comparison?")
    print("Enter CSV file path or '-' to skip")
    baseline_csv = input("CSV path: ").strip('"')
    
    # Load baseline data if provided
    if baseline_csv != "-":
        sequential_times = load_sequential_times(baseline_csv)
    else:
        sequential_times = {}
    
    # Select method
    print("\nSelect denoising method:")
    print("1. Sequential")
    print("2. OpenMP")
    print("3. MPI")
    method_choice = input("Enter your choice (1-3): ")
    
    if method_choice == '1':
        method = "sequential"
        counts = [1]
    elif method_choice == '2':
        method = "omp"
        counts = omp_thread_counts
    elif method_choice == '3':
        method = "mpi"
        counts = mpi_process_counts
    else:
        print("Invalid choice, defaulting to sequential")
        method = "sequential"
        counts = [1]
    
    # Select resolution
    print("\nAvailable resolutions:")
    for i, (w, h) in enumerate(resolutions):
        print(f"{i+1}. {w}x{h}")
    res_choice = int(input("Enter resolution number: ").strip() or "1") - 1
    
    if 0 <= res_choice < len(resolutions):
        w, h = resolutions[res_choice]
        res_str = f"{w}x{h}"
    else:
        print("Invalid choice, defaulting to 512x512")
        w, h = 512, 512
        res_str = "512x512"
    
    # Select image index
    img_idx = int(input("Enter image index (0-9): ").strip() or "0")
    
    # Select thread/process count
    if method != "sequential":
        print(f"\n{method.capitalize()} {'process' if method == 'mpi' else 'thread'} counts:")
        for i, count in enumerate(counts):
            print(f"{i+1}. {count}")
        count_choice = int(input(f"Enter {method} {'process' if method == 'mpi' else 'thread'} count number: ").strip() or "1") - 1
        
        if 0 <= count_choice < len(counts):
            count = counts[count_choice]
        else:
            print("Invalid choice, defaulting to 1")
            count = 1
    else:
        count = 1
    
    # Get parameters for this resolution
    if res_str in param_map:
        lam, tau, sigma, theta, max_iter = param_map[res_str]
    else:
        print(f"No parameters defined for {res_str}, using defaults")
        lam, tau, sigma, theta, max_iter = 1.0, 0.05, 0.05, 1.0, 500
    
    # Allow custom parameters
    print(f"\nDefault parameters for {res_str}:")
    print(f"lambda={lam}, tau={tau}, sigma={sigma}, theta={theta}, iterations={max_iter}")
    
    custom_params = input("Use custom parameters? (y/n): ").lower().startswith('y')
    
    if custom_params:
        try:
            lam = float(input(f"Enter lambda value [{lam}]: ") or lam)
            tau = float(input(f"Enter tau value [{tau}]: ") or tau)
            sigma = float(input(f"Enter sigma value [{sigma}]: ") or sigma)
            theta = float(input(f"Enter theta value [{theta}]: ") or theta)
            max_iter = int(input(f"Enter max iterations [{max_iter}]: ") or max_iter)
        except ValueError:
            print("Invalid input, using default parameters")
    
    # Get input and clean paths
    input_path = os.path.abspath(f"input_images/image_{img_idx}_{res_str}.jpg")
    clean_path = os.path.abspath(f"clean_images/image_{img_idx}_{res_str}.jpg")
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Check if clean path exists - but ALWAYS use it for metrics if it exists
    if not os.path.exists(clean_path):
        print(f"Warning: Clean image not found at {clean_path}, using input as reference")
        clean_path = input_path  # Only use input as fallback if clean doesn't exist
    else:
        print(f"Using clean reference image for metrics: {clean_path}")
    
    # Create output directory and use absolute path
    output_dir = os.path.abspath("output_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path with absolute path
    output_path = get_output_filename(method, img_idx, count, res_str, output_dir)
    
    print(f"\nRunning test with:")
    print(f"- Method: {method}")
    print(f"- Resolution: {res_str}")
    print(f"- {'Processes' if method == 'mpi' else 'Threads'}: {count}")
    print(f"- Image: {img_idx}")
    print(f"- Parameters: lambda={lam}, tau={tau}, sigma={sigma}, theta={theta}, iter={max_iter}")
    print(f"- Output will be saved to: {output_path}")
    
    # Run the test - ALWAYS use clean path for metrics (only use input as clean if clean doesn't exist)
    result = run_denoising(
        method, input_path, clean_path, output_path,
        (lam, tau, sigma, theta, max_iter), count
    )
    
    if not result:
        print("Test failed.")
        return
    
    # Verify that output file was created
    if os.path.exists(output_path):
        print(f"✅ Output file successfully created at: {output_path}")
    else:
        print(f"❌ WARNING: Output file was not created at: {output_path}")
    
    # Add additional information
    result["Image_ID"] = img_idx
    result["Resolution"] = res_str
    result["Width"] = w
    result["Height"] = h
    
    # Calculate additional metrics
    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    num_pixels = w * h
    exec_time = result["Execution_Time_sec"]
    time_per_mp = exec_time / (num_pixels / 1e6)
    throughput = num_pixels / exec_time
    
    result["Size_MB"] = round(size_mb, 5)
    result["Time_per_Megapixel"] = round(time_per_mp, 5)
    result["Throughput_pixels_per_sec"] = round(throughput, 2)
    
    # Compare with baseline data if available
    key = (img_idx, res_str)
    if key in sequential_times:
        seq_time = sequential_times[key]
        speedup = seq_time / exec_time
        result["Speedup"] = round(speedup, 3)
        print(f"\nBaseline comparison:")
        print(f"Sequential time from CSV: {seq_time:.2f} seconds")
        print(f"Speedup vs. Sequential time: {speedup:.2f}x")
    
    # Add speedup for sequential method too
    if method == "sequential":
        result["Speedup"] = 1.0
    
    # Save result
    df = organize_dataframe(pd.DataFrame([result]))
    csv_path = os.path.join(session_dir, f"{method}_single_test.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults:")
    print(f"Total execution time: {exec_time:.2f} seconds")
    
    # Always show quality metrics if available
    if "PSNR" in result:
        print(f"PSNR: {result['PSNR']:.2f} dB")
    if "SSIM" in result and result["SSIM"] is not None:
        print(f"SSIM: {result['SSIM']:.4f}")
    if "MSE" in result:
        print(f"MSE: {result['MSE']:.2f}")
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Results saved to: {csv_path}")

# Update the run_full_benchmark function to use horizontal bar charts
def run_full_benchmark():
    """Run full benchmark on all resolutions"""
    # Create a unique session name for this benchmark
    session_name = input("Enter a name for this benchmark session: ")
    if not session_name:
        session_name = f"benchmark_{SESSION_ID}"
    
    # Create session directory
    session_dir = os.path.join(results_base_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    print(f"Results will be saved to: {session_dir}")
    
    output_name = session_name

    # Ask user if they want to use baseline CSV data for comparison
    print("\nWould you like to use baseline data from a CSV file for speedup comparison?")
    print("Enter CSV file path or '-' to skip")
    baseline_csv = input("CSV path: ").strip('"')
    
    # Load baseline data if provided
    if baseline_csv != "-":
        sequential_times = load_sequential_times(baseline_csv)
    else:
        sequential_times = {}
    
    # Select methods to benchmark
    print("\nSelect methods to benchmark:")
    print("1. Sequential only")
    print("2. OpenMP only")
    print("3. MPI only")
    print("4. All methods (Sequential, OpenMP, MPI)")
    print("5. CUDA Visualization (Load existing CUDA results - unable to run here, only can run in colab)")
    method_choice = input("Enter your choice (1-5): ")
    
    if method_choice == '1':
        methods = ["sequential"]
    elif method_choice == '2':
        methods = ["omp"]
    elif method_choice == '3':
        methods = ["mpi"]
    elif method_choice == '4':
        methods = ["sequential", "omp", "mpi"]
    elif method_choice == '5':
        generate_cuda_horizontal_bar_chart(session_name)
        return
    else:
        print("Invalid choice, defaulting to all methods")
        methods = ["sequential", "omp", "mpi"]
    
    # Select image indices (now supports comma-separated)
    img_indices_input = input("Enter image indices to test (comma-separated, 0-9): ").strip() or "0"
    
    # Parse comma-separated indices
    try:
        img_indices = [int(idx.strip()) for idx in img_indices_input.split(',')]
    except ValueError:
        print("Invalid image indices, defaulting to image 0")
        img_indices = [0]
    
    # Select resolutions
    print("\nSelect resolutions to test:")
    for i, (w, h) in enumerate(resolutions):
        print(f"{i+1}. {w}x{h}")
    res_choices = input("Enter resolution numbers (comma-separated, or 'all'): ")
    
    if res_choices.lower() == 'all':
        selected_resolutions = resolutions
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in res_choices.split(',')]
            selected_resolutions = [resolutions[i] for i in indices if 0 <= i < len(resolutions)]
        except:
            print("Invalid selection, using all resolutions")
            selected_resolutions = resolutions
    
    # Run benchmarks
    all_results = []
    
    # Always run sequential first (if selected) to establish baseline
    if "sequential" in methods:
        print("\n=== Running Sequential Benchmarks (Baseline) ===")
        seq_results = run_method_benchmarks("sequential", img_indices, selected_resolutions, session_dir, sequential_times)
        all_results.extend(seq_results)
    
    # Then run other methods
    for method in methods:
        if method == "sequential" and "sequential" in methods:
            continue  # Already ran sequential
        
        print(f"\n=== Running {method.upper()} Benchmarks ===")
        method_results = run_method_benchmarks(method, img_indices, selected_resolutions, session_dir, sequential_times)
        all_results.extend(method_results)
    
    # Save all results
    if all_results:
        df = organize_dataframe(pd.DataFrame(all_results))
        csv_path = os.path.join(session_dir, f"{session_name}_complete.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nComplete results saved to: {csv_path}")
        
        # Generate horizontal bar charts for better within-method comparison
        chart_dir = os.path.join(session_dir, "charts")
        generate_horizontal_bar_chart(df, output_name, chart_dir, session_name)
    else:
        print("No benchmark results to save.")

# Update the main menu function to use the new implementations
def main_menu():
    """Display main menu and handle user choices"""
    while True:
        print("\n==== Image Denoising Benchmark Tool ====")
        print("1. Run full benchmark (all resolutions)")
        print("2. Test a single configuration")
        print("3. Test with a new image")
        print("4. Compare results from existing CSV files")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            run_full_benchmark()
        elif choice == '2':
            test_single_configuration()
        elif choice == '3':
            test_new_image()
        elif choice == '4':
            visualize_existing_results()
        elif choice == '5':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Entry point
if __name__ == "__main__":
    main_menu()