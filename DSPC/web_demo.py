import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import subprocess
import time
import tempfile
import shutil

# Set page config
st.set_page_config(
    page_title="Image Denoising Demo",
    page_icon="🖼️",
    layout="wide"
)

# Title
st.title("🖼️ Parallel Image Denoising Demo")
st.markdown("Compare sequential, OpenMP, and MPI denoising algorithms in real-time!")

# Sidebar for parameters
st.sidebar.header("Parameters")

# Method selection
method = st.sidebar.selectbox(
    "Denoising Method",
    ["sequential", "omp", "mpi"],
    help="Choose the parallelization method"
)

# Thread/Process count
if method == "sequential":
    count = 1
else:
    max_count = 16 if method == "mpi" else 16
    count = st.sidebar.slider(
        f"Number of {'processes' if method == 'mpi' else 'threads'}",
        min_value=1,
        max_value=max_count,
        value=4,
        step=1
    )

# Algorithm parameters
st.sidebar.subheader("Algorithm Parameters")
lam = st.sidebar.slider("Lambda", 0.01, 10.0, 2.0, 0.1)
tau = st.sidebar.slider("Tau", 0.001, 1.0, 0.05, 0.001)
sigma = st.sidebar.slider("Sigma", 0.001, 1.0, 0.05, 0.001)
theta = st.sidebar.slider("Theta", 0.1, 2.0, 1.0, 0.1)
max_iter = st.sidebar.slider("Max Iterations", 10, 1000, 300, 10)

# File upload
st.header("Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose a noisy image to denoise",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image that needs denoising"
)

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, use_column_width=True)

        # Get image dimensions
        width, height = original_image.size
        st.write(f"Dimensions: {width} x {height}")

    # Process button
    if st.button("🚀 Start Denoising", type="primary"):
        with st.spinner("Processing image... This may take a few moments."):

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded image
                input_path = os.path.join(temp_dir, "input.jpg")
                original_image.save(input_path)

                # Output path
                output_path = os.path.join(temp_dir, "output.jpg")

                # Build command
                if method == "mpi":
                    cmd = [
                        r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe",
                        "-n", str(count),
                        "x64/Release/DSPC.exe",
                        input_path,
                        output_path,
                        str(lam), str(tau), str(sigma), str(theta), str(max_iter)
                    ]
                else:
                    # For sequential and OpenMP
                    env = os.environ.copy()
                    if method == "omp":
                        env["OMP_NUM_THREADS"] = str(count)

                    cmd = [
                        "x64/Release/DSPC.exe",
                        input_path,
                        output_path,
                        str(lam), str(tau), str(sigma), str(theta), str(max_iter)
                    ]

                # Run the denoising
                start_time = time.time()
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd(),
                        env=env if method != "mpi" else None,
                        timeout=300  # 5 minute timeout
                    )

                    execution_time = time.time() - start_time

                    if result.returncode == 0:
                        # Success
                        st.success(f"✅ Denoising completed in {execution_time:.2f} seconds!")

                        # Display result
                        with col2:
                            st.subheader("Denoised Image")
                            if os.path.exists(output_path):
                                denoised_image = Image.open(output_path)
                                st.image(denoised_image, use_column_width=True)
                            else:
                                st.error("Output image not found")

                        # Show metrics if available
                        st.subheader("Performance Metrics")
                        col3, col4, col5 = st.columns(3)

                        with col3:
                            st.metric("Execution Time", f"{execution_time:.2f}s")

                        with col4:
                            st.metric("Method", method.upper())

                        with col5:
                            st.metric("Threads/Processes", count)

                        # Show command output if available
                        if result.stdout:
                            with st.expander("Command Output"):
                                st.code(result.stdout)

                    else:
                        st.error("❌ Denoising failed!")
                        st.error(f"Error: {result.stderr}")

                except subprocess.TimeoutExpired:
                    st.error("⏰ Processing timed out after 5 minutes")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Benchmark section
st.header("📊 Performance Benchmark")
st.markdown("Compare different methods and configurations")

if st.button("Run Benchmark Comparison"):
    with st.spinner("Running benchmarks... This will take several minutes."):

        # This would call your existing benchmark functions
        # For demo purposes, we'll show sample results
        st.info("Benchmark functionality would integrate with your existing evaluation_script.py")

        # Sample benchmark data
        benchmark_data = {
            "Method": ["Sequential", "OpenMP (4 threads)", "MPI (4 processes)"],
            "Time (s)": [45.2, 12.8, 11.5],
            "Speedup": [1.0, 3.53, 3.93]
        }

        df = pd.DataFrame(benchmark_data)

        # Display results
        st.dataframe(df)

        # Create comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df["Method"], df["Time (s)"])
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("Performance Comparison")
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom')

        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Parallel Image Processing Demo")
st.markdown("*Note: This demo requires the compiled DSPC.exe executable to be present.*")