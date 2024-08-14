# Use the official NVIDIA CUDA 11.3 development image with Ubuntu 22.04
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y nano mlocate less git cmake-curses-gui cmake-gui jq wget unzip debconf-utils

RUN git clone https://github.com/trexob1t/aegnn.git /app/aegnn

# Install Miniconda
RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    chmod -R 755 /miniconda3 && \
    rm -rf /miniconda3/miniconda.sh

ENV PATH="/miniconda3/bin:$PATH"

# Create the conda environment using the environment.yml file
RUN conda init bash && \
    bash ~/.bashrc && \
    . ~/.bashrc && \
    conda env create -f /app/aegnn/environment.yml -y && \
    conda activate aegnn_train && \
    pip install -e /app/aegnn/

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the default command
# CMD ["conda", "run", "--no-capture-output", "-n", "aegnn_train", "python", "your_script.py"]
