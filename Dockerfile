# Use the official NVIDIA CUDA image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables for NVIDIA and CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CUDA_HOME /usr/local/cuda
# Set environment variables for non-interactive apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install basic utilities
RUN apt-get update && apt-get install -y \
    tzdata \ 
    git \
    ssh \
    curl \
    wget \
    build-essential \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure timezone for tzdata
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata


# Install Python and pip
RUN apt-get update && apt-get install -y python3.9-venv python3.9-distutils && apt-get clean
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip



# Install Hatch for Python environment management
RUN pip install --no-cache-dir hatch

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Expose ports (optional)
EXPOSE 5000
EXPOSE 8888

# Default command (modify if necessary)
CMD ["hatch", "run", "python", "main.py"]
