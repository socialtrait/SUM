# Use PyTorch's official Docker image with CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

COPY requirements.txt /app/

# Install Python dependencies (Assumes a requirements.txt file exists)
RUN pip install -r requirements.txt
# Install any additional dependencies or libraries for SUM
# Add commands here if SUM requires extra installation steps

# Copy the repository into the container
COPY . /app

# Expose necessary ports (adjust as needed based on the SUM project)
EXPOSE 8000 5000 7860

# run python /app/gradio_app.py
CMD ["python", "fastapi_app.py"]
