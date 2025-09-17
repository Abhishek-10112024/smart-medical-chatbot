# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set PYTHONPATH so Python can find 'app' package
ENV PYTHONPATH=/app

# Install system dependencies (for PyTorch, llama-cpp, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
