FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-pyqt5 \
    postgresql-client \
    iputils-ping \
    net-tools \
    cmake \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY docker/face_recognition/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV CAMERA_ID=0
ENV CONFIDENCE_THRESHOLD=0.6
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=security_system
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Run the application
CMD ["/entrypoint.sh"] 