FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    postgresql-client \
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
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .
COPY docker/migrations/scripts/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=security_system
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Run migrations with wait-for-it script
CMD ["/wait-for-it.sh", "postgres", "5432", "alembic", "upgrade", "head"] 