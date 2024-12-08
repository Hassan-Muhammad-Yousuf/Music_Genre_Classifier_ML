# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Cython for potential C extensions
RUN pip install --no-cache-dir Cython

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
