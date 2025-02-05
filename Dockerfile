# Use the official TensorFlow image
FROM tensorflow/tensorflow

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY pip_requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && \
    pip install -r pip_requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entrypoint to run the main script
ENTRYPOINT ["python", "main.py"]