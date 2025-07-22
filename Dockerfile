# Use official Python 3.10 image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy everything from current folder to /app in container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

# Run your training script
CMD ["python","serve_model.py"]
