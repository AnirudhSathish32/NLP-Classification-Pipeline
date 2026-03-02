# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (caches installs if unchanged)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

EXPOSE 8000

# Default command to run the pipeline
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
