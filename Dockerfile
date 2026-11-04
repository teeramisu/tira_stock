# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY run.sh .

# Make run script executable
RUN chmod +x run.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["./run.sh"]