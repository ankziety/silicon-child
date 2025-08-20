FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Playwright browsers are already installed in the base image

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/screenshots

# Set environment variables
ENV PYTHONPATH=/app
ENV DISPLAY=:99

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "-m", "ai_infant", "--duration", "10"]
