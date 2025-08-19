FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    procps \
    libxss1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

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
