FROM python:3.11-slim
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Digital Ocean App Platform automatically sets PORT
# We'll use this as fallback for local development
ENV PORT=${PORT:-3000}

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port - using variable
EXPOSE $PORT

# Start the application with dynamic port binding
CMD uvicorn main:app --host 0.0.0.0 --port $PORT