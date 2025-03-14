FROM python:3.12-slim

WORKDIR /app

# Copy requirements and packages files
COPY requirements.txt /app/
COPY packages.txt /app/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends $(cat packages.txt) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Add any other environment variables your app might need

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]