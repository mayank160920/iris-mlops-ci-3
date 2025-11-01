# Use official python image as base image
FROM python:3.10-slim
# Set working directory in the container
WORKDIR /app
# Copy the files to the container
COPY . /app
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port the app runs on
EXPOSE 8200
# Command to run the application
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]