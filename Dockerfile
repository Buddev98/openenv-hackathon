FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860
ENV PYTHONPATH="/app/src"

# Set working directory
WORKDIR /app

# Install python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Command to run the application using the wrapper
CMD ["python", "server/app.py"]
