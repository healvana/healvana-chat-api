# Use a lightweight Python image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the poetry.lock and pyproject.toml
COPY pyproject.toml poetry.lock /app/

RUN poetry lock

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the app
COPY . /app

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "python", "serve.py"]