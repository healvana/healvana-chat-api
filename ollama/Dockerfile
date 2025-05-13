# Base image for Ollama
FROM ollama/ollama

# Install curl (and optionally bash if needed)
RUN apt-get update && apt-get install -y curl

# Set working directory
WORKDIR /app

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Make it executable
RUN chmod +x /app/entrypoint.sh

# Run it as the default command
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
