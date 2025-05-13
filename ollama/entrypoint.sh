#!/bin/sh

# Start Ollama server in background
ollama serve &

# Wait until the server is ready (responding to HTTP)
until curl -sf http://localhost:11434/ | grep -q "Ollama"; do
  echo "⏳ Waiting for Ollama to be ready..."
  sleep 2
done

# Pull the primary model
echo "⬇️ Pulling OLLAMA_MODEL: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

# Pull the embeddings model
echo "⬇️ Pulling EMBEDDINGS_MODEL: $EMBEDDINGS_MODEL"
ollama pull "$EMBEDDINGS_MODEL"

# Wait for background process
wait
