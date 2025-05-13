# Healvana Global Chat API with RAG

**Version:** -0.1.0

## Description

This project provides a production-ready FastAPI application serving a conversational AI with the persona of "Healvana," a professional psychiatrist and mental health specialist. It integrates with a local Ollama instance to run Large Language Models (LLMs), supports Retrieval Augmented Generation (RAG) for context-aware responses, and offers localized personas for a global audience. The API features streaming responses for a real-time chat experience.

Key features include:
- **Localized Healvana Personas**: The AI is primed with detailed system prompts loaded from external files, allowing for different "Healvana" personas tailored to specific locales (e.g., `en-US`, `en-GB`). This includes localized initial greetings and crisis hotline information.
- **Retrieval Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a local document store, providing more contextually rich and informed interactions. Uses Ollama embeddings and an in-memory FAISS vector store.
- **Session Management**: In-memory conversation history per session ID, with persona and greeting determined by the selected locale.
- **Streaming Chat**: Real-time token-by-token responses via Server-Sent Events (SSE).
- **Configuration**: Ollama model, embeddings model, connection parameters, and default locale are configurable via environment variables.
- **Comprehensive API**: Endpoints for chat interaction, session history management, and retrieval of persona configurations.
- **Dockerized**: Designed to be run easily using Docker and Docker Compose.

## Prerequisites

- **Docker and Docker Compose**: For running the application and its dependencies (Ollama).
- **Ollama**: Installed and running.
    - The primary LLM model (e.g., `llama3`) must be pulled: `ollama pull llama3`
    - The embeddings model (`nomic-embed-text` by default) must be pulled: `ollama pull nomic-embed-text`
- **Python 3.9+** (if running outside Docker for development).
- **Poetry** (if running outside Docker for Python dependency management).

## Project Structure 

Your main application code (`serve.py`) resides in the `langchain` directory. The API expects the following subdirectories within the `langchain` directory (which is mounted to `/app` in Docker):
```bash

├── README.md
├── docker-compose.yml
├── kubernetes-manifests.yaml
├── langchain
│   ├── Dockerfile
│   ├── documents
│   ├── personas
│   │   └── en-US
│   │       ├── greeting.txt
│   │       ├── hotlines.json
│   │       └── system.prompt
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── serve.py
├── ollama
│   ├── Dockerfile
│   └── entrypoint.sh
└── web
    ├── Dockerfile
    ├── package-lock.json
    ├── package.json
    ├── src
    │   ├── app.d.ts
    │   ├── app.html
    │   └── routes
    │       ├── +layout.ts
    │       └── +page.svelte
    ├── static
    │   └── favicon.png
    ├── svelte.config.js
    ├── tsconfig.json
    └── vite.config.ts
```

## Setup and Running the API

The primary method for running this API is using Docker Compose.

1.  **Clone the Repository (if applicable)**
    ```bash
    # git clone git@github.com:healvana/healvana-chat-api.git
    # cd healvana-chat-api
    ```

2.  **Prepare Persona and RAG Document Files**:
    * Inside the `langchain` directory (or the directory mapped to `/app` in your Docker container), create the `personas` directory.
    * For each locale you want to support (e.g., `en-US`, `en-GB`), create a subdirectory within `personas`.
    * Inside each locale subdirectory, add:
        * `system.prompt`: A plain text file with the detailed system prompt for that locale.
        * `greeting.txt`: A plain text file with the initial AI greeting for that locale.
        * `hotlines.json`: A JSON file listing crisis hotlines for that locale (see example format in the "API Endpoints" section or Python code).
    * Create the `documents` directory inside `langchain`.
    * Add `.txt` files containing information you want the RAG system to use (e.g., FAQs, articles on coping strategies).

3.  **Configure Environment Variables**:
    Create a `.env` file in the root directory of your project (or configure these directly in your `docker-compose.yml` or deployment environment):

    ```env
    # .env file
    OLLAMA_MODEL=llama3 # Replace with your desired Ollama model (e.g., llama3:8b-instruct)
    EMBEDDINGS_MODEL=nomic-embed-text # Ollama model for text embeddings
    # OLLAMA_HOST=ollama    # Usually set in docker-compose.yml for langchain service
    # OLLAMA_PORT=11434   # Usually set in docker-compose.yml for langchain service
    OLLAMA_TEMPERATURE=0.2  # LLM generation temperature
    # DEFAULT_LOCALE=en-US # Optional: Overrides the default locale in the code
    LOG_LEVEL=INFO
    ```

4.  **Build and Run with Docker Compose**:
    From the root directory of the project (where `docker-compose.yml` is located):
    ```bash
    docker-compose up --build
    ```
    To run in detached mode:
    ```bash
    docker-compose up --build -d
    ```
    The API will typically be available at `http://localhost:8000`.

## Environment Variables

The `langchain` service uses the following environment variables:

-   `OLLAMA_MODEL` (Required): The name of the Ollama model for chat generation (e.g., `llama3`, `qwen2:7b`).
-   `EMBEDDINGS_MODEL` (Optional, Default: `nomic-embed-text`): The name of the Ollama model used for generating text embeddings for RAG.
-   `OLLAMA_HOST` (Optional, Default: `ollama` in code, typically overridden by Docker Compose): Hostname for the Ollama service.
-   `OLLAMA_PORT` (Optional, Default: `11434` in code, typically overridden by Docker Compose): Port for the Ollama service.
-   `OLLAMA_BASE_URL` (Optional): Full base URL for Ollama. If set, overrides `OLLAMA_HOST` and `OLLAMA_PORT`.
-   `OLLAMA_TEMPERATURE` (Optional, Default: `0.2`): Temperature for LLM generation.
-   `DEFAULT_LOCALE` (Optional, Default: `en-US`): The default locale to use if a client request doesn't specify one.
-   `LOG_LEVEL` (Optional, Default: `INFO`): Logging level for the application.

## API Endpoints

API documentation (Swagger UI) is available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

---

### Management Endpoints

#### 1. Health Check

-   **Endpoint**: `GET /health`
-   **Description**: Checks API health, LLM initialization, RAG status, and loaded persona configurations.
-   **`curl` Example**:
    ```bash
    curl -X GET http://localhost:8000/health
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "status": "ok", // or "degraded" if LLM/RAG components failed
      "model_name": "llama3",
      "embeddings_model": "nomic-embed-text",
      "ollama_url_used": "http://ollama:11434",
      "ollama_temperature": 0.2,
      "default_locale": "en-US",
      "available_personas": ["en-US", "en-GB"],
      "rag_status": "Core RAG Initialized (VectorStore & Retriever)"
    }
    ```

#### 2. List Available Personas

-   **Endpoint**: `GET /config/personas`
-   **Description**: Lists all successfully loaded persona configurations (locale IDs and names).
-   **`curl` Example**:
    ```bash
    curl -X GET http://localhost:8000/config/personas
    ```
-   **Example Success Response (200 OK)**:
    ```json
    [
      {
        "locale_id": "en-US",
        "persona_name": "Healvana (en-US)"
      },
      {
        "locale_id": "en-GB",
        "persona_name": "Healvana (en-GB)"
      }
    ]
    ```

#### 3. Get Specific Persona Details

-   **Endpoint**: `GET /config/personas/{locale_id}`
-   **Description**: Retrieves the detailed configuration for a specific persona locale, including system prompt, greeting, and crisis hotlines.
-   **`curl` Example** (for `en-US`):
    ```bash
    curl -X GET http://localhost:8000/config/personas/en-US
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "locale_id": "en-US",
      "persona_name": "Healvana (en-US)",
      "system_prompt": "You are Healvana, a US-based professional psychiatrist...",
      "initial_greeting": "Hello, I'm Healvana, your mental wellness companion for the US...",
      "crisis_hotlines": [
        {
          "name": "988 Suicide & Crisis Lifeline (USA)",
          "contact": "Call or text 988",
          "description": "Provides 24/7, free and confidential support..."
        }
      ]
    }
    ```

---

### Chat Interaction Endpoints

#### 1. Standard Chat (Non-Streaming, RAG-enabled)

-   **Endpoint**: `POST /chat`
-   **Description**: Sends a message to the AI. The AI uses RAG to retrieve relevant context and generates a complete response based on the selected locale's persona.
-   **Request Body**:
    ```json
    {
      "session_id": "your-unique-session-id-123",
      "message": "I've been feeling overwhelmed by work lately.",
      "locale": "en-US" // Optional: Defaults to server's default_locale
    }
    ```
-   **`curl` Example**:
    ```bash
    curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
          "session_id": "rag-session-001",
          "message": "How can I manage work stress?",
          "locale": "en-US"
        }'
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "rag-session-001",
      "locale_used": "en-US",
      "response": "Work stress is common. Can you share more?" // Example concise response
    }
    ```

#### 2. Streaming Chat (RAG-enabled)

-   **Endpoint**: `POST /chat/stream`
-   **Description**: Sends a message and receives a RAG-enhanced, streamed response (Server-Sent Events) based on the selected locale's persona.
-   **Request Body**:
    ```json
    {
      "session_id": "your-unique-session-id-456",
      "message": "What are some grounding techniques?",
      "locale": "en-US" // Optional
    }
    ```
-   **`curl` Example**:
    ```bash
    curl -X POST http://localhost:8000/chat/stream \
    -H "Content-Type: application/json" \
    -N \
    -d '{
          "session_id": "rag-stream-002",
          "message": "Tell me about mindfulness.",
          "locale": "en-US"
        }'
    ```
-   **Example Streamed Response (text/event-stream)**:
    ```
    data: {"token": "Mindfulness", "locale_used": "en-US"}

    data: {"token": " helps focus.", "locale_used": "en-US"}

    data: {"token": " Interested?", "locale_used": "en-US"}

    data: {"end_stream": true, "session_id": "rag-stream-002", "locale_used": "en-US"}
    ```

---

### Session Management Endpoints

#### 1. Get Session History

-   **Endpoint**: `GET /chat/sessions/{session_id}/history`
-   **Description**: Retrieves the conversation history for a specific session. The history reflects the locale used when the session was initiated.
-   **`curl` Example** (replace `your-session-id`):
    ```bash
    curl -X GET http://localhost:8000/chat/sessions/rag-stream-002/history
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "rag-stream-002",
      "locale_used": "en-US",
      "history": [
        {
          "type": "system",
          "content": "You are Healvana, a US-based professional psychiatrist..."
        },
        {
          "type": "ai",
          "content": "Hello, I'm Healvana, your mental wellness companion for the US..."
        },
        {
          "type": "human",
          "content": "Tell me about mindfulness."
        },
        {
          "type": "ai",
          "content": "Mindfulness helps focus. Interested?"
        }
      ]
    }
    ```

#### 2. Clear Session History

-   **Endpoint**: `DELETE /chat/sessions/{session_id}/history`
-   **Description**: Clears the conversation history for a specific session. The session will be re-initialized with the appropriate persona and greeting on its next use.
-   **`curl` Example** (replace `your-session-id`):
    ```bash
    curl -X DELETE http://localhost:8000/chat/sessions/rag-stream-002/history
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "rag-stream-002",
      "message": "Session history cleared."
    }
    ```

#### 3. Get Initial AI Greeting for a Session

-   **Endpoint**: `GET /chat/sessions/{session_id}/greeting`
-   **Query Parameter**: `locale` (Optional, e.g., `?locale=en-GB`) - Specifies the locale for which to get the greeting. Defaults to the server's default locale if not provided.
-   **Description**: Gets the initial greeting for the specified (or default) Healvana persona. If the session ID is new, this also initializes the session for that locale.
-   **`curl` Example** (for `en-GB` locale):
    ```bash
    curl -X GET "http://localhost:8000/chat/sessions/new-gb-session/greeting?locale=en-GB"
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "new-gb-session",
      "locale_used": "en-GB",
      "greeting": "Hello, I'm Healvana, your mental wellness companion for the UK..."
    }
    ```

## RAG Details

-   **Embeddings**: Uses `OllamaEmbeddings` with the model specified by the `EMBEDDINGS_MODEL` environment variable (defaults to `nomic-embed-text`).
-   **Document Loading**: Loads `.txt` files from the `./documents` directory (relative to `serve.py`).
-   **Vector Store**: Uses FAISS (in-memory). The vector store is built at application startup.
-   **Retriever**: Retrieves the top K (default 3) most relevant document chunks.
-   **Chain**: A history-aware RAG chain is used. It first reformulates the user's question based on chat history to be a standalone question, then retrieves relevant documents, and finally generates an answer using the retrieved context, chat history, and the locale-specific Healvana system prompt.

## Notes on Production

-   **CORS**: Restrict `allow_origins` in `CORSMiddleware` to your specific frontend domain(s).
-   **Session & RAG Memory**:
    -   Chat session history (`session_histories`) is currently stored **in-memory**. This data will be lost on server restart. For production, integrate a persistent store (e.g., Redis, PostgreSQL).
    -   The FAISS vector store for RAG is also built **in-memory** at startup. For very large document sets or persistence across restarts, consider using a persistent vector database (e.g., ChromaDB, Weaviate, Pinecone) and a more robust document ingestion pipeline.
-   **Scalability**: For high traffic, deploy multiple API instances behind a load balancer. This necessitates a shared session store and potentially a shared/replicated vector store solution.
-   **Security**: Implement robust authentication and authorization if handling sensitive data or requiring user-specific access. The current API is open.
-   **LLM & Embeddings Model Management**: Ensure your Ollama instance is appropriately resourced for the chosen LLM and embeddings model.
-   **Configuration Management**: For managing multiple persona files and RAG documents, consider a more structured approach like storing them in a dedicated configuration repository or object storage, and syncing them to the application instances.
-   **Error Handling & Resilience**: Enhance error handling for file loading (personas, RAG documents) and external service calls (Ollama).
-   **Data Privacy**: Adhere strictly to data privacy regulations (GDPR, HIPAA, etc.) relevant to your target regions, especially when handling mental health-related conversations.

