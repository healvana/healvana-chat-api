# Healvana Persona Chat API

**Version:** 0.1.0

## Description

This project provides a production-ready FastAPI application serving a conversational AI with the persona of "Healvana," a professional psychiatrist and mental health specialist. It integrates with a local Ollama instance to run Large Language Models (LLMs) and supports streaming responses for a real-time chat experience.

Key features include:
- **Healvana Persona**: The AI is primed with a detailed system prompt to embody "Healvana," offering empathetic support, structured assessment guidance, and concise interactions.
- **Session Management**: In-memory conversation history per session ID.
- **Streaming Chat**: Real-time token-by-token responses via Server-Sent Events (SSE).
- **Configuration**: Ollama model and connection parameters are configurable via environment variables.
- **Comprehensive API**: Endpoints for chat interaction, session history management, and configuration retrieval.
- **Dockerized**: Designed to be run easily using Docker and Docker Compose.

## Prerequisites

- **Docker and Docker Compose**: For running the application and its dependencies (Ollama).
- **Ollama**: Installed and running, with the desired LLM model pulled (e.g., `ollama pull llama3.2:3b-instruct-q5_K_M` or `ollama pull llama3.2:3b-instruct-q5_K_M`). The model name needs to be configured via an environment variable.
- **Python 3.9+** (if running outside Docker for development).
- **Poetry** (if running outside Docker for Python dependency management).

## Project Structure (Assumed)

```bash
├── README.md
├── docker-compose.yml
├── langchain
│   ├── Dockerfile
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── serve.py
├── ollama
│   ├── Dockerfile
│   └── start-ollama.sh
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

The primary method for running this API is using Docker Compose, which manages the FastAPI application and the Ollama service.

1.  **Clone the Repository (if applicable)**
    ```bash
    # git clone git@github.com:healvana/healvana-chat-api.git
    # cd healvana-chat-api
    ```

2.  **Configure Environment Variables**:
    Create a `.env` file in the root directory of your project (or configure these directly in your `docker-compose.yml` or deployment environment):

    ```env
    # .env file
    OLLAMA_MODEL=llama3.2:3b-instruct-q5_K_M # Replace with your desired Ollama model (e.g., llama3:8b-instruct)
    # OLLAMA_HOST=ollama    # Already set in docker-compose.yml for langchain service
    # OLLAMA_PORT=11434   # Already set in docker-compose.yml for langchain service
    OLLAMA_TEMPERATURE=0.2  # LLM generation temperature
    # OLLAMA_BASE_URL=http://ollama:11434 # Optional: if OLLAMA_HOST/PORT are not used
    ```
    The `OLLAMA_HOST` and `OLLAMA_PORT` are typically set within the `docker-compose.yml` for the `langchain` service to point to the `ollama` service.

3.  **Build and Run with Docker Compose**:
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

The application uses the following environment variables for configuration:

-   `OLLAMA_MODEL` (Required): The name of the Ollama model to use (e.g., `llama3.2:3b-instruct-q5_K_M`, `llama3.2:3b-instruct-q5_K_M`).
-   `OLLAMA_HOST` (Optional, Default: `ollama`): The hostname for the Ollama service. Used if `OLLAMA_BASE_URL` is not set.
-   `OLLAMA_PORT` (Optional, Default: `11434`): The port for the Ollama service. Used if `OLLAMA_BASE_URL` is not set.
-   `OLLAMA_BASE_URL` (Optional): The full base URL for the Ollama API (e.g., `http://my-custom-ollama:11434`). If set, this overrides `OLLAMA_HOST` and `OLLAMA_PORT`.
-   `OLLAMA_TEMPERATURE` (Optional, Default: `0.2`): The temperature setting for the LLM's generation, controlling randomness. Lower values make the output more deterministic.
-   `LOG_LEVEL` (Optional, Default: `INFO`): The logging level for the application (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).

## API Endpoints

The API documentation (Swagger UI) is available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc` when the server is running.

---

### Management Endpoints

#### 1. Health Check

-   **Endpoint**: `GET /health`
-   **Description**: Checks the health of the API and the status of the LLM initialization.
-   **`curl` Example**:
    ```bash
    curl -X GET http://localhost:8000/health
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "status": "ok",
      "model_name": "llama3.2:3b-instruct-q5_K_M",
      "ollama_url_used": "http://ollama:11434",
      "ollama_temperature": 0.2
    }
    ```

#### 2. Get Persona Configuration

-   **Endpoint**: `GET /config/persona`
-   **Description**: Retrieves the current Healvana system prompt and initial greeting.
-   **`curl` Example**:
    ```bash
    curl -X GET http://localhost:8000/config/persona
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "system_prompt": "You are Healvana, a professional psychiatrist...",
      "initial_greeting": "Hello, I'm Healvana, a mental wellness companion..."
    }
    ```

---

### Chat Interaction Endpoints

#### 1. Standard Chat (Non-Streaming)

-   **Endpoint**: `POST /chat`
-   **Description**: Sends a message to the AI and receives a complete response. Maintains conversation history per `session_id`.
-   **Request Body**:
    ```json
    {
      "session_id": "your-unique-session-id-123",
      "message": "Hello, I'm feeling a bit down today."
    }
    ```
-   **`curl` Example**:
    ```bash
    curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
          "session_id": "session-abc-123",
          "message": "Hi Healvana, how are you?"
        }'
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "session-abc-123",
      "response": "I'm here to listen. What's on your mind?"
    }
    ```

#### 2. Streaming Chat

-   **Endpoint**: `POST /chat/stream`
-   **Description**: Sends a message and receives a streamed response (Server-Sent Events), allowing for real-time token display. Maintains conversation history per `session_id`.
-   **Request Body**:
    ```json
    {
      "session_id": "your-unique-session-id-456",
      "message": "Can you tell me about coping strategies?"
    }
    ```
-   **`curl` Example**:
    ```bash
    curl -X POST http://localhost:8000/chat/stream \
    -H "Content-Type: application/json" \
    -N \
    -d '{
          "session_id": "session-xyz-789",
          "message": "I feel stressed about work."
        }'
    ```
    *(The `-N` flag disables buffering in curl, allowing you to see the stream.)*
-   **Example Streamed Response (text/event-stream)**:
    ```
    data: {"token": "Work"}

    data: {"token": " stress"}

    data: {"token": " can"}

    data: {"token": " be"}

    data: {"token": " tough."}

    data: {"token": " What"}

    data: {"token": " aspects"}

    data: {"token": " are"}

    data: {"token": " challenging"}

    data: {"token": " you?"}

    data: {"end_stream": true, "session_id": "session-xyz-789"}
    ```

---

### Session Management Endpoints

#### 1. Get Session History

-   **Endpoint**: `GET /chat/sessions/{session_id}/history`
-   **Description**: Retrieves the full conversation history for a specific session, including system messages.
-   **`curl` Example** (replace `your-session-id`):
    ```bash
    curl -X GET http://localhost:8000/chat/sessions/session-xyz-789/history
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "session-xyz-789",
      "history": [
        {
          "type": "system",
          "content": "You are Healvana, a professional psychiatrist..."
        },
        {
          "type": "ai",
          "content": "Hello, I'm Healvana, a mental wellness companion... How have you been feeling lately?"
        },
        {
          "type": "human",
          "content": "I feel stressed about work."
        },
        {
          "type": "ai",
          "content": "Work stress can be tough. What aspects are challenging you?"
        }
      ]
    }
    ```
-   **Example Error Response (404 Not Found)** (if session_id does not exist):
    ```json
    {
      "detail": "Session session-xyz-789 not found or has no history."
    }
    ```

#### 2. Clear Session History

-   **Endpoint**: `DELETE /chat/sessions/{session_id}/history`
-   **Description**: Clears the conversation history for a specific session. The session will be re-initialized with the Healvana persona and greeting on its next use.
-   **`curl` Example** (replace `your-session-id`):
    ```bash
    curl -X DELETE http://localhost:8000/chat/sessions/session-xyz-789/history
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "session-xyz-789",
      "message": "Session history cleared successfully. It will be re-initialized with persona on next interaction."
    }
    ```
-   **Example Error Response (404 Not Found)** (if session_id does not exist):
    ```json
    {
      "detail": "Session session-xyz-789 not found."
    }
    ```

#### 3. Get Initial AI Greeting for a Session

-   **Endpoint**: `GET /chat/sessions/{session_id}/greeting`
-   **Description**: Gets Healvana's initial greeting. If the session is new, this endpoint will also initialize it. Useful for UIs to display the greeting before the user's first message.
-   **`curl` Example** (replace `your-session-id` or use a new one):
    ```bash
    curl -X GET http://localhost:8000/chat/sessions/new-session-for-greeting/greeting
    ```
-   **Example Success Response (200 OK)**:
    ```json
    {
      "session_id": "new-session-for-greeting",
      "greeting": "Hello, I'm Healvana, a mental wellness companion. I'd like to help understand how you've been feeling recently. Over the next few minutes, I'll ask you some standard questions that mental health professionals typically use in assessments. This helps provide more structured support, though it's not a replacement for professional care. How have you been feeling lately?"
    }
    ```

## Notes on Production

-   **CORS**: The current CORS (Cross-Origin Resource Sharing) settings (`allow_origins=["*"]`) are permissive. For production, restrict `allow_origins` to your specific frontend domain(s).
-   **Session Memory**: The chat session history is stored in-memory. This means all session data will be lost if the FastAPI server restarts. For persistent sessions in a production environment, consider integrating a database (e.g., PostgreSQL, MySQL) or a persistent key-value store (e.g., Redis) to store `session_memories`.
-   **Scalability**: For high-traffic applications, consider deploying multiple instances of the API behind a load balancer. This would necessitate a shared session store (as mentioned above).
-   **Security**: Ensure appropriate authentication and authorization mechanisms are in place if the API handles sensitive data or requires user-specific access. The current API is open.
-   **LLM Resource Management**: Monitor the resource usage of your Ollama instance, as LLMs can be resource-intensive.

