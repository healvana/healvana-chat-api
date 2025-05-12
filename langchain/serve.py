import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, AsyncGenerator, Optional, List

from fastapi import FastAPI, HTTPException, Body, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage


# --- Configuration ---
class Settings(BaseSettings):
    """Manages application settings using environment variables."""
    ollama_model: str = Field(..., description="Name of the Ollama model to use.", validation_alias='OLLAMA_MODEL')

    ollama_base_url_env: Optional[str] = Field(default=None, description="Optional: Full base URL for Ollama.",
                                               validation_alias='OLLAMA_BASE_URL')
    ollama_host: str = Field(default="ollama", description="Hostname for the Ollama service.",
                             validation_alias='OLLAMA_HOST')
    ollama_port: int = Field(default=11434, description="Port for the Ollama service.", validation_alias='OLLAMA_PORT')
    ollama_temperature: float = Field(default=0.2, description="Temperature for Ollama LLM generation.",
                                      validation_alias='OLLAMA_TEMPERATURE')

    actual_ollama_base_url: Optional[str] = None

    app_title: str = "Comprehensive LangChain Chat API with Healvana Persona"
    app_version: str = "1.3.0"  # Updated version for more APIs
    log_level: str = "INFO"

    # Healvana Psychiatrist Persona Configuration - Reinforced
    healvana_system_prompt: str = (
        "You are Healvana, a professional psychiatrist and mental health specialist. "
        "**Your responses MUST be very short and concise, typically 10-15 words MAXIMUM, and usually a question to guide the conversation.** "  # Reinforced instruction
        "Maintain a conversational flow. Talk to the user as if you genuinely care. Use clinical language balanced with warm, accessible explanations. "
        "You have extensive clinical experience treating various mental health conditions and are trained in evidence-based therapeutic approaches.\n\n"
        "CLINICAL REFERENCE KNOWLEDGE:\n"
        "You have expertise in treating conditions including:\n"
        "- Work stress and social isolation (like case of Mr. J, 30-year-old engineer with remote work challenges)\n"
        "- Low self-esteem and negative self-perception (like case of Mr. B, 35-year-old who attributes success to luck)\n"
        "- Depression with motivational issues (like case of Ms. J, 32-year-old with anhedonia and self-critical thoughts)\n"
        "- Social anxiety and specific phobias (like case of Mr. GB, 23-year-old with situational anxiety)\n"
        "- Major depressive disorder with suicidal ideation (like case of Mr. I with pessimistic outlook)\n"
        "- Generalized anxiety disorder (like case of Mr. X with excessive worry after father's death)\n"
        "- Complicated grief and bereavement (like case of Ms. T after stillbirth experience)\n"
        "- Panic disorder with physical symptoms (like case of Ms. S with catastrophic misinterpretation)\n"
        "- Mixed anxiety and depression (like case of Ms. P with somatic symptoms)\n\n"
        "STRUCTURED ASSESSMENT APPROACH:\n"
        "1. Begin with a warm greeting and brief explanation of the assessment process.\n"
        "2. Conduct a PHQ-2 screen for depression (interest/pleasure in activities and feeling down/hopeless).\n"
        "3. Conduct a GAD-2 screen for anxiety (nervousness/anxiety and worry).\n"
        "4. Explore sleep patterns and any changes.\n"
        "5. Inquire about appetite or weight changes.\n"
        "6. Assess energy levels and fatigue.\n"
        "7. Explore social support systems and relationships.\n"
        "8. Screen for self-harm or suicidal thoughts with appropriate protocols.\n\n"
        "EVIDENCE-BASED THERAPEUTIC TECHNIQUES:\n"
        "For depression symptoms:\n"
        "- Behavioral activation: Start with small, manageable activities that provide pleasure or a sense of achievement (like walking with a friend for 5-10 minutes as with Ms. M).\n"
        "- Activity scheduling with mood rating (as used with Ms. J).\n"
        "- Thought record work to identify and challenge negative thoughts (Date|Situation|Thought|Emotion|Behavior format).\n"
        "- Identifying cognitive distortions like mind reading, emotional reasoning, catastrophizing, and overgeneralization.\n"
        "For anxiety symptoms:\n"
        "- Deep breathing exercises (5-10 minutes when first noticing anxiety symptoms).\n"
        "- Progressive muscle relaxation before sleep (as used with Mr. X).\n"
        "- Cognitive restructuring with evidence for and against anxious predictions.\n"
        "- Gradual exposure to feared situations while using relaxation techniques.\n"
        "- Identifying and replacing anxious automatic thoughts.\n"
        "For interpersonal issues:\n"
        "- Problem-solving approach: Identify issue, brainstorm solutions, evaluate pros/cons, implement best option (as with Mr. J).\n"
        "- Assertiveness training with \"I\" statements (as used with Ms. K).\n"
        "- Role-playing difficult conversations (as used with Ms. T for grief communication).\n"
        "- Social reconnection strategies (like Mr. J reaching out to old friends).\n"
        "For self-esteem and self-criticism:\n"
        "- Strength identification and tracking daily use of strengths (as with Mr. B).\n"
        "- Self-compassion letter writing exercise (as with Mr. B).\n"
        "- Savoring positive experiences technique (as with Mr. B and Mr. J).\n"
        "- Challenging unrealistic expectations and perfectionism.\n\n"
        "SESSION STRUCTURE MODEL:\n"
        "1. Open with \"How have you been feeling since we last spoke?\" or \"How has your week been?\"\n"
        "2. Review previously suggested techniques and homework assignments\n"
        "4. Introduce one new concept or technique with clear rationale\n"
        "5. Practice the technique during session when possible\n"
        "6. Assign specific, achievable homework (thought records, relaxation practice, etc.)\n\n"
        "RISK ASSESSMENT PROTOCOL:\n"
        "For suicidal ideation or self-harm:\n"
        "1. Directly address concerning statements and assess risk level\n"
        "2. Provide immediate crisis resources:\n"
        "   - 988 Suicide & Crisis Lifeline: Call or text 988\n"
        "   - Crisis Text Line: Text HOME to 741741\n"
        "   - Emergency Services: 911 or nearest emergency room\n"
        "3. Advise against staying alone if at high risk\n"
        "5. Continue to monitor risk in follow-up conversations\n\n"
        "INTERACTION GUIDELINES:\n"
        "- **Your response MUST be very short and concise, typically 10-15 words MAXIMUM, and usually a question.**\n"  # Reinforced instruction
        "- Talk to me like you care.\n"
        "- Use clinical language balanced with warm, accessible explanations\n"
        "- Ask about symptom duration, severity, context, and history\n"
        "- Use validation statements frequently: \"That sounds difficult\" or \"I understand how that could be overwhelming\"\n"
        "- Use Socratic questioning to help challenge thoughts: \"What evidence supports this thought?\" \"What evidence contradicts it?\"\n"
        "- Progress gradually from simpler to more complex techniques\n"
        "- Space sessions further apart as improvement occurs\n\n"
        "IMPORTANT ETHICAL BOUNDARIES:\n"
        "- Balance clinical assessment with empathy and support\n\n"
        "Begin with a warm welcome and open-ended question about current feelings, then gradually move into structured assessment based on their initial response. **Remember to keep your responses very short (10-15 words max) and ask guiding questions.**"
    )
    healvana_initial_greeting: str = (
        "Hello, I'm Healvana, a mental wellness companion. I'd like to help understand how you've been feeling recently. "
        "Over the next few minutes, I'll ask you some standard questions that mental health professionals typically use in assessments. "
        "This helps provide more structured support, though it's not a replacement for professional care. How have you been feeling lately?"
    )

    @model_validator(mode='after')
    def assemble_ollama_url(self) -> 'Settings':
        if self.ollama_base_url_env:
            self.actual_ollama_base_url = self.ollama_base_url_env
        else:
            self.actual_ollama_base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        return self

    class Config:
        env_file = '.env'
        extra = 'ignore'


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & Initialization ---
try:
    settings = Settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    if settings.ollama_base_url_env:
        logger.info(f"Using OLLAMA_BASE_URL: {settings.actual_ollama_base_url}")
    else:
        logger.info(
            f"Constructed Ollama URL from OLLAMA_HOST ({settings.ollama_host}) and OLLAMA_PORT ({settings.ollama_port}): {settings.actual_ollama_base_url}")
    logger.info(f"Ollama Temperature: {settings.ollama_temperature}")


except Exception as e:
    logger.exception("Failed to initialize settings. Ensure OLLAMA_MODEL is set.")
    raise SystemExit(f"Configuration error: {e}")

session_memories: Dict[str, ConversationBufferMemory] = {}
llm: Optional[OllamaLLM] = None


# --- Pydantic Models for API ---
class MessageOutput(BaseModel):
    """Model for representing a single message in the history."""
    type: str  # 'human', 'ai', 'system'
    content: str

    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> 'MessageOutput':
        return cls(type=message.type, content=message.content)


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[MessageOutput]


class PersonaConfigResponse(BaseModel):
    system_prompt: str
    initial_greeting: str


class InitialGreetingResponse(BaseModel):
    session_id: str
    greeting: str


class ClearSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the chat session.")
    message: str = Field(..., description="The user's message.")


class ChatResponse(BaseModel):
    session_id: str
    response: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    ollama_url_used: str
    ollama_temperature: float


# --- Helper Functions ---
def get_session_memory(session_id: str, create_if_not_exists: bool = True) -> Optional[ConversationBufferMemory]:
    """
    Retrieves or creates a ConversationBufferMemory for a given session ID.
    If create_if_not_exists is True and memory doesn't exist, it's initialized with the Healvana persona.
    Returns None if memory doesn't exist and create_if_not_exists is False.
    """
    global settings
    if session_id not in session_memories and create_if_not_exists:
        logger.info(f"Creating new memory and Healvana persona for session: {session_id}")
        new_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

        system_message = SystemMessage(content=settings.healvana_system_prompt)
        new_memory.chat_memory.add_message(system_message)

        ai_greeting = AIMessage(content=settings.healvana_initial_greeting)
        new_memory.chat_memory.add_message(ai_greeting)

        session_memories[session_id] = new_memory
    elif session_id not in session_memories and not create_if_not_exists:
        return None

    return session_memories.get(session_id)


# --- FastAPI Lifespan Events ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global llm, settings
    logger.info("Application startup...")
    logger.info(
        f"Using settings (excluding system prompt): {settings.model_dump(exclude_none=True, exclude={'healvana_system_prompt'})}")

    if not settings.ollama_model:
        logger.error("OLLAMA_MODEL environment variable is not set or empty.")
        raise EnvironmentError("OLLAMA_MODEL environment variable is not set or empty.")

    if not settings.actual_ollama_base_url:
        logger.error("Ollama base URL could not be determined.")
        raise EnvironmentError("Ollama base URL could not be determined.")

    logger.info(
        f"Initializing Ollama LLM with model: {settings.ollama_model}, "
        f"URL: {settings.actual_ollama_base_url}, "
        f"Temperature: {settings.ollama_temperature}"
    )
    try:
        llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.actual_ollama_base_url,
            temperature=settings.ollama_temperature
        )
        logger.info("LLM initialized successfully.")
    except Exception as e:
        logger.exception(f"FATAL: Failed to initialize OllamaLLM: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}") from e
    yield
    logger.info("Application shutdown...")
    session_memories.clear()
    logger.info("Cleared session memories.")


# --- FastAPI Application ---
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="A comprehensive chat interface powered by LangChain + Ollama, with Healvana persona and session management.",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],  # Added DELETE
    allow_headers=["*"],
)


# --- API Endpoints ---

# --- Management Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Management"])
async def health_check():
    if not llm or not settings or not settings.actual_ollama_base_url:
        raise HTTPException(status_code=503, detail="Service Unavailable: LLM or settings not initialized.")
    return HealthResponse(
        status="ok",
        model_name=settings.ollama_model,
        ollama_url_used=settings.actual_ollama_base_url,
        ollama_temperature=settings.ollama_temperature
    )


@app.get("/config/persona", response_model=PersonaConfigResponse, tags=["Management"])
async def get_persona_configuration():
    """Retrieves the current Healvana system prompt and initial greeting."""
    if not settings:
        raise HTTPException(status_code=503, detail="Settings not initialized.")
    return PersonaConfigResponse(
        system_prompt=settings.healvana_system_prompt,
        initial_greeting=settings.healvana_initial_greeting
    )


# --- Chat Interaction Endpoints ---
@app.post("/chat", response_model=ChatResponse, tags=["Chat Interaction"])
async def chat_endpoint(request: ChatRequest = Body(...)):
    if not llm:
        logger.error("LLM not available for /chat request.")
        raise HTTPException(status_code=503, detail="LLM not available.")

    logger.info(f"Received chat request for session: {request.session_id} with message: '{request.message[:50]}...'")
    try:
        memory = get_session_memory(request.session_id, create_if_not_exists=True)
        if not memory:  # Should not happen if create_if_not_exists is True
            raise HTTPException(status_code=404,
                                detail=f"Session {request.session_id} memory could not be initialized.")

        chat_history_messages = memory.load_memory_variables({})['history']
        messages_for_llm = chat_history_messages + [HumanMessage(content=request.message)]
        ai_response_str = await llm.ainvoke(messages_for_llm)
        memory.save_context(
            {"input": request.message},
            {"output": ai_response_str}
        )
        logger.info(
            f"Successfully processed chat for session: {request.session_id}. Response length: {len(ai_response_str)}")
        return ChatResponse(session_id=request.session_id, response=ai_response_str)
    except Exception as e:
        logger.exception(f"Error processing chat for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/chat/stream", tags=["Chat Interaction"])
async def chat_stream_endpoint(request: ChatRequest = Body(...)):
    if not llm:
        logger.error("LLM not available for /chat/stream request.")
        raise HTTPException(status_code=503, detail="LLM not available.")

    logger.info(
        f"Received streaming chat request for session: {request.session_id} with message: '{request.message[:50]}...'")

    memory = get_session_memory(request.session_id, create_if_not_exists=True)
    if not memory:  # Should not happen
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} memory could not be initialized.")

    chat_history_messages = memory.load_memory_variables({})['history']
    messages_for_llm = chat_history_messages + [HumanMessage(content=request.message)]

    async def stream_generator() -> AsyncGenerator[str, None]:
        full_response_content = ""
        try:
            logger.debug(f"Starting LLM stream for session {request.session_id}")
            async for str_chunk in llm.astream(messages_for_llm):
                token = str_chunk
                if token:
                    full_response_content += token
                    escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r',
                                                                                                                 '\\r')
                    yield f"data: {{\"token\": \"{escaped_token}\"}}\n\n"

            yield f"data: {{\"end_stream\": true, \"session_id\": \"{request.session_id}\"}}\n\n"
            logger.info(
                f"Stream ended for session: {request.session_id}. Total response length: {len(full_response_content)}")
            memory.save_context({"input": request.message}, {"output": full_response_content})
            logger.info(f"Memory updated for session: {request.session_id} after streaming.")
        except Exception as e:
            logger.exception(f"Error during streaming for session {request.session_id}: {e}")
            escaped_error = str(e).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            yield f"data: {{\"error\": \"{escaped_error}\", \"session_id\": \"{request.session_id}\"}}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# --- Session Management Endpoints ---
@app.get("/chat/sessions/{session_id}/history", response_model=SessionHistoryResponse, tags=["Session Management"])
async def get_session_chat_history(
        session_id: str = Path(..., description="The ID of the session to retrieve history for.")):
    """Retrieves the conversation history for a specific session."""
    memory = get_session_memory(session_id, create_if_not_exists=False)  # Don't create if just fetching
    if not memory:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or has no history.")

    raw_messages = memory.load_memory_variables({})['history']
    history_output = [MessageOutput.from_langchain_message(msg) for msg in raw_messages]

    return SessionHistoryResponse(session_id=session_id, history=history_output)


@app.delete("/chat/sessions/{session_id}/history", response_model=ClearSessionResponse, tags=["Session Management"])
async def clear_session_chat_history(session_id: str = Path(..., description="The ID of the session to clear.")):
    """Clears the conversation history for a specific session. The session will be re-initialized on next use."""
    if session_id in session_memories:
        del session_memories[session_id]
        logger.info(f"Cleared memory for session: {session_id}")
        return ClearSessionResponse(session_id=session_id,
                                    message="Session history cleared successfully. It will be re-initialized with persona on next interaction.")
    else:
        logger.warning(f"Attempted to clear non-existent session: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")


@app.get("/chat/sessions/{session_id}/greeting", response_model=InitialGreetingResponse, tags=["Session Management"])
async def get_initial_ai_greeting(session_id: str = Path(..., description="The ID of the session.")):
    """
    Gets Healvana's initial greeting. If the session is new, it initializes it.
    Useful for UIs to display the greeting before the user's first message.
    """
    memory = get_session_memory(session_id, create_if_not_exists=True)  # Ensure session is created if new
    if not memory:  # Should not happen with create_if_not_exists=True
        raise HTTPException(status_code=500, detail="Failed to initialize session.")

    # The greeting is always the second message after the system prompt in a new session
    # Or, more robustly, just return the configured greeting directly as it's static for new sessions.
    return InitialGreetingResponse(session_id=session_id, greeting=settings.healvana_initial_greeting)


# --- Development Server Runner ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn development server...")
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
