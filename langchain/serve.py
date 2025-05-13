import os
import logging
import uvicorn
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, AsyncGenerator, Optional, List, Any, Tuple

from fastapi import FastAPI, HTTPException, Body, Path as FastApiPath  # Renamed Path to avoid conflict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

# Langchain Imports
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings  # Use community for embeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Constants ---
PERSONA_DIR = Path("personas")  # Expects 'personas' directory in the same location as serve.py
DOCUMENTS_DIR = Path("documents")  # Expects 'documents' directory in the same location as serve.py


# --- Persona and Localization Configuration ---

class CrisisHotline(BaseModel):
    name: str
    contact: str
    description: Optional[str] = None


class PersonaConfig(BaseModel):
    locale_id: str = Field(..., description="Locale identifier (e.g., en-US, es-ES)")
    persona_name: str = Field(..., description="Display name for the persona")
    system_prompt: str = Field(..., description="The detailed system prompt for the LLM.")
    initial_greeting: str = Field(..., description="The first message the AI sends.")
    crisis_hotlines: List[CrisisHotline] = Field(default_factory=list)


def load_persona_config(locale_id: str) -> Optional[PersonaConfig]:
    """Loads persona configuration from files for a given locale."""
    # Assumes PERSONA_DIR is relative to the current working directory of serve.py
    # When running in Docker, this will be /app/personas if ./langchain is mounted to /app
    current_script_dir = Path(__file__).parent
    locale_path = current_script_dir / PERSONA_DIR / locale_id

    if not locale_path.is_dir():
        logger.warning(f"Locale directory not found: {locale_path}")
        return None

    try:
        system_prompt_file = locale_path / "system.prompt"
        greeting_file = locale_path / "greeting.txt"
        hotlines_file = locale_path / "hotlines.json"

        system_prompt = system_prompt_file.read_text(encoding="utf-8") if system_prompt_file.exists() else \
            "You are a helpful assistant. Please keep your responses concise."
        initial_greeting = greeting_file.read_text(encoding="utf-8") if greeting_file.exists() else \
            "Hello! How can I assist you today?"

        hotlines_data = []
        if hotlines_file.exists():
            with open(hotlines_file, 'r', encoding='utf-8') as f:
                hotlines_raw = json.load(f)
                if isinstance(hotlines_raw, list):
                    hotlines_data = [CrisisHotline(**item) for item in hotlines_raw]
        else:
            logger.warning(f"Hotlines file not found for locale '{locale_id}': {hotlines_file}")

        persona_name = f"Healvana ({locale_id})"

        return PersonaConfig(
            locale_id=locale_id,
            persona_name=persona_name,
            system_prompt=system_prompt,
            initial_greeting=initial_greeting,
            crisis_hotlines=hotlines_data
        )
    except Exception as e:
        logger.error(f"Error loading persona config for locale '{locale_id}' from {locale_path}: {e}")
        return None


def load_all_personas() -> Dict[str, PersonaConfig]:
    """Loads all persona configurations from the persona directory."""
    personas = {}
    current_script_dir = Path(__file__).parent
    effective_persona_dir = current_script_dir / PERSONA_DIR

    if not effective_persona_dir.is_dir():
        logger.warning(
            f"Persona directory '{effective_persona_dir}' not found. Ensure it exists and is populated relative to serve.py.")
        return personas

    for locale_dir in effective_persona_dir.iterdir():
        if locale_dir.is_dir():
            locale_id = locale_dir.name
            config = load_persona_config(locale_id)  # load_persona_config now uses absolute path based on script dir
            if config:
                personas[locale_id] = config
    if not personas:
        logger.warning(f"No persona configurations were loaded from {effective_persona_dir}.")
    return personas


# --- Application Settings ---
class Settings(BaseSettings):
    ollama_model: str = Field(..., validation_alias='OLLAMA_MODEL')
    ollama_base_url_env: Optional[str] = Field(default=None, validation_alias='OLLAMA_BASE_URL')
    ollama_host: str = Field(default="ollama", validation_alias='OLLAMA_HOST')
    ollama_port: int = Field(default=11434, validation_alias='OLLAMA_PORT')
    ollama_temperature: float = Field(default=0.2, validation_alias='OLLAMA_TEMPERATURE')
    embeddings_model: str = Field(default="nomic-embed-text", description="Default Ollama model for embeddings.",
                                  validation_alias='EMBEDDINGS_MODEL')
    # Add option to control whether RAG is enabled by default
    enable_rag_by_default: bool = Field(default=True, validation_alias='ENABLE_RAG_BY_DEFAULT')

    actual_ollama_base_url: Optional[str] = None
    app_title: str = "Global Healvana Chat API with RAG"
    app_version: str = "2.2.0"  # Updated version for RAG option
    log_level: str = "INFO"
    default_locale: str = Field(default="en-US")

    personas: Dict[str, PersonaConfig] = Field(default_factory=load_all_personas)

    @model_validator(mode='after')
    def assemble_ollama_url(self) -> 'Settings':
        if self.ollama_base_url_env:
            self.actual_ollama_base_url = self.ollama_base_url_env
        else:
            self.actual_ollama_base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        return self

    def get_persona_config(self, locale: Optional[str]) -> PersonaConfig:
        target_locale = locale or self.default_locale
        persona = self.personas.get(target_locale)
        if not persona:
            logger.warning(
                f"Persona for locale '{target_locale}' not found. Falling back to default '{self.default_locale}'.")
            persona = self.personas.get(self.default_locale)
            if not persona:
                logger.error(f"CRITICAL: Default persona '{self.default_locale}' not found in loaded configurations!")
                return PersonaConfig(
                    locale_id=self.default_locale,
                    persona_name="Healvana (Emergency Fallback)",
                    system_prompt="You are a helpful assistant. Please be concise.",
                    initial_greeting="Hello! How can I help you today?",
                    crisis_hotlines=[]
                )
        return persona

    class Config:
        env_file = '.env'
        extra = 'ignore'


# --- Logging, Globals ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    settings = Settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    logger.info(f"Default locale set to: {settings.default_locale}")
    logger.info(f"Loaded {len(settings.personas)} personas: {list(settings.personas.keys())}")
    logger.info(f"RAG enabled by default: {settings.enable_rag_by_default}")
    if not settings.personas.get(settings.default_locale):
        logger.error(
            f"Default persona for locale '{settings.default_locale}' failed to load. API might not function as expected.")
    elif not settings.personas:
        logger.warning(f"No personas loaded. Check '{Path(__file__).parent / PERSONA_DIR}' directory.")

except Exception as e:
    logger.exception("Failed to initialize settings.")
    raise SystemExit(f"Configuration error: {e}")

llm: Optional[OllamaLLM] = None
embeddings: Optional[OllamaEmbeddings] = None
vector_store: Optional[FAISS] = None
retriever: Optional[Any] = None
# rag_chain is now assembled per request due to locale-specific system prompts

session_histories: Dict[str, List[BaseMessage]] = {}


# --- Pydantic Models for API ---
class MessageOutput(BaseModel):
    type: str
    content: str

    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> 'MessageOutput':
        return cls(type=message.type, content=message.content)


class SessionHistoryResponse(BaseModel):
    session_id: str
    locale_used: str
    history: List[MessageOutput]


class PersonaInfo(BaseModel):
    locale_id: str
    persona_name: str


class PersonaDetailResponse(PersonaConfig): pass


class InitialGreetingResponse(BaseModel):
    session_id: str
    locale_used: str
    greeting: str


class ClearSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(...)
    message: str = Field(...)
    locale: Optional[str] = Field(None)
    # Add option to enable/disable RAG for individual requests
    use_rag: Optional[bool] = Field(None, description="Override the default RAG setting")


class ChatResponse(BaseModel):
    session_id: str
    locale_used: str
    response: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    embeddings_model: str
    ollama_url_used: str
    ollama_temperature: float
    default_locale: str
    available_personas: List[str]
    rag_status: str
    rag_enabled_by_default: bool
    rag_available: bool


# --- Direct LLM Chain Function ---
def create_direct_llm_chain(llm_model: OllamaLLM) -> Any:
    """Creates a chain that sends messages directly to the LLM without retrieval."""
    logger.debug("Creating direct LLM chain without RAG")
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return prompt | llm_model | StrOutputParser()


# --- RAG Chain Creation Functions ---
def create_rag_retriever(vs: FAISS, k: int = 3) -> Any:
    return vs.as_retriever(search_kwargs={"k": k})


def create_history_aware_retriever_chain(llm_model: OllamaLLM, rag_retriever: Any) -> Any:
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever_runnable = create_history_aware_retriever(
        llm_model, rag_retriever, contextualize_q_prompt
    )
    return history_aware_retriever_runnable


def create_document_question_chain(llm_model: OllamaLLM, persona_specific_system_prompt: str) -> Any:
    qa_system_template = f"""{persona_specific_system_prompt}

You will be provided with relevant context based on the user's question and chat history.
Use this context to inform your answer, but maintain your Healvana persona, tone, and interaction guidelines (especially the 10-15 word limit and asking questions).
If the context doesn't provide a direct answer, rely on your clinical knowledge base while still adhering to the persona.
Do not mention the context explicitly unless it's natural within the persona (e.g., referencing a specific coping strategy discussed).

Context:
{{context}}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Add document_variable_name parameter to match the placeholder in the template
    question_answer_chain = create_stuff_documents_chain(
        llm_model,
        qa_prompt,
        document_variable_name="context"  # This must match the {context} in the template
    )
    return question_answer_chain


# --- Helper Functions ---
def get_session_history(session_id: str, locale: Optional[str] = None, create_if_not_exists: bool = True) -> Tuple[
    List[BaseMessage], str]:
    global settings
    effective_locale = locale or settings.default_locale

    if session_id not in session_histories and create_if_not_exists:
        logger.info(f"Creating new history for session: {session_id} with locale: {effective_locale}")
        persona_config = settings.get_persona_config(effective_locale)
        session_histories[session_id] = [
            SystemMessage(content=persona_config.system_prompt),
            AIMessage(content=persona_config.initial_greeting)
        ]
    elif session_id not in session_histories and not create_if_not_exists:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    current_session_locale = effective_locale
    if session_id in session_histories and session_histories[session_id]:
        first_message = session_histories[session_id][0]
        if isinstance(first_message, SystemMessage):
            for loc_id, p_conf in settings.personas.items():
                if p_conf.system_prompt == first_message.content:
                    current_session_locale = loc_id
                    break

    return session_histories.get(session_id, []), current_session_locale


def clear_session_history_func(session_id: str) -> bool:
    if session_id in session_histories:
        del session_histories[session_id]
        logger.info(f"Cleared in-memory history for session: {session_id}")
        return True
    return False


# --- FastAPI Lifespan & App Initialization ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global llm, embeddings, vector_store, retriever, settings
    rag_status = "Not Initialized"
    rag_available = False

    logger.info("Application startup...")
    logger.info(
        f"Using settings (excluding persona details): {settings.model_dump(exclude_none=True, exclude={'personas'})}")

    if not settings.ollama_model: raise EnvironmentError("OLLAMA_MODEL missing.")
    if not settings.actual_ollama_base_url: raise EnvironmentError("Ollama base URL could not be determined.")

    # Initialize LLM first - this is required regardless of RAG status
    logger.info(
        f"Initializing Ollama LLM: model={settings.ollama_model}, url={settings.actual_ollama_base_url}, temp={settings.ollama_temperature}")
    try:
        llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.actual_ollama_base_url,
            temperature=settings.ollama_temperature
        )
        logger.info("LLM initialized successfully.")
    except Exception as e:
        logger.exception(f"FATAL: Failed to initialize OllamaLLM: {e}")
        llm = None

    # Only initialize RAG components if LLM is available
    if llm:
        try:
            # Initialize Embeddings and check for documents regardless of enable_rag_by_default
            # This allows us to know if RAG is available even if not enabled by default
            logger.info(
                f"Checking RAG availability: Initializing Ollama Embeddings: model={settings.embeddings_model}, url={settings.actual_ollama_base_url}")
            embeddings = OllamaEmbeddings(model=settings.embeddings_model, base_url=settings.actual_ollama_base_url)

            current_script_dir = Path(__file__).parent
            effective_documents_dir = current_script_dir / DOCUMENTS_DIR

            # Check if documents directory exists and has documents
            if not effective_documents_dir.is_dir():
                logger.warning(
                    f"Documents directory '{effective_documents_dir}' not found. RAG will be unavailable unless documents are added.")
                docs = []
            else:
                logger.info(f"Checking for documents in: {effective_documents_dir}")
                loader = DirectoryLoader(str(effective_documents_dir), glob="**/*.txt", loader_cls=TextLoader,
                                         show_progress=True, use_multithreading=False)
                docs = loader.load()
                logger.info(f"Found {len(docs)} documents for potential RAG usage.")

            # If we have documents or settings say to initialize RAG regardless
            if docs or settings.enable_rag_by_default:
                logger.info("Initializing vector store and retriever for RAG...")
                if not docs:
                    logger.warning("No documents loaded. Vector store will be minimal with placeholder documents.")
                    vector_store = FAISS.from_texts(
                        [
                            "No external documents available for Healvana's knowledge base at this time. Rely on your general knowledge and persona training."],
                        embeddings
                    )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    logger.info(f"Split documents into {len(splits)} chunks.")
                    logger.info("Creating FAISS vector store...")
                    vector_store = FAISS.from_documents(splits, embeddings)
                    logger.info("FAISS vector store created successfully.")

                retriever = create_rag_retriever(vector_store, k=3)
                logger.info("Retriever created successfully.")
                rag_status = f"RAG Initialized with {len(docs)} documents" if docs else "RAG Initialized with placeholder document"
                rag_available = True
            else:
                logger.info("No documents found and RAG not enabled by default. RAG components not initialized.")
                rag_status = "RAG Not Initialized (No documents and not enabled by default)"
                rag_available = False
        except Exception as e:
            logger.exception(f"ERROR: Failed to initialize RAG components: {e}")
            rag_status = f"RAG Error: {e}"
            retriever = None
            vector_store = None
            embeddings = None
            rag_available = False
    else:
        rag_status = "RAG SKIPPED: LLM initialization failed."
        logger.error(rag_status)
        rag_available = False

    # Store RAG status in app state
    app_instance.state.rag_status = rag_status
    app_instance.state.rag_available = rag_available

    logger.info(f"Application startup complete. RAG status: {rag_status}")

    yield

    logger.info("Application shutdown...")
    session_histories.clear()
    logger.info("Cleared in-memory session histories.")


# --- FastAPI App ---
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="A comprehensive, global-ready chat API with localized Healvana personas and optional RAG.",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"], allow_headers=["*"],
)


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Management"])
async def health_check():
    rag_status = app.state.rag_status if hasattr(app.state, 'rag_status') else "Unknown"
    rag_available = app.state.rag_available if hasattr(app.state, 'rag_available') else False

    return HealthResponse(
        status="ok" if llm else "degraded",
        model_name=settings.ollama_model,
        embeddings_model=settings.embeddings_model,
        ollama_url_used=str(settings.actual_ollama_base_url),
        ollama_temperature=settings.ollama_temperature,
        default_locale=settings.default_locale,
        available_personas=list(settings.personas.keys()),
        rag_status=rag_status,
        rag_enabled_by_default=settings.enable_rag_by_default,
        rag_available=rag_available
    )


@app.get("/config/personas", response_model=List[PersonaInfo], tags=["Management"])
async def list_available_personas():
    return [PersonaInfo(locale_id=lc, persona_name=pc.persona_name) for lc, pc in settings.personas.items()]


@app.get("/config/personas/{locale_id}", response_model=PersonaDetailResponse, tags=["Management"])
async def get_persona_details(locale_id: str = FastApiPath(..., description="The locale ID (e.g., en-US).")):
    persona_config = settings.personas.get(locale_id)
    if not persona_config:
        raise HTTPException(status_code=404, detail=f"Persona for locale '{locale_id}' not found.")
    return persona_config


# --- Chat Interaction Endpoints (Modified for Optional RAG) ---
async def get_locale_specific_rag_chain(active_locale: str) -> Any:
    """Dynamically assembles the RAG chain with the locale-specific system prompt."""
    global llm, retriever, settings
    if not llm or not retriever:
        logger.error(
            f"get_locale_specific_rag_chain: LLM is {'set' if llm else 'None'}, Retriever is {'set' if retriever else 'None'}")
        raise HTTPException(status_code=503, detail="Core RAG components not ready.")

    persona_config = settings.get_persona_config(active_locale)

    history_aware_retriever_chain_runnable = create_history_aware_retriever_chain(llm, retriever)
    question_answer_chain_runnable = create_document_question_chain(llm, persona_config.system_prompt)

    return create_retrieval_chain(history_aware_retriever_chain_runnable, question_answer_chain_runnable)


@app.post("/chat", response_model=ChatResponse, tags=["Chat Interaction"])
async def chat_endpoint(request: ChatRequest = Body(...)):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not available.")

    # Determine if RAG should be used for this request
    use_rag = request.use_rag if request.use_rag is not None else settings.enable_rag_by_default
    rag_available = app.state.rag_available if hasattr(app.state, 'rag_available') else False

    # If RAG is requested but not available, log a warning and fall back to direct LLM
    if use_rag and not rag_available:
        logger.warning(f"RAG requested for session {request.session_id} but not available. Falling back to direct LLM.")
        use_rag = False

    active_locale = request.locale or settings.default_locale
    logger.info(
        f"Chat request for session: {request.session_id}, locale: {active_locale}, use_rag: {use_rag}, message: '{request.message[:50]}...'")

    history, locale_used_for_session = get_session_history(request.session_id, locale=active_locale,
                                                           create_if_not_exists=True)

    try:
        ai_response_str = ""

        if use_rag:
            # RAG path
            logger.debug(f"Using RAG for session {request.session_id}")
            current_rag_chain = await get_locale_specific_rag_chain(active_locale)
            response = await current_rag_chain.ainvoke({
                "input": request.message,
                "chat_history": history[1:]  # Pass history *without* the initial SystemMessage
            })
            ai_response_str = response.get("answer", "Sorry, I couldn't generate a response.")
        else:
            # Direct LLM path
            logger.debug(f"Using direct LLM (no RAG) for session {request.session_id}")
            direct_chain = create_direct_llm_chain(llm)
            ai_response_str = await direct_chain.ainvoke({
                "input": request.message,
                "chat_history": history  # Include the full history with SystemMessage
            })

        # Update history (append user message and AI response)
        updated_history = history + [
            HumanMessage(content=request.message),
            AIMessage(content=ai_response_str)
        ]
        session_histories[request.session_id] = updated_history
        logger.debug(f"History updated for session {request.session_id}. Response length: {len(ai_response_str)}")

        return ChatResponse(session_id=request.session_id, locale_used=active_locale, response=ai_response_str)
    except Exception as e:
        logger.exception(f"Error processing chat for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/chat/stream", tags=["Chat Interaction"])
async def chat_stream_endpoint(request: ChatRequest = Body(...)):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not available.")

    # Determine if RAG should be used for this request
    use_rag = request.use_rag if request.use_rag is not None else settings.enable_rag_by_default
    rag_available = app.state.rag_available if hasattr(app.state, 'rag_available') else False

    # If RAG is requested but not available, log a warning and fall back to direct LLM
    if use_rag and not rag_available:
        logger.warning(
            f"RAG requested for streaming session {request.session_id} but not available. Falling back to direct LLM.")
        use_rag = False

    active_locale = request.locale or settings.default_locale
    logger.info(
        f"Streaming request for session: {request.session_id}, locale: {active_locale}, use_rag: {use_rag}, message: '{request.message[:50]}...'")

    history, locale_used_for_session = get_session_history(request.session_id, locale=active_locale,
                                                           create_if_not_exists=True)

    async def stream_generator() -> AsyncGenerator[str, None]:
        full_response_content = ""
        last_chunk = None  # Initialize last_chunk to handle the case where no chunks are generated

        try:
            if use_rag:
                # RAG path
                logger.debug(f"Using RAG for streaming session {request.session_id}")
                current_rag_chain = await get_locale_specific_rag_chain(active_locale)

                # Stream the response from the RAG chain
                async for chunk in current_rag_chain.astream({
                    "input": request.message,
                    "chat_history": history[1:]  # Pass history *without* the initial SystemMessage
                }):
                    last_chunk = chunk  # Save the last chunk for potential fallback
                    if "answer" in chunk and chunk["answer"] is not None:
                        token = chunk["answer"]
                        if isinstance(token, str):
                            full_response_content += token
                            escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n',
                                                                                                    '\\n').replace('\r',
                                                                                                                   '\\r')
                            yield f"data: {{\"token\": \"{escaped_token}\", \"locale_used\": \"{active_locale}\"}}\n\n"
            else:
                # Direct LLM path
                logger.debug(f"Using direct LLM (no RAG) for streaming session {request.session_id}")
                direct_chain = create_direct_llm_chain(llm)

                # Stream the response from the direct LLM chain
                async for token in direct_chain.astream({
                    "input": request.message,
                    "chat_history": history  # Include the full history with SystemMessage
                }):
                    if isinstance(token, str):
                        full_response_content += token
                        escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace(
                            '\r', '\\r')
                        yield f"data: {{\"token\": \"{escaped_token}\", \"locale_used\": \"{active_locale}\"}}\n\n"

            # Fallback if streaming didn't yield individual tokens (e.g. full answer in last chunk)
            if not full_response_content and last_chunk is not None and "answer" in last_chunk and isinstance(
                    last_chunk["answer"], str):
                logger.debug(f"Using fallback for session {request.session_id} as no tokens were streamed")
                full_response_content = last_chunk["answer"]
                # No need to yield again if it was already yielded as the last token

            yield f"data: {{\"end_stream\": true, \"session_id\": \"{request.session_id}\", \"locale_used\": \"{active_locale}\"}}\n\n"

            if full_response_content:
                # Get the latest history state before appending
                final_history_state, _ = get_session_history(request.session_id, create_if_not_exists=False)
                final_history_state.append(HumanMessage(content=request.message))
                final_history_state.append(AIMessage(content=full_response_content))
                session_histories[request.session_id] = final_history_state
                logger.info(
                    f"History updated for session {request.session_id} after stream. Response length: {len(full_response_content)}")
            else:
                logger.warning(f"No content was generated or streamed for session {request.session_id}.")

        except Exception as e:
            logger.exception(f"Error during streaming for session {request.session_id}: {e}")
            escaped_error = str(e).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            yield f"data: {{\"error\": \"{escaped_error}\", \"session_id\": \"{request.session_id}\", \"locale_used\": \"{active_locale}\"}}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# --- Session Management Endpoints ---
@app.get("/chat/sessions/{session_id}/history", response_model=SessionHistoryResponse, tags=["Session Management"])
async def get_session_chat_history(session_id: str = FastApiPath(...)):
    history_list, locale_used = get_session_history(session_id, create_if_not_exists=False)
    history_output = [MessageOutput.from_langchain_message(msg) for msg in history_list]
    return SessionHistoryResponse(session_id=session_id, locale_used=locale_used, history=history_output)


@app.delete("/chat/sessions/{session_id}/history", response_model=ClearSessionResponse, tags=["Session Management"])
async def clear_session_chat_history_endpoint(session_id: str = FastApiPath(...)):
    if clear_session_history_func(session_id):
        return ClearSessionResponse(session_id=session_id, message="Session history cleared.")
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")


@app.get("/chat/sessions/{session_id}/greeting", response_model=InitialGreetingResponse, tags=["Session Management"])
async def get_initial_ai_greeting(session_id: str = FastApiPath(...), locale: Optional[str] = None):
    active_locale = locale or settings.default_locale
    get_session_history(session_id, locale=active_locale, create_if_not_exists=True)
    persona_config = settings.get_persona_config(active_locale)
    return InitialGreetingResponse(session_id=session_id, locale_used=active_locale,
                                   greeting=persona_config.initial_greeting)


# --- Development Server Runner ---
if __name__ == "__main__":
    # Ensure persona and document directories exist before starting
    current_dir = Path(__file__).parent
    if not (current_dir / PERSONA_DIR).exists():
        (current_dir / PERSONA_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created missing persona directory: {current_dir / PERSONA_DIR}")
    if not (current_dir / DOCUMENTS_DIR).exists():
        (current_dir / DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created missing documents directory: {current_dir / DOCUMENTS_DIR}")

    uvicorn.run(
        "__main__:app", host="0.0.0.0", port=8000, reload=True,
        log_level=settings.log_level.lower()
    )