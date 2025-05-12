<script lang="ts">
  import { onMount, tick } from 'svelte';

  // Define the structure for a chat message
  interface Message {
    id: string;
    text: string;
    sender: 'user' | 'ai';
    timestamp: Date;
    streaming?: boolean; // Optional: to indicate if AI message is still streaming
  }

  // Reactive variables for the chat state
  let messages: Message[] = [];
  let currentMessageText: string = '';
  let isLoading: boolean = false;
  let sessionId: string = '';
  let errorMessage: string | null = null;
  let chatContainerElement: HTMLElement; // For auto-scrolling

  // Agent Name State
  let agentNameInput: string = 'Local Assistant'; // Default agent name suggestion
  let agentName: string | null = null;
  let isSettingAgentName: boolean = true; // Start by asking for agent name

  // API Configuration
  const API_BASE_URL = 'http://localhost:8000'; // Your FastAPI backend URL

  // Generate a unique session ID when the component mounts
  onMount(() => {
    sessionId = crypto.randomUUID();
    // Check if an agent name was previously set for this session (optional persistence)
    const storedAgentName = localStorage.getItem(`agentName-${sessionId}`);
    if (storedAgentName) {
      agentName = storedAgentName;
      isSettingAgentName = false; // Skip asking if already set
      loadMessages(); // Load messages if agent name is already set
    }
    scrollToBottom();
  });

  async function loadMessages() {
    const storedMessages = localStorage.getItem(`chatMessages-${sessionId}`);
    if (storedMessages) {
      messages = JSON.parse(storedMessages).map((msg: Message) => ({...msg, timestamp: new Date(msg.timestamp) }));
    }
  }

  // Function to handle setting the agent's name
  function handleSetAgentName() {
    const trimmedAgentName = agentNameInput.trim();
    if (!trimmedAgentName) {
      errorMessage = "Agent name cannot be empty.";
      return;
    }
    agentName = trimmedAgentName;
    isSettingAgentName = false;
    errorMessage = null;
    localStorage.setItem(`agentName-${sessionId}`, agentName); // Optional: persist agent name
    loadMessages(); // Load messages now that agent name is set
    scrollToBottom();
  }


  // Function to scroll to the bottom of the chat container
  async function scrollToBottom() {
    await tick(); // Wait for DOM updates
    if (chatContainerElement) {
      chatContainerElement.scrollTop = chatContainerElement.scrollHeight;
    }
  }

  // Function to handle sending a message
  async function sendMessage() {
    const trimmedMessage = currentMessageText.trim();
    if (!trimmedMessage || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      text: trimmedMessage,
      sender: 'user',
      timestamp: new Date(),
    };
    messages = [...messages, userMessage];
    currentMessageText = '';
    isLoading = true;
    errorMessage = null;
    await scrollToBottom();

    let aiMessageId = crypto.randomUUID();
    let aiMessagePlaceholder: Message = {
        id: aiMessageId,
        text: '',
        sender: 'ai',
        timestamp: new Date(),
        streaming: true,
    };
    messages = [...messages, aiMessagePlaceholder];
    await scrollToBottom();

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: trimmedMessage,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const messageString = buffer.substring(0, boundary);
          buffer = buffer.substring(boundary + 2);

          if (messageString.startsWith('data: ')) {
            const jsonString = messageString.substring(6);
            try {
              const data = JSON.parse(jsonString);

              if (data.token) {
                messages = messages.map(msg =>
                  msg.id === aiMessageId ? { ...msg, text: msg.text + data.token, streaming: true } : msg
                );
                await scrollToBottom();
              } else if (data.end_stream) {
                messages = messages.map(msg =>
                  msg.id === aiMessageId ? { ...msg, streaming: false } : msg
                );
                isLoading = false;
                localStorage.setItem(`chatMessages-${sessionId}`, JSON.stringify(messages));
                break;
              } else if (data.error) {
                errorMessage = `Stream error: ${data.error}`;
                messages = messages.map(msg =>
                  msg.id === aiMessageId ? { ...msg, text: `Error: ${data.error}`, streaming: false, sender: 'ai' } : msg
                );
                isLoading = false;
                break;
              }
            } catch (e) {
              console.error('Failed to parse SSE JSON:', e, jsonString);
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
         if (isLoading === false) break;
      }
      if (isLoading) {
        messages = messages.map(msg => msg.id === aiMessageId ? { ...msg, streaming: false } : msg);
        isLoading = false;
      }

    } catch (error: any) {
      console.error('Failed to send message:', error);
      errorMessage = error.message || 'An unexpected error occurred.';
      messages = messages.map(msg =>
        msg.id === aiMessageId ? { ...msg, text: `Error: ${errorMessage}`, streaming: false } : msg
      );
      isLoading = false;
    } finally {
      if (isLoading) {
         messages = messages.map(msg => msg.id === aiMessageId ? { ...msg, streaming: false } : msg);
         isLoading = false;
      }
      localStorage.setItem(`chatMessages-${sessionId}`, JSON.stringify(messages));
      await scrollToBottom();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function handleAgentNameKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleSetAgentName();
    }
  }

  function formatTime(date: Date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
</script>

<div class="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 to-slate-700 text-white p-4">
  {#if isSettingAgentName}
    <div class="w-full max-w-md bg-slate-800 shadow-2xl rounded-xl p-8 text-center">
      <h1 class="text-2xl font-semibold mb-6">Name Your Chat Agent</h1>
      <input
        type="text"
        bind:value={agentNameInput}
        on:keydown={handleAgentNameKeydown}
        placeholder="E.g., My Assistant"
        class="w-full p-3 mb-4 bg-slate-700 text-slate-100 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none placeholder-slate-400"
      />
      <button
        on:click={handleSetAgentName}
        disabled={!agentNameInput.trim()}
        class="w-full px-6 py-3 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
      >
        Start Chatting
      </button>
      {#if errorMessage && isSettingAgentName}
        <p class="mt-4 text-red-400 text-sm">{errorMessage}</p>
      {/if}
    </div>
  {:else}
    <div class="w-full max-w-2xl flex flex-col h-[calc(100vh-4rem)] bg-slate-800 shadow-2xl rounded-xl overflow-hidden">
      <header class="bg-slate-900 p-4 text-center shadow-md">
        <h1 class="text-2xl font-semibold">{agentName || 'Chat'}</h1>
        <p class="text-xs text-slate-400">Session ID: {sessionId.substring(0,8)}...</p>
      </header>

      <div bind:this={chatContainerElement} class="chat-messages-container flex-grow p-6 space-y-4 overflow-y-auto">
        {#each messages as message (message.id)}
          <div class="flex" class:justify-end={message.sender === 'user'} class:justify-start={message.sender === 'ai'}>
            <div
              class="max-w-[70%] p-3 rounded-xl shadow"
              class:bg-sky-600={message.sender === 'user'}
              class:text-white={message.sender === 'user'}
              class:bg-slate-700={message.sender === 'ai'}
              class:text-slate-100={message.sender === 'ai'}
            >
              <p class="whitespace-pre-wrap break-words">
                {message.text}
                {#if message.streaming}
                  <span class="inline-block w-1 h-4 bg-slate-300 ml-1 animate-pulse"></span>
                {/if}
              </p>
              <p class="text-xs mt-1" class:text-sky-200={message.sender === 'user'} class:text-slate-400={message.sender === 'ai'}>
                {formatTime(message.timestamp)}
              </p>
            </div>
          </div>
        {/each}
        {#if isLoading && messages[messages.length -1]?.sender !== 'ai' && messages[messages.length -1]?.streaming !== false}
          <div class="flex justify-start">
            <div class="max-w-[70%] p-3 rounded-lg shadow bg-slate-700 text-slate-100">
              <p class="animate-pulse">{agentName || 'AI'} is thinking...</p>
            </div>
          </div>
        {/if}
      </div>

      {#if errorMessage && !isSettingAgentName}
        <div class="p-4 bg-red-500 text-white text-sm text-center">
          <p>Error: {errorMessage}</p>
        </div>
      {/if}

      <div class="p-4 bg-slate-900 border-t border-slate-700 shadow-inner">
        <div class="flex items-center space-x-3">
          <textarea
            rows="1"
            class="flex-grow p-3 bg-slate-700 text-slate-100 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none resize-none placeholder-slate-400"
            placeholder="Type your message to {agentName || 'the assistant'}..."
            bind:value={currentMessageText}
            on:keydown={handleKeydown}
            disabled={isLoading}
          ></textarea>
          <button
            on:click={sendMessage}
            disabled={isLoading || !currentMessageText.trim()}
            class="px-6 py-3 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  textarea {
    min-height: 44px;
    max-height: 150px;
  }
</style>
