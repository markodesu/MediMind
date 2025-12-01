import type { ChatRequest, ChatResponse, Message } from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Log API URL on module load (only in dev)
if (import.meta.env.DEV) {
  console.log('🔗 API URL:', API_URL);
  console.log('📡 VITE_API_URL env:', import.meta.env.VITE_API_URL || '(not set, using default)');
}

export async function sendMessage(question: string): Promise<Message> {
  const requestBody = { message: question }; // Backend expects 'message' field
  const url = `${API_URL}/api/v1/chat`; // Backend uses /api/v1 prefix

  if (import.meta.env.DEV) {
    console.log('📤 Sending request to:', url);
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error('❌ API Error:', {
        status: response.status,
        statusText: response.statusText,
        body: errorText,
        url,
      });
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data: ChatResponse = await response.json();

    if (import.meta.env.DEV) {
      console.log('✅ Received response:', data);
    }

    return {
      id: `msg-${Date.now()}-${Math.random()}`,
      role: 'assistant',
      text: data.answer,
      timestamp: new Date(),
      confidence: data.confidence,
      safe: data.safe !== undefined ? data.safe : (data.confidence > 0.5), // Backend may not always include 'safe'
    };
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error('🌐 Network Error - Backend may not be running:', {
        url,
        error: error.message,
        hint: 'Make sure the backend is running on http://localhost:8000',
      });
    } else {
      console.error('❌ Error sending message:', error);
    }
    throw error;
  }
}

