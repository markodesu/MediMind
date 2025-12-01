export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: Date;
  confidence?: number;
  safe?: boolean;
}

export interface ChatResponse {
  answer: string;
  confidence: number;
  safe?: boolean; // Optional - backend may not always include this
}

export interface MessageHistory {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string; // Backend expects 'message' field
  history?: MessageHistory[]; // Optional conversation history
}

