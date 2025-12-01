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
  safe: boolean;
}

export interface ChatRequest {
  message: string; // Backend expects 'message' field
}

