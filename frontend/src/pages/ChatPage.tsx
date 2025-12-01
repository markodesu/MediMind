import React, { useState, useCallback } from 'react';
import type { Message } from '../lib/types';
import { sendMessage } from '../lib/api';
import Header from '../components/Header';
import ChatWindow from '../components/ChatWindow';

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = useCallback(async (text: string) => {
    // Create user message
    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const assistantMessage = await sendMessage(text);
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        text: 'Sorry, I encountered an error. Please try again later.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleClearChat = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the chat?')) {
      setMessages([]);
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-medical-bg dark:bg-medical-bg-dark">
      <Header />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onSendMessage={handleSendMessage}
          onClearChat={handleClearChat}
        />
      </main>
    </div>
  );
};

export default ChatPage;

