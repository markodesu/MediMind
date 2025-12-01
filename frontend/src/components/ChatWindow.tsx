import React from 'react';
import type { Message } from '../lib/types';
import MessageList from './MessageList';
import Composer from './Composer';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  onSendMessage,
  onClearChat,
}) => {
  return (
    <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full px-4 sm:px-6 py-6">
      <div className="flex-1 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-medical-secondary/20 dark:border-slate-700 flex flex-col overflow-hidden">
        {/* Chat Header with Clear Button */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-medical-secondary/20 dark:border-slate-700">
          <h2 className="text-lg font-semibold text-medical-accent dark:text-medical-primary-dark">Chat</h2>
          {messages.length > 0 && (
            <button
              onClick={onClearChat}
              className="text-sm text-medical-text/60 dark:text-medical-text-dark/60 hover:text-medical-accent dark:hover:text-medical-primary-dark transition-colors duration-200 flex items-center space-x-1 px-3 py-1.5 rounded-lg hover:bg-medical-bg dark:hover:bg-slate-700"
              aria-label="Clear chat"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              <span>Clear</span>
            </button>
          )}
        </div>

        {/* Messages Area */}
        <MessageList messages={messages} isLoading={isLoading} />

        {/* Composer */}
        <Composer onSendMessage={onSendMessage} disabled={isLoading} />
      </div>
    </div>
  );
};

export default ChatWindow;

