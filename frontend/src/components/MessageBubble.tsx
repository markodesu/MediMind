import React from 'react';
import type { Message } from '../lib/types';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex items-start space-x-3 mb-4 animate-fade-in ${
        isUser ? 'flex-row-reverse space-x-reverse' : ''
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-medical-primary'
            : 'bg-medical-secondary border-2 border-medical-secondary'
        }`}
        aria-label={isUser ? 'User avatar' : 'Assistant avatar'}
      >
        {isUser ? (
          <svg
            className="w-5 h-5 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
            />
          </svg>
        ) : (
          <svg
            className="w-5 h-5 text-medical-accent"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        )}
      </div>

      {/* Message Content */}
      <div
        className={`flex-1 max-w-[80%] sm:max-w-[70%] ${
          isUser ? 'items-end' : 'items-start'
        } flex flex-col`}
      >
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm ${
            isUser
              ? 'bg-medical-primary dark:bg-medical-primary-dark text-white rounded-tr-sm'
              : 'bg-white dark:bg-slate-700 text-medical-text dark:text-medical-text-dark border border-medical-secondary/30 dark:border-slate-600 rounded-tl-sm'
          }`}
        >
          <p className="text-sm sm:text-base leading-relaxed whitespace-pre-wrap break-words">
            {message.text}
          </p>
        </div>

        {/* Timestamp and metadata */}
        <div
          className={`mt-1 text-xs text-medical-text/50 flex items-center space-x-2 ${
            isUser ? 'flex-row-reverse' : ''
          }`}
        >
          <span>
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {!isUser && message.confidence !== undefined && (
            <span className="text-medical-secondary">
              {Math.round(message.confidence * 100)}% confidence
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;

