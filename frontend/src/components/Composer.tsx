import React, { useState, useRef, useEffect } from 'react';

interface ComposerProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

const Composer: React.FC<ComposerProps> = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-medical-secondary/20 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-3 sm:py-4"
    >
      <div className="max-w-3xl mx-auto flex items-end space-x-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your symptoms or ask a medical question…"
            disabled={disabled}
            rows={1}
            className="w-full px-4 py-3 pr-12 bg-medical-bg dark:bg-slate-700 border border-medical-secondary/30 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-medical-primary dark:focus:ring-medical-primary-dark focus:border-transparent resize-none text-sm sm:text-base text-medical-text dark:text-medical-text-dark placeholder:text-medical-text/50 dark:placeholder:text-medical-text-dark/50 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ minHeight: '48px', maxHeight: '200px' }}
            aria-label="Message input"
          />
          <div className="absolute bottom-2 right-2 text-xs text-medical-text/40 dark:text-medical-text-dark/40">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className="flex-shrink-0 px-6 py-3 bg-gradient-to-r from-medical-primary to-medical-accent dark:from-medical-primary-dark dark:to-medical-accent-dark text-white rounded-xl font-medium shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-md focus:outline-none focus:ring-2 focus:ring-medical-primary dark:focus:ring-medical-primary-dark focus:ring-offset-2"
          aria-label="Send message"
        >
          <svg
            className="w-5 h-5 sm:w-6 sm:h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </div>
    </form>
  );
};

export default Composer;

