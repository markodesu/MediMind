import { render, screen } from '@testing-library/react';
import MessageBubble from '../MessageBubble';
import type { Message } from '../../lib/types';

describe('MessageBubble', () => {
  const userMessage: Message = {
    id: '1',
    role: 'user',
    text: 'Hello, I have a headache.',
    timestamp: new Date('2024-01-01T10:00:00'),
  };

  const assistantMessage: Message = {
    id: '2',
    role: 'assistant',
    text: 'I understand you have a headache. Can you describe the severity?',
    timestamp: new Date('2024-01-01T10:01:00'),
    confidence: 0.85,
    safe: true,
  };

  it('renders user message correctly', () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.getByText('Hello, I have a headache.')).toBeInTheDocument();
  });

  it('renders assistant message correctly', () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText(/I understand you have a headache/)).toBeInTheDocument();
  });

  it('displays confidence for assistant messages', () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText(/85% confidence/)).toBeInTheDocument();
  });

  it('does not display confidence for user messages', () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });
});

