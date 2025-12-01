import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Composer from '../Composer';

describe('Composer', () => {
  const mockOnSendMessage = jest.fn();

  beforeEach(() => {
    mockOnSendMessage.mockClear();
  });

  it('renders input field and send button', () => {
    render(<Composer onSendMessage={mockOnSendMessage} />);
    expect(screen.getByPlaceholderText(/Describe your symptoms/)).toBeInTheDocument();
    expect(screen.getByLabelText('Send message')).toBeInTheDocument();
  });

  it('calls onSendMessage when form is submitted', async () => {
    const user = userEvent.setup();
    render(<Composer onSendMessage={mockOnSendMessage} />);
    
    const input = screen.getByPlaceholderText(/Describe your symptoms/);
    const button = screen.getByLabelText('Send message');

    await user.type(input, 'I have a headache');
    await user.click(button);

    expect(mockOnSendMessage).toHaveBeenCalledWith('I have a headache');
  });

  it('sends message on Enter key press', async () => {
    const user = userEvent.setup();
    render(<Composer onSendMessage={mockOnSendMessage} />);
    
    const input = screen.getByPlaceholderText(/Describe your symptoms/);
    await user.type(input, 'Test message{Enter}');

    await waitFor(() => {
      expect(mockOnSendMessage).toHaveBeenCalledWith('Test message');
    });
  });

  it('does not send message on Shift+Enter', async () => {
    const user = userEvent.setup();
    render(<Composer onSendMessage={mockOnSendMessage} />);
    
    const input = screen.getByPlaceholderText(/Describe your symptoms/);
    await user.type(input, 'Test message{Shift>}{Enter}{/Shift}');

    expect(mockOnSendMessage).not.toHaveBeenCalled();
  });

  it('disables input when disabled prop is true', () => {
    render(<Composer onSendMessage={mockOnSendMessage} disabled={true} />);
    const input = screen.getByPlaceholderText(/Describe your symptoms/);
    expect(input).toBeDisabled();
  });

  it('clears input after sending message', async () => {
    const user = userEvent.setup();
    render(<Composer onSendMessage={mockOnSendMessage} />);
    
    const input = screen.getByPlaceholderText(/Describe your symptoms/);
    await user.type(input, 'Test message');
    await user.click(screen.getByLabelText('Send message'));

    await waitFor(() => {
      expect(input).toHaveValue('');
    });
  });
});

