# MediMind Frontend

A beautiful, production-quality React frontend for the MediMind AI Health Guidance Chatbot.

## Features

- 🎨 **Medical-themed Design**: Clean, soft medical colors with a professional yet friendly aesthetic
- 🌙 **Dark Mode**: Toggle between light and dark themes
- 📱 **Responsive**: Works seamlessly on both desktop and mobile devices
- ♿ **Accessible**: ARIA labels, proper contrast ratios, and keyboard navigation
- ⚡ **Fast**: Built with Vite for lightning-fast development and builds
- 🧪 **Tested**: Includes Jest and React Testing Library setup

## Tech Stack

- **React 18** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS** for styling
- **Jest** + **React Testing Library** for testing

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional):
```env
VITE_API_URL=http://localhost:8000
```

If `VITE_API_URL` is not set, it defaults to `http://localhost:8000`.

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### Building for Production

Build the production bundle:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

### Testing

Run tests:
```bash
npm test
```

Run tests in watch mode:
```bash
npm run test:watch
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ChatWindow.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── MessageList.tsx
│   │   ├── Composer.tsx
│   │   ├── Header.tsx
│   │   ├── LoadingDots.tsx
│   │   └── __tests__/       # Component tests
│   ├── pages/
│   │   └── ChatPage.tsx     # Main chat page
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   └── types.ts         # TypeScript types
│   ├── contexts/
│   │   └── ThemeContext.tsx # Dark mode context
│   ├── docs/
│   │   └── backend-endpoints-discovery.md
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
│   └── logo.svg
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Connecting to Backend

The frontend is configured to connect to the backend API running on `http://localhost:8000` by default.

### Backend Requirements

1. The backend must be running on port 8000 (or update `VITE_API_URL`)
2. CORS must be enabled on the backend to allow requests from `http://localhost:3000`
3. The backend should implement the `/chat` endpoint as described in `src/docs/backend-endpoints-discovery.md`

### API Endpoint

The frontend expects a POST endpoint at `/chat`:

**Request:**
```json
{
  "question": "I have a headache"
}
```

**Response:**
```json
{
  "answer": "I understand you have a headache...",
  "confidence": 0.85,
  "safe": true
}
```

## Design System

### Colors

- **Background**: `#F7FBFC` (soft ice-blue)
- **Primary**: `#4BA3C3` (blue-green medical tone)
- **Secondary**: `#9AD4D6` (light teal)
- **Accent**: `#2D6A7E` (deep medical blue)
- **Text**: `#083344` (very dark teal)

### Dark Mode

Dark mode uses adjusted color variants for better contrast and readability in low-light environments.

## Features in Detail

### Chat Interface

- **Message Bubbles**: User messages appear on the right (blue), assistant messages on the left (white with border)
- **Avatars**: Visual indicators for user and assistant messages
- **Timestamps**: Each message shows when it was sent
- **Confidence Scores**: Assistant messages display confidence percentages
- **Loading States**: Animated dots while waiting for responses
- **Auto-scroll**: Automatically scrolls to the latest message

### Input Composer

- **Auto-expanding**: Textarea grows as you type
- **Keyboard Shortcuts**: 
  - `Enter` to send
  - `Shift+Enter` for new line
- **Disabled State**: Prevents sending while loading

### Header

- **Logo**: Medical-themed icon
- **Title**: "MediMind" with subtitle
- **Dark Mode Toggle**: Switch between light and dark themes

## Important Notes

⚠️ **DO NOT MODIFY THE BACKEND**

- All development happens in the `frontend/` folder only
- If backend changes are needed, document them in `frontend/docs/backend-issue.md`
- Never modify files in the `backend/` directory

## Troubleshooting

### CORS Errors

If you see CORS errors, ensure the backend has CORS enabled for `http://localhost:3000`.

### API Connection Issues

1. Verify the backend is running on port 8000
2. Check the `VITE_API_URL` environment variable
3. Review the backend endpoint documentation in `src/docs/backend-endpoints-discovery.md`

### Build Errors

1. Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
2. Clear Vite cache: `rm -rf node_modules/.vite`
3. Check TypeScript errors: `npx tsc --noEmit`

## License

Part of the MediMind project for UCA.

