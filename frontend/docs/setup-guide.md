# Frontend Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

**Note**: If `VITE_API_URL` is not set, it defaults to `http://localhost:8000`.

### 3. Start Development Server

```bash
npm run dev
```

The frontend will start on **http://localhost:3000** and should automatically open in your browser.

## Troubleshooting

### "localhost refused to connect"

If you see "localhost refused to connect" when trying to access the frontend:

1. **Check if the dev server is running:**
   ```bash
   # In PowerShell
   netstat -ano | findstr :3000
   ```

2. **Verify the port is not in use:**
   - If port 3000 is busy, Vite will try the next available port
   - Check the terminal output for the actual URL

3. **Restart the dev server:**
   ```bash
   # Stop the current server (Ctrl+C)
   npm run dev
   ```

### Frontend Loads But Can't Connect to Backend

1. **Check if backend is running:**
   ```bash
   # Test backend connection
   curl http://localhost:8000/
   # Should return: {"message":"MediMind API is running!"}
   ```

2. **Verify VITE_API_URL:**
   - Check browser console for: `🔗 API URL: http://localhost:8000`
   - If it shows a different URL, check your `.env` file

3. **Check CORS configuration:**
   - See `frontend/docs/cors-warning.md` for CORS setup
   - Backend must allow requests from `http://localhost:3000`

4. **Check browser console:**
   - Open DevTools (F12) → Console tab
   - Look for error messages with details about the failed request

### Environment Variable Not Working

1. **Restart the dev server** after creating/modifying `.env` file
2. **Check file location:** `.env` must be in `frontend/` directory (same level as `package.json`)
3. **Verify syntax:** No spaces around `=`, no quotes needed:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

### Port Already in Use

If port 3000 is already in use:

1. **Option 1:** Kill the process using port 3000:
   ```powershell
   # Find process ID
   netstat -ano | findstr :3000
   # Kill process (replace PID with actual process ID)
   taskkill /PID <PID> /F
   ```

2. **Option 2:** Use a different port:
   - Vite will automatically try the next available port
   - Or modify `vite.config.ts` to use a different port

## Development Workflow

1. **Start Backend** (in a separate terminal):
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

2. **Start Frontend** (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open Browser:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Debugging Tips

### Browser Console Logs

The frontend logs helpful information in development mode:

- `🔗 API URL:` - Shows which API URL is being used
- `📤 Sending request to:` - Shows the full request URL
- `✅ Received response:` - Shows the API response
- `❌ API Error:` - Shows detailed error information
- `🌐 Network Error` - Indicates backend connection issues

### Network Tab

1. Open DevTools (F12)
2. Go to Network tab
3. Send a message from the frontend
4. Look for the `/chat` request:
   - **Status 200**: Success
   - **Status 404**: Backend endpoint not found
   - **Status 500**: Backend server error
   - **CORS error**: Backend CORS not configured
   - **Failed/ERR_CONNECTION_REFUSED**: Backend not running

## Common Issues

### Issue: "Cannot GET /"

**Solution**: This is normal. The frontend is a Single Page Application (SPA). All routes are handled by React Router (if configured) or the root `/` route.

### Issue: White Screen / Blank Page

**Solution**:
1. Check browser console for JavaScript errors
2. Verify all dependencies are installed: `npm install`
3. Clear browser cache and hard refresh (Ctrl+Shift+R)

### Issue: Styles Not Loading

**Solution**:
1. Verify Tailwind CSS is configured correctly
2. Check `tailwind.config.js` and `postcss.config.js` exist
3. Restart the dev server

## Next Steps

- See `frontend/README.md` for full documentation
- See `frontend/docs/backend-endpoints-discovery.md` for API details
- See `frontend/docs/backend-not-running.md` if backend is down
- See `frontend/docs/cors-warning.md` for CORS issues

