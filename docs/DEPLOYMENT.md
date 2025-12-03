# MediMind Deployment Guide

Complete guide for deploying MediMind to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Backend Deployment](#backend-deployment)
4. [Frontend Deployment](#frontend-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Production Considerations](#production-considerations)
7. [Platform-Specific Guides](#platform-specific-guides)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ Python 3.10+ installed
- ✅ Node.js 18+ and npm installed
- ✅ GPU with 8GB+ VRAM (for model inference) OR CPU with sufficient RAM
- ✅ All environment variables configured
- ✅ Model files downloaded (if using fine-tuned model)

---

## Deployment Options

### Option 1: Single Server Deployment (Recommended for Small Scale)

Deploy both backend and frontend on the same server:
- Backend: FastAPI on port 8000
- Frontend: Built static files served by backend or nginx

### Option 2: Separate Services

Deploy backend and frontend separately:
- Backend: FastAPI API server
- Frontend: Static files on CDN or separate web server

### Option 3: Containerized Deployment

Use Docker containers for easy deployment:
- Backend container
- Frontend container
- Easy scaling and management

---

## Backend Deployment

### Step 1: Prepare Backend Environment

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install production server (Gunicorn with Uvicorn workers)
pip install gunicorn
```

### Step 2: Configure Environment Variables

Create `.env` file in `backend/` directory:

```env
# Model configuration
MODEL_NAME=microsoft/phi-2
LORA_MODEL_PATH=  # Optional: path to fine-tuned model
MAX_NEW_TOKENS=180
CONFIDENCE_THRESHOLD=0.6

# UCA Medical Services (REQUIRED - set your actual values)
UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
UCA_MEDICAL_PHONE=+996XXXXXXXXX  # Set your actual phone number
UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM

# API configuration
API_TITLE=MediMind API - University of Central Asia
API_VERSION=1.0.0

# Production settings
DEBUG=False
LOG_LEVEL=INFO
```

### Step 3: Test Backend Locally

```bash
# Test with development server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test with production server
gunicorn app.main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Step 4: Production Deployment

#### Using Gunicorn (Recommended)

```bash
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

#### Using Uvicorn (Alternative)

```bash
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info \
    --no-access-log
```

### Step 5: Run as Systemd Service (Linux)

Create `/etc/systemd/system/medimind-backend.service`:

```ini
[Unit]
Description=MediMind Backend API
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/path/to/MediMind/backend
Environment="PATH=/path/to/MediMind/backend/venv/bin"
ExecStart=/path/to/MediMind/backend/venv/bin/gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable medimind-backend
sudo systemctl start medimind-backend
sudo systemctl status medimind-backend
```

---

## Frontend Deployment

### Step 1: Build Frontend

```bash
cd frontend

# Install dependencies
npm install

# Build for production
npm run build
```

This creates a `dist/` folder with optimized static files.

### Step 2: Update API URL (if needed)

If your backend is on a different domain, update `frontend/src/lib/api.ts`:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://your-backend-domain:8000';
```

Or set environment variable:
```bash
VITE_API_URL=http://your-backend-domain:8000 npm run build
```

### Step 3: Serve Static Files

#### Option A: Serve with Backend (Simple)

Add to `backend/app/main.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Serve frontend static files
if os.path.exists("../frontend/dist"):
    app.mount("/static", StaticFiles(directory="../frontend/dist"), name="static")
    
    @app.get("/")
    async def read_root():
        return FileResponse("../frontend/dist/index.html")
```

#### Option B: Use Nginx (Recommended for Production)

Create `/etc/nginx/sites-available/medimind`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend static files
    root /path/to/MediMind/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend API proxy
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    # WebSocket support (if needed)
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/medimind /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Option C: Use Node.js Server (Alternative)

```bash
npm install -g serve
serve -s dist -l 3000
```

---

## Environment Configuration

### Backend Environment Variables

Required variables in `backend/.env`:

```env
# Required
UCA_MEDICAL_PHONE=+996XXXXXXXXX  # Set actual phone number
UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM

# Optional (has defaults)
MODEL_NAME=microsoft/phi-2
CONFIDENCE_THRESHOLD=0.6
MAX_NEW_TOKENS=180
```

### Frontend Environment Variables

Create `frontend/.env.production`:

```env
VITE_API_URL=http://your-backend-domain:8000
```

Build with:
```bash
npm run build
```

---

## Production Considerations

### 1. Security

- ✅ **Never commit `.env` files** - Already in `.gitignore`
- ✅ **Use HTTPS** - Set up SSL/TLS certificates (Let's Encrypt)
- ✅ **CORS Configuration** - Update CORS settings in `backend/app/main.py`
- ✅ **Rate Limiting** - Consider adding rate limiting middleware
- ✅ **Input Validation** - Already handled by Pydantic schemas

### 2. Performance

- **Model Loading**: Model loads on first request (lazy loading)
- **Caching**: Consider caching common responses
- **GPU Memory**: Monitor GPU memory usage
- **Worker Processes**: Adjust Gunicorn workers based on CPU cores

### 3. Monitoring

- **Logs**: Check logs in `backend/logs/` directory
- **Health Checks**: Use `/api/v1/` endpoint for health monitoring
- **Error Tracking**: Consider adding Sentry or similar

### 4. Scaling

- **Horizontal Scaling**: Run multiple backend instances behind load balancer
- **Model Caching**: Use shared model storage or model server
- **Database**: Add database for conversation history (if needed)

---

## Platform-Specific Guides

### Deploy to Ubuntu/Debian Server

1. **Install Dependencies**:
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv nodejs npm nginx
```

2. **Clone Repository**:
```bash
git clone https://github.com/yourusername/MediMind.git
cd MediMind
```

3. **Setup Backend**:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt gunicorn
# Create .env file with your settings
```

4. **Setup Frontend**:
```bash
cd ../frontend
npm install
npm run build
```

5. **Configure Systemd and Nginx** (see sections above)

### Deploy with Docker

#### Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

#### Frontend Dockerfile

Create `frontend/Dockerfile`:

```dockerfile
# Build stage
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - UCA_MEDICAL_PHONE=${UCA_MEDICAL_PHONE}
      - UCA_MEDICAL_CONTACT_NAME=${UCA_MEDICAL_CONTACT_NAME}
      - UCA_MEDICAL_LOCATION=${UCA_MEDICAL_LOCATION}
    volumes:
      - ./backend:/app
      - model_cache:/root/.cache/huggingface
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  model_cache:
```

Run:
```bash
docker-compose up -d
```

### Deploy to Cloud Platforms

#### Heroku

1. **Backend**: Add `Procfile`:
```
web: gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

2. **Frontend**: Use Heroku static buildpack or deploy separately

#### AWS EC2

1. Launch EC2 instance (Ubuntu)
2. Follow Ubuntu deployment guide above
3. Configure security groups for ports 80, 443, 8000
4. Set up Elastic IP

#### Google Cloud Platform

1. Use Cloud Run for backend (containerized)
2. Use Cloud Storage + Cloud CDN for frontend
3. Or use Compute Engine (similar to EC2)

#### DigitalOcean

1. Create Droplet (Ubuntu)
2. Follow Ubuntu deployment guide
3. Use App Platform for easier deployment

---

## Quick Deployment Checklist

- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] `.env` file configured with actual phone number
- [ ] Backend tested locally
- [ ] Frontend built (`npm run build`)
- [ ] Production server configured (Gunicorn/systemd)
- [ ] Nginx configured (if using)
- [ ] SSL certificate installed (for HTTPS)
- [ ] Firewall configured
- [ ] Monitoring set up
- [ ] Backups configured

---

## Troubleshooting

### Backend won't start

- Check `.env` file exists and has required variables
- Verify Python version (3.10+)
- Check port 8000 is not in use: `lsof -i :8000`
- Check logs: `journalctl -u medimind-backend -f`

### Frontend can't connect to backend

- Check CORS settings in `backend/app/main.py`
- Verify API URL in frontend build
- Check firewall rules
- Test backend directly: `curl http://localhost:8000/api/v1/`

### Model loading issues

- Check GPU availability: `nvidia-smi`
- Verify model files downloaded
- Check disk space for model cache
- Increase timeout in Gunicorn config

---

**Last Updated:** 2025-12-03

