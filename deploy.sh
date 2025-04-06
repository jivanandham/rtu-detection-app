#!/bin/bash

# Deployment script for production environment

# Exit on error
set -e

# Configuration
ENV_FILE=".env"
FRONTEND_DIR="frontend"
BACKEND_DIR="backend"

# Create environment files if they don't exist
cp ${ENV_FILE}.example ${ENV_FILE} 2>/dev/null || echo "Environment file already exists"

cd ${FRONTEND_DIR}

echo "Building frontend..."
npm install
cp .env.example .env
npm run build

cd ../${BACKEND_DIR}

echo "Setting up backend..."
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Initialize database
python -m app.init_db

echo "Starting backend in production mode..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
