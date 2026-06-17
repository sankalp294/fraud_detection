# Docker Deployment Guide for Fraud Detection Project

A complete beginner's guide to containerizing and deploying the fraud detection system using Docker.

## Table of Contents

1. [What is Docker?](#what-is-docker)
2. [Why Use Docker?](#why-use-docker)
3. [Installation](#installation)
4. [Understanding the Files](#understanding-the-files)
5. [Building the Docker Image](#building-the-docker-image)
6. [Running with Docker](#running-with-docker)
7. [Running with Docker Compose](#running-with-docker-compose)
8. [Deployment Options](#deployment-options)
9. [Troubleshooting](#troubleshooting)

---

## What is Docker?

Docker is a **containerization platform** that packages your entire application (code, dependencies, environment) into a single unit called a **container**.

### Key Concepts:

- **Image**: A blueprint/template for creating containers (like a recipe)
- **Container**: A running instance of an image (like a cooked dish from the recipe)
- **Dockerfile**: Instructions to build an image (like cooking instructions)
- **Docker Hub**: Repository where you can share/download images

### Analogy:

Think of Docker like shipping containers:
- **Dockerfile** = Container factory specifications
- **Image** = An empty container ready to be filled
- **Container** = Container loaded with your application
- **Docker Host** = Ship that carries the container anywhere

---

## Why Use Docker?

✅ **Consistency**: Works the same on your computer, your coworker's, and the server
✅ **Isolation**: Each container is independent, no dependency conflicts
✅ **Easy Deployment**: Ship your app anywhere without reinstalling dependencies
✅ **Scalability**: Run multiple containers to handle more users
✅ **Version Control**: Track different versions of your app as images
✅ **Development Environment**: Everyone uses the exact same setup

---

## Installation

### Windows

1. **Download Docker Desktop**
   - Go to https://www.docker.com/products/docker-desktop
   - Click "Download for Windows"
   - Choose your Windows version (10 or 11)

2. **Install Docker Desktop**
   - Run the installer
   - Follow the setup wizard
   - Restart your computer when prompted

3. **Verify Installation**
   ```bash
   docker --version
   docker run hello-world
   ```

### macOS

1. **Download Docker Desktop**
   - Go to https://www.docker.com/products/docker-desktop
   - Choose Intel or Apple Silicon

2. **Install**
   - Open the .dmg file
   - Drag Docker to Applications folder

3. **Verify**
   ```bash
   docker --version
   ```

### Linux

```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
docker --version
```

---

## Understanding the Files

### 1. **Dockerfile** - The Recipe

```dockerfile
FROM python:3.10-slim
```
- **What**: Start with Python 3.10 (minimal version)
- **Why**: We need Python to run our code

```dockerfile
WORKDIR /app
```
- **What**: Create a directory `/app` inside the container
- **Why**: All our code will live here

```dockerfile
ENV PYTHONUNBUFFERED=1
```
- **What**: Don't buffer Python output
- **Why**: See error messages immediately in logs

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
- **What**: Copy your dependencies and install them
- **Why**: Container needs the same libraries you use locally

```dockerfile
COPY . .
```
- **What**: Copy all your project files into the container
- **Why**: Container needs your code to run

```dockerfile
EXPOSE 5000
```
- **What**: Container will listen on port 5000
- **Why**: Flask API runs on this port

```dockerfile
CMD ["python", "Inference/app.py"]
```
- **What**: Default command when container starts
- **Why**: Automatically runs your Flask API

---

### 2. **docker-compose.yml** - The Orchestrator

Docker Compose manages multiple containers and their settings.

```yaml
version: '3.8'
```
- Compose file format version

```yaml
services:
  fraud-detection-api:
```
- Define a service called "fraud-detection-api"

```yaml
build:
  context: .
  dockerfile: Dockerfile
```
- **context**: Where to find files (current directory `.`)
- **dockerfile**: Which Dockerfile to use

```yaml
ports:
  - "5000:5000"
```
- **Left side** (5000): Port on your computer
- **Right side** (5000): Port inside container
- This means: Access the container at `localhost:5000`

```yaml
environment:
  - FLASK_APP=Inference/app.py
  - FLASK_ENV=production
```
- Variables available inside the container
- Configure Flask to run in production mode

```yaml
volumes:
  - .:/app
  - ./model_training/models:/app/model_training/models
```
- **Mount folders** from your computer into the container
- Changes you make locally reflect immediately
- Keep trained models persistent

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/predict || exit 1"]
  interval: 30s
```
- Automatically check if the API is healthy every 30 seconds

---

### 3. **.dockerignore** - The Exclude List

Similar to `.gitignore`, this tells Docker which files NOT to copy into the container.

**Why?** Skip unnecessary files (cache, logs, virtual environments) to make the image smaller and build faster.

---

## Building the Docker Image

### Step 1: Open Terminal/PowerShell

Navigate to your project folder:
```bash
cd C:\Users\Lenovo\Desktop\combat\fraud_detection
```

### Step 2: Build the Image

```bash
docker build -t fraud-detection:latest .
```

**Breakdown:**
- `docker build`: Tell Docker to build an image
- `-t fraud-detection:latest`: Tag the image (name:version)
- `.`: Use Dockerfile in current directory

**What Happens:**
1. Docker reads the Dockerfile
2. Downloads Python 3.10 base image
3. Installs your dependencies from requirements.txt
4. Copies your code
5. Creates the image

**Time:** First build takes 2-5 minutes (depends on dependencies)

**Verify:**
```bash
docker images
```
You should see `fraud-detection` listed.

---

## Running with Docker

### Method 1: Manual Docker Commands

**Start the container:**
```bash
docker run -p 5000:5000 -v "C:\Users\Lenovo\Desktop\combat\fraud_detection:/app" fraud-detection:latest
```

**Breakdown:**
- `docker run`: Start a new container
- `-p 5000:5000`: Map ports
- `-v`: Mount volume (your code folder)
- `fraud-detection:latest`: Use this image

**Test the API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"policy_id":"POL123","insured_id":"INS456",...}'
```

**Stop the container:**
```bash
docker stop <container_id>
```

### Method 2: Using Docker Compose (Easier!)

**Start:**
```bash
docker-compose up
```

**What happens:**
1. Builds the image (if not built)
2. Starts the container
3. Shows real-time logs

**Stop:**
```bash
docker-compose down
```

**Run in background:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f
```

---

## Deployment Options

### Option 1: Local Development (What You've Just Done)
- Run containers on your laptop
- Use for testing and development

### Option 2: Cloud Deployment (AWS, Google Cloud, Azure)

**AWS EC2:**
1. Launch an EC2 instance
2. Install Docker on the instance
3. Upload your code
4. Run `docker-compose up`

**AWS ECS (Managed Service):**
1. Push image to AWS ECR (Elastic Container Registry)
2. Create ECS service
3. AWS manages scaling and updates

**Heroku:**
```bash
heroku login
heroku create fraud-detection-api
heroku container:push web
heroku container:release web
heroku logs --tail
```

**Docker Hub:**
```bash
docker tag fraud-detection:latest yourusername/fraud-detection:latest
docker push yourusername/fraud-detection:latest
```

### Option 3: Kubernetes (Advanced)
- For large-scale deployments
- Automatic scaling, load balancing
- Use later when you need this complexity

---

## Quick Start Commands

### First Time Setup:
```bash
# 1. Build the image
docker build -t fraud-detection:latest .

# 2. Start with docker-compose
docker-compose up
```

### Daily Usage:
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Rebuild After Code Changes:
```bash
# Stop
docker-compose down

# Rebuild
docker build -t fraud-detection:latest .

# Restart
docker-compose up
```

---

## Troubleshooting

### Issue: "Port 5000 already in use"
```bash
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or use a different port in docker-compose.yml:
# ports:
#   - "8000:5000"
```

### Issue: "Image not found"
```bash
# Make sure you built it:
docker build -t fraud-detection:latest .

# List images:
docker images
```

### Issue: "Container exits immediately"
```bash
# Check the logs:
docker-compose logs

# Run bash inside container to debug:
docker run -it fraud-detection:latest /bin/bash
```

### Issue: "Connection refused"
```bash
# Check if container is running:
docker ps

# Container might be crashed, check logs:
docker-compose logs
```

### Issue: "Permission denied" (Linux)
```bash
# Run with sudo:
sudo docker-compose up

# OR add user to docker group:
sudo usermod -aG docker $USER
```

---

## Next Steps

1. **Test your API**: Make requests to `http://localhost:5000/predict`
2. **Push to Docker Hub**: Share your image online
3. **Deploy to Cloud**: Move to AWS, Google Cloud, or Azure
4. **Setup CI/CD**: Automate building and pushing images
5. **Monitor**: Use tools like Prometheus to monitor container health

---

## Useful Docker Commands

| Command | Description |
|---------|-------------|
| `docker build -t name:tag .` | Build an image |
| `docker run image:tag` | Run a container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker logs container_id` | View container logs |
| `docker exec -it container_id bash` | Enter container shell |
| `docker stop container_id` | Stop a container |
| `docker remove container_id` | Delete a container |
| `docker images` | List all images |
| `docker rmi image_id` | Delete an image |
| `docker-compose up` | Start services |
| `docker-compose down` | Stop services |
| `docker-compose logs` | View logs |

---

## Summary

You've now containerized your fraud detection project! 🎉

- **Dockerfile**: Defines how to build your application
- **docker-compose.yml**: Manages running and configuration
- **.dockerignore**: Excludes unnecessary files

Your application is now:
- ✅ Portable (runs anywhere Docker is installed)
- ✅ Isolated (no conflicts with other apps)
- ✅ Scalable (can run multiple containers)
- ✅ Reproducible (same behavior everywhere)

Next: Deploy to the cloud and scale! 🚀
