# ================================
# start-jupyter.ps1
# Auto-build and run Jupyter in Docker
# ================================

Write-Host "🔧 Starting Mayo-Demo Jupyter environment..." -ForegroundColor Cyan

# Move to this script's folder
Set-Location $PSScriptRoot

# Stop and remove any previous container with the same name
if (docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch "mayo1-nb") {
    Write-Host "🧹 Removing old container 'mayo1-nb'..." -ForegroundColor Yellow
    docker rm -f mayo1-nb | Out-Null
}

# Build the latest image
Write-Host "🐳 Building Docker image (mayo-demo1)..." -ForegroundColor Cyan
docker build -t mayo-demo1 . | Out-Host

# Run container with Jupyter mode and volume persistence
Write-Host "🚀 Launching Jupyter Notebook container..." -ForegroundColor Cyan
docker run -d `
  --name mayo1-nb `
  -p 8888:8888 `
  -e MODE=jupyter `
  -v "${PWD}:/app" `
  mayo-demo1 | Out-Host

# Show status
Start-Sleep -Seconds 3
Write-Host "`n📋 Current containers:" -ForegroundColor Green
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Show Jupyter logs (optional)
Write-Host "`n📜 Fetching Jupyter logs (press Ctrl+C to stop watching)..." -ForegroundColor Yellow
docker logs -f mayo1-nb
