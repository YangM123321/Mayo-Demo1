# Creates .venv and installs Python deps (Windows / PowerShell)
# Run from project root: .\scripts\setup.ps1

# Fail on errors in this script
Continue = "Stop"

# Helper: resolve python command (python or py)
function Get-Python {
  if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
  elseif (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
  else {
    throw "Python not found. Install Python 3.11+ and ensure 'python' or 'py' is on PATH."
  }
}

 = Get-Python
Write-Host "Using Python command: "

# Create venv
&  -m venv .venv

# Activate venv for this session
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r .\requirements.txt

# Sanity checks
Write-Host "Python:" (python --version)
Write-Host "Pip list (top few):"
pip list | Select-String -Pattern "fastapi|uvicorn|pandas|numpy|scikit|pyarrow|pydantic|requests|python-dotenv"

Write-Host "
Setup complete. Activate later with:  .\.venv\Scripts\Activate.ps1"
