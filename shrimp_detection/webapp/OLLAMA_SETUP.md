# LLM Interpretation System Setup Guide

## Overview
This guide will help you set up the Ollama LLM server for automatic shrimp interpretation generation.

## Prerequisites
- Windows 10/11 with at least 8GB RAM (16GB recommended)
- NVIDIA GPU with 4GB+ VRAM (optional but recommended for faster inference)
- Python 3.8+ installed

---

## Step 1: Install Ollama for Windows

### Download and Install
1. Visit: https://ollama.com/download
2. Download the Windows version
3. Run the installer and follow the prompts
4. Ollama will install and run as a background service

### Verify Installation
Open PowerShell and run:
```powershell
ollama --version
```

You should see the version number if installed correctly.

---

## Step 2: Download the Vision Model

The system uses **Llava 1.6** (Large Language and Vision Assistant) for image analysis.

### Pull the Model
Open PowerShell and run:
```powershell
ollama pull llava:13b
```

This will download the 13B parameter model (~7.5GB). Wait for the download to complete.

**Alternative models** (if you want to try different options):
- Smaller/faster: `ollama pull llava:7b`
- Better quality: `ollama pull bakllava`
- Latest version: `ollama pull llava`

### Verify Model Download
```powershell
ollama list
```

You should see `llava:13b` in the list.

---

## Step 3: Start Ollama Server

Ollama typically runs automatically as a background service. To verify it's running:

### Check if Ollama is Running
```powershell
ollama list
```

If you get a connection error, start the server manually:
```powershell
ollama serve
```

Keep this PowerShell window open while using the webapp.

### Test the API
Open another PowerShell window and test:
```powershell
curl http://localhost:11434/api/tags
```

You should see a JSON response with available models.

---

## Step 4: Install Python Dependencies

Navigate to the webapp directory and install requirements:

```powershell
cd F:\YOLOV8_SHRIMP\shrimp_detection\webapp
pip install -r requirements.txt
```

This will install the `requests` library needed for Ollama communication.

---

## Step 5: Test the LLM Integration

### Quick Test with Ollama Directly
Test the model directly from command line:
```powershell
ollama run llava:13b "What is aquaculture?"
```

### Test with Image (Optional)
Create a test script to verify the integration:

```python
# test_llm.py
from webapp.llm_client import OllamaClient

client = OllamaClient()

# Check connection
if client.check_connection():
    print("✓ Connected to Ollama server")
    print("Available models:", client.list_models())
else:
    print("✗ Cannot connect to Ollama")
```

---

## Step 6: Run the Web Application

Start the Flask app:
```powershell
cd F:\YOLOV8_SHRIMP\shrimp_detection\webapp
python app.py
```

The app will start on `http://localhost:5000`

---

## Step 7: Use the Interpretation Feature

1. Open your browser to `http://localhost:5000`
2. Wait for images to load
3. Click the **"👨‍⚕️ Generate Interpretation"** button
4. Wait 5-15 seconds for the LLM to analyze the image
5. The interpretation will appear in the text box automatically

---

## Troubleshooting

### Issue: "Cannot connect to Ollama server"
**Solution:**
```powershell
# Start Ollama server
ollama serve
```

### Issue: "Model not found"
**Solution:**
```powershell
# Re-download the model
ollama pull llava:13b
```

### Issue: Slow generation (>30 seconds)
**Solutions:**
- Use smaller model: `ollama pull llava:7b` then update `app.py` line with `model='llava:7b'`
- Close other applications to free up RAM
- Ensure GPU acceleration is enabled (NVIDIA GPUs supported)

### Issue: Out of memory error
**Solutions:**
- Use smaller model (7b instead of 13b)
- Increase virtual memory (Windows page file)
- Close other applications

### Issue: Interpretation returns fallback message
**Causes:**
- Ollama server not responding
- Model not loaded properly
- Timeout (increase timeout in `llm_client.py` line 56)

**Solution:**
Check Ollama logs:
```powershell
ollama serve --debug
```

---

## Configuration Options

### Change Model
Edit `shrimp_detection/webapp/app.py` line ~260:
```python
model='llava:13b'  # Change to 'llava:7b' or 'bakllava'
```

### Change Timeout
Edit `shrimp_detection/webapp/llm_client.py` line 56:
```python
timeout=60  # Increase to 90 or 120 for slower systems
```

### Use Different Prompt Template
In the frontend JavaScript (index.html), change:
```javascript
template_type: 'main'  // Options: 'main', 'short', 'disease', 'growth'
```

---

## Performance Tips

1. **First Run**: The first interpretation may be slow as the model loads
2. **Subsequent Runs**: Should be faster (5-15 seconds typical)
3. **GPU Acceleration**: Ensure NVIDIA drivers are updated
4. **RAM**: Close browser tabs and other apps if experiencing slowness

---

## How It Works

1. **Image Selection**: System fetches latest AI-processed shrimp image from database
2. **Data Collection**: Gathers metrics (count, length, weight, pond conditions)
3. **Historical Analysis**: Queries last 5 detections for trend analysis
4. **LLM Processing**: Sends image + metrics to Ollama with expert prompt
5. **Vision Analysis**: Llava model analyzes the shrimp image visually
6. **Text Generation**: LLM generates 3-4 sentence interpretation
7. **Display**: Result appears in the interpretation textbox

---

## Features

### What the LLM Analyzes:
- ✅ Visual appearance of shrimp (color, shell condition)
- ✅ Disease symptoms (spots, discoloration, deformities)
- ✅ Size estimation relative to environment
- ✅ Growth metrics (count, length, weight trends)
- ✅ Environmental factors (water depth, pond conditions)

### Output Includes:
- ✅ Growth stage assessment
- ✅ Health indicators and disease risk
- ✅ Specific recommendations for feeding/water quality
- ✅ Actionable aquaculture management advice

---

## Next Steps

### Optional Enhancements:
1. **Fine-tune Model**: Collect shrimp disease data for custom fine-tuning
2. **Multiple Models**: Add dropdown to select different LLM models
3. **Cache Results**: Store interpretations to avoid regenerating
4. **Batch Analysis**: Analyze multiple images at once
5. **Export Reports**: Generate PDF reports with interpretations

### Maintenance:
- Update Ollama: `ollama update`
- Update model: `ollama pull llava:13b` (re-pull for updates)
- Monitor disk space (models take ~7-15GB)

---

## Support

For issues with:
- **Ollama installation**: https://ollama.ai/help
- **Llava model**: https://ollama.ai/library/llava
- **This system**: Check logs in `shrimp_detection/webapp/` folder

---

## Quick Reference Commands

```powershell
# Start Ollama
ollama serve

# List models
ollama list

# Download model
ollama pull llava:13b

# Test model
ollama run llava:13b "Hello"

# Remove model
ollama rm llava:13b

# View Ollama logs
ollama serve --debug
```

---

**Ready to use!** 🎉

Your shrimp monitoring system now has AI-powered interpretation capabilities!
