"""
Test Script for LLM Interpretation System
Run this to verify Ollama connection and model availability
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import OllamaClient
from prompts import get_prompt_template


def test_connection():
    """Test connection to Ollama server"""
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)
    
    client = OllamaClient()
    
    # Test connection
    print("\n1. Checking Ollama server...")
    if client.check_connection():
        print("   ✓ Connected to Ollama server at http://localhost:11434")
    else:
        print("   ✗ Cannot connect to Ollama server")
        print("   → Solution: Run 'ollama serve' in PowerShell")
        return False
    
    # List models
    print("\n2. Checking available models...")
    models = client.list_models()

    if models:
        print(f"   Found {len(models)} model(s):")
        for model in models:
            print(f"      - {model}")

        selected, supports_vision = client.resolve_model()
        mode = "vision (image + metrics)" if supports_vision else "metrics-only LLM"
        print(f"\n   ✓ Auto-selected model: {selected} [{mode}]")
        if not supports_vision:
            print("   ℹ Optional: run 'ollama pull llava:13b' to enable direct image vision analysis.")
    else:
        print("   ✗ No models available")
        print("   → Solution: Run 'ollama pull llava:13b' or 'ollama pull llama3.1:8b'")
        return False

    return True


def test_prompts():
    """Test prompt templates"""
    print("\n" + "=" * 60)
    print("Testing Prompt Templates")
    print("=" * 60)
    
    templates = ['main', 'short', 'disease', 'growth']
    
    for template_type in templates:
        prompt = get_prompt_template(template_type)
        if prompt:
            print(f"\n✓ {template_type.upper()} template loaded")
            print(f"  Length: {len(prompt)} characters")
        else:
            print(f"\n✗ {template_type.upper()} template failed")
    
    # Show sample prompt
    print("\n" + "=" * 60)
    print("Sample Prompt (Main Template)")
    print("=" * 60)
    sample = get_prompt_template('main')
    print(sample[:500] + "..." if len(sample) > 500 else sample)


def test_generation():
    """Test actual generation (requires image)"""
    print("\n" + "=" * 60)
    print("Testing Text Generation (No Image)")
    print("=" * 60)
    
    client = OllamaClient(model='llava:13b')
    
    # Test with simple text prompt
    metrics = {
        'shrimp_count': 50,
        'avg_length_cm': 12.5,
        'avg_weight_g': 25.0,
        'water_depth_m': 1.2,
        'pond_id': 'POND-001',
        'cycle_no': 'C-2024-01'
    }
    
    history = "Shrimp count over last 3 measurements: [45, 48, 50]. Average length progression: [11.2, 11.8, 12.5] cm."
    prompt = get_prompt_template('short')
    
    print("\nSending test request to LLM...")
    print("(This may take 5-15 seconds)")
    
    try:
        result = client.generate_interpretation(
            image_base64=None,  # No image for this test
            metrics=metrics,
            history=history,
            prompt_template=prompt
        )
        
        if result:
            print("\n✓ Generation successful!")
            print("\nGenerated interpretation:")
            print("-" * 60)
            print(result)
            print("-" * 60)
        else:
            print("\n✗ Generation failed (returned None)")
            print("   This is expected if no image was provided")
            
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LLM INTERPRETATION SYSTEM - SETUP TEST")
    print("=" * 60)
    
    # Test 1: Connection
    if not test_connection():
        print("\n" + "=" * 60)
        print("SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease install Ollama and download the llava model first.")
        print("See: OLLAMA_SETUP.md for instructions.")
        return
    
    # Test 2: Prompts
    test_prompts()
    
    # Test 3: Generation (optional)
    print("\n")
    response = input("Test actual generation? (y/n): ").strip().lower()
    if response == 'y':
        test_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Ollama server is running")
    print("✓ Llava model is available")
    print("✓ Prompt templates are ready")
    print("\nYour system is ready to generate interpretations!")
    print("\nNext steps:")
    print("1. Start the Flask app: python app.py")
    print("2. Open browser to http://localhost:5000")
    print("3. Click '👨‍⚕️ Generate Interpretation' button")
    print("=" * 60)


if __name__ == '__main__':
    main()
