#!/usr/bin/env python3
"""
Demo script for GPT-2 Text Generation
This script demonstrates the capabilities of the text generation system.
"""

import torch
from gpt2_text_generator import GPT2TextGenerator
import time
import os

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_basic_generation():
    """Demonstrate basic text generation."""
    print_section("Basic Text Generation")
    
    generator = GPT2TextGenerator()
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Python programming",
        "Natural language processing"
    ]
    
    for prompt in prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        print("⏳ Generating...")
        
        start_time = time.time()
        generated_texts = generator.generate_text(
            prompt=prompt,
            max_length=60,
            temperature=0.7,
            num_return_sequences=1
        )
        end_time = time.time()
        
        print(f"✨ Generated: {generated_texts[0]}")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")

def demo_parameter_variation():
    """Demonstrate how different parameters affect generation."""
    print_section("Parameter Variation Demo")
    
    generator = GPT2TextGenerator()
    prompt = "The future of technology"
    
    # Test different temperatures
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        generated_texts = generator.generate_text(
            prompt=prompt,
            max_length=50,
            temperature=temp,
            num_return_sequences=1
        )
        print(f"Generated: {generated_texts[0]}")

def demo_multiple_sequences():
    """Demonstrate generating multiple sequences."""
    print_section("Multiple Sequence Generation")
    
    generator = GPT2TextGenerator()
    prompt = "Innovation in artificial intelligence"
    
    print(f"🎯 Prompt: '{prompt}'")
    print("Generating 3 different sequences...")
    
    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=70,
        temperature=0.8,
        num_return_sequences=3
    )
    
    for i, text in enumerate(generated_texts, 1):
        print(f"\n📄 Sequence {i}: {text}")

def demo_fine_tuning_process():
    """Demonstrate the fine-tuning process."""
    print_section("Fine-tuning Process Demo")
    
    # Check if fine-tuned model exists and has proper files
    model_path = "fine_tuned_model"
    model_files_exist = (
        os.path.exists(model_path) and 
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) and
        os.path.exists(os.path.join(model_path, "config.json"))
    )
    
    if model_files_exist:
        print("✅ Fine-tuned model found! Loading...")
        try:
            generator = GPT2TextGenerator()
            generator.load_fine_tuned_model(model_path)
            
            # Test with fine-tuned model
            prompt = "Deep learning models"
            print(f"\n🎯 Testing fine-tuned model with: '{prompt}'")
            
            generated_texts = generator.generate_text(
                prompt=prompt,
                max_length=60,
                temperature=0.7
            )
            
            print(f"✨ Fine-tuned output: {generated_texts[0]}")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("Using pre-trained model instead...")
            demo_with_pretrained_model()
    else:
        print("ℹ️  No fine-tuned model found.")
        print("💡 To create a fine-tuned model, run: python gpt2_text_generator.py")
        print("Using pre-trained model for demonstration...")
        demo_with_pretrained_model()

def demo_with_pretrained_model():
    """Demonstrate with pre-trained model."""
    generator = GPT2TextGenerator()
    
    # Test with pre-trained model
    prompt = "Deep learning models"
    print(f"\n🎯 Testing pre-trained model with: '{prompt}'")
    
    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=60,
        temperature=0.7
    )
    
    print(f"✨ Pre-trained output: {generated_texts[0]}")

def demo_model_evaluation():
    """Demonstrate model evaluation capabilities."""
    print_section("Model Evaluation Demo")
    
    generator = GPT2TextGenerator()
    
    # Test prompts for evaluation
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Python programming"
    ]
    
    print("🧪 Evaluating model performance...")
    evaluation_results = generator.evaluate_model(test_prompts)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Average text length: {evaluation_results['average_length']:.2f} words")
    print(f"  Average perplexity: {evaluation_results['average_perplexity']:.2f}")
    
    print(f"\n📝 Generated examples:")
    for i, (prompt, text) in enumerate(zip(evaluation_results['prompts'], evaluation_results['generated_texts'])):
        print(f"  {i+1}. Prompt: '{prompt}'")
        print(f"     Generated: {text}")

def demo_interactive_features():
    """Demonstrate interactive features."""
    print_section("Interactive Features Demo")
    
    print("🎮 The interactive interface provides these features:")
    print("  • Real-time text generation")
    print("  • Adjustable parameters (temperature, top-k, top-p)")
    print("  • Multiple sequence generation")
    print("  • Example prompts")
    print("  • Settings management")
    
    print("\n💡 To try the interactive interface, run:")
    print("   python interactive_generator.py")

def main():
    """Main demo function."""
    print_header("GPT-2 Text Generation Demo")
    
    print("🚀 Welcome to the GPT-2 Text Generation Demo!")
    print("This demo showcases the capabilities of our text generation system.")
    
    try:
        # Run all demos
        demo_basic_generation()
        demo_parameter_variation()
        demo_multiple_sequences()
        demo_fine_tuning_process()
        demo_model_evaluation()
        demo_interactive_features()
        
        print_header("Demo Completed Successfully!")
        print("🎉 All demos completed successfully!")
        print("\n📚 What you've seen:")
        print("  ✅ Basic text generation from prompts")
        print("  ✅ Parameter variation effects")
        print("  ✅ Multiple sequence generation")
        print("  ✅ Fine-tuned model capabilities")
        print("  ✅ Model evaluation metrics")
        print("  ✅ Interactive interface features")
        
        print("\n🔧 Next steps:")
        print("  1. Try the interactive interface: python interactive_generator.py")
        print("  2. Fine-tune on your own data: python gpt2_text_generator.py")
        print("  3. Experiment with different parameters")
        print("  4. Add your own training data to data/sample_texts.txt")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
