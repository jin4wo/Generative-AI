import torch
from gpt2_text_generator import GPT2TextGenerator
import os
import sys

def interactive_generator():
    """
    Interactive interface for GPT-2 text generation.
    """
    print("🤖 GPT-2 Text Generation Interactive Interface")
    print("=" * 50)
    
    # Initialize the generator
    try:
        generator = GPT2TextGenerator()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check for fine-tuned model
    model_path = "fine_tuned_model"
    model_files_exist = (
        os.path.exists(model_path) and 
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) and
        os.path.exists(os.path.join(model_path, "config.json"))
    )
    
    if model_files_exist:
        try:
            generator.load_fine_tuned_model(model_path)
            print("✅ Fine-tuned model loaded!")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("Using pre-trained model instead.")
    else:
        print("ℹ️  No fine-tuned model found. Using pre-trained GPT-2.")
    
    print("\n🎯 Available commands:")
    print("  generate <prompt> - Generate text from a prompt")
    print("  settings - Adjust generation parameters")
    print("  examples - Show example prompts")
    print("  help - Show this help message")
    print("  quit - Exit the program")
    print("-" * 50)
    
    # Default generation parameters
    params = {
        'max_length': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'num_sequences': 1
    }
    
    while True:
        try:
            user_input = input("\n💬 Enter command: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n🎯 Available commands:")
                print("  generate <prompt> - Generate text from a prompt")
                print("  settings - Adjust generation parameters")
                print("  examples - Show example prompts")
                print("  help - Show this help message")
                print("  quit - Exit the program")
            
            elif user_input.lower() == 'examples':
                print("\n📝 Example prompts you can try:")
                print("  generate The future of artificial intelligence")
                print("  generate Machine learning is")
                print("  generate Python programming")
                print("  generate Natural language processing")
                print("  generate Deep learning models")
                print("  generate The quick brown fox")
                print("  generate Innovation in technology")
            
            elif user_input.lower() == 'settings':
                print(f"\n⚙️  Current settings:")
                print(f"  Max length: {params['max_length']}")
                print(f"  Temperature: {params['temperature']}")
                print(f"  Top-k: {params['top_k']}")
                print(f"  Top-p: {params['top_p']}")
                print(f"  Number of sequences: {params['num_sequences']}")
                
                print("\n🔧 Adjust settings (press Enter to keep current value):")
                
                # Max length
                new_max_length = input(f"Max length ({params['max_length']}): ").strip()
                if new_max_length:
                    try:
                        params['max_length'] = int(new_max_length)
                    except ValueError:
                        print("❌ Invalid input. Keeping current value.")
                
                # Temperature
                new_temp = input(f"Temperature ({params['temperature']}): ").strip()
                if new_temp:
                    try:
                        params['temperature'] = float(new_temp)
                    except ValueError:
                        print("❌ Invalid input. Keeping current value.")
                
                # Top-k
                new_top_k = input(f"Top-k ({params['top_k']}): ").strip()
                if new_top_k:
                    try:
                        params['top_k'] = int(new_top_k)
                    except ValueError:
                        print("❌ Invalid input. Keeping current value.")
                
                # Top-p
                new_top_p = input(f"Top-p ({params['top_p']}): ").strip()
                if new_top_p:
                    try:
                        params['top_p'] = float(new_top_p)
                    except ValueError:
                        print("❌ Invalid input. Keeping current value.")
                
                # Number of sequences
                new_num_seq = input(f"Number of sequences ({params['num_sequences']}): ").strip()
                if new_num_seq:
                    try:
                        params['num_sequences'] = int(new_num_seq)
                    except ValueError:
                        print("❌ Invalid input. Keeping current value.")
                
                print("✅ Settings updated!")
            
            elif user_input.lower().startswith('generate '):
                prompt = user_input[9:]  # Remove 'generate ' prefix
                
                if not prompt:
                    print("❌ Please provide a prompt after 'generate'")
                    continue
                
                print(f"\n🎯 Generating text for: '{prompt}'")
                print("⏳ Please wait...")
                
                try:
                    generated_texts = generator.generate_text(
                        prompt=prompt,
                        max_length=params['max_length'],
                        temperature=params['temperature'],
                        top_k=params['top_k'],
                        top_p=params['top_p'],
                        num_return_sequences=params['num_sequences']
                    )
                    
                    print("\n✨ Generated text:")
                    print("-" * 40)
                    
                    for i, text in enumerate(generated_texts, 1):
                        if params['num_sequences'] > 1:
                            print(f"\n📄 Sequence {i}:")
                        print(text)
                        print("-" * 40)
                    
                    # Show word count
                    total_words = len(text.split())
                    print(f"\n📊 Generated {total_words} words")
                    
                except Exception as e:
                    print(f"❌ Error generating text: {e}")
            
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def quick_test():
    """
    Quick test function to verify the model works.
    """
    print("🧪 Running quick test...")
    
    try:
        generator = GPT2TextGenerator()
        
        # Test with a simple prompt
        test_prompt = "The future of"
        print(f"Testing with prompt: '{test_prompt}'")
        
        generated_texts = generator.generate_text(
            test_prompt,
            max_length=30,
            temperature=0.7,
            num_return_sequences=1
        )
        
        print(f"✅ Test successful! Generated: {generated_texts[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        interactive_generator()
