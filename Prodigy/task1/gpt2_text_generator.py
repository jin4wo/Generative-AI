import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import json
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2TextGenerator:
    """
    A class for fine-tuning GPT-2 and generating text based on prompts.
    """
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the GPT-2 text generator.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on device {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def prepare_dataset(self, text_file_path: str, output_dir: str = "processed_data"):
        """
        Prepare the dataset for training by tokenizing the text.
        
        Args:
            text_file_path: Path to the text file containing training data
            output_dir: Directory to save the processed dataset
        """
        logger.info(f"Preparing dataset from {text_file_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into lines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Tokenize the text
        tokenized_texts = []
        for line in lines:
            tokens = self.tokenizer.encode(line, add_special_tokens=True)
            tokenized_texts.append(tokens)
        
        # Save tokenized data
        with open(os.path.join(output_dir, 'tokenized_data.json'), 'w') as f:
            json.dump(tokenized_texts, f)
        
        logger.info(f"Dataset prepared and saved to {output_dir}")
        return tokenized_texts
    
    def create_dataset(self, tokenized_texts: List[List[int]], max_length: int = 512):
        """
        Create a PyTorch dataset from tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized text sequences
            max_length: Maximum sequence length
        
        Returns:
            TextDataset object
        """
        # Flatten and truncate sequences
        flat_tokens = []
        for tokens in tokenized_texts:
            if len(tokens) > max_length:
                # Split long sequences
                for i in range(0, len(tokens), max_length):
                    flat_tokens.extend(tokens[i:i + max_length])
            else:
                flat_tokens.extend(tokens)
        
        # Create temporary file for dataset
        temp_file = "temp_dataset.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for i in range(0, len(flat_tokens), max_length):
                chunk = flat_tokens[i:i + max_length]
                text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                f.write(text + '\n')
        
        # Create dataset
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=temp_file,
            block_size=max_length
        )
        
        # Clean up temporary file
        os.remove(temp_file)
        
        return dataset
    
    def fine_tune(self, 
                  dataset, 
                  output_dir: str = "fine_tuned_model",
                  num_epochs: int = 3,
                  batch_size: int = 4,
                  learning_rate: float = 5e-5,
                  save_steps: int = 500,
                  logging_steps: int = 100):
        """
        Fine-tune the GPT-2 model on the provided dataset.
        
        Args:
            dataset: Training dataset
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            save_steps: Number of steps between model saves
            logging_steps: Number of steps between logging
        """
        logger.info("Starting fine-tuning process")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_steps=100,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Start training
        logger.info("Training started...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load a fine-tuned model from the specified path.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
        logger.info("Fine-tuned model loaded successfully")
    
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 100, 
                     temperature: float = 0.8,
                     top_k: int = 50,
                     top_p: float = 0.9,
                     num_return_sequences: int = 1,
                     do_sample: bool = True) -> List[str]:
        """
        Generate text based on a given prompt.
        
        Args:
            prompt: The input prompt to generate text from
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            num_return_sequences: Number of different sequences to generate
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            List of generated text sequences
        """
        logger.info(f"Generating text with prompt: '{prompt}'")
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_texts = []
        for sequence in output:
            generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        logger.info(f"Generated {len(generated_texts)} text sequence(s)")
        return generated_texts
    
    def evaluate_model(self, test_prompts: List[str]) -> dict:
        """
        Evaluate the model on a set of test prompts.
        
        Args:
            test_prompts: List of test prompts
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        results = {
            'prompts': [],
            'generated_texts': [],
            'text_lengths': [],
            'perplexity_scores': []
        }
        
        for prompt in test_prompts:
            # Generate text
            generated_texts = self.generate_text(prompt, max_length=50, num_return_sequences=1)
            generated_text = generated_texts[0]
            
            # Calculate metrics
            text_length = len(generated_text.split())
            
            # Calculate perplexity (simplified)
            input_ids = self.tokenizer.encode(generated_text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            results['prompts'].append(prompt)
            results['generated_texts'].append(generated_text)
            results['text_lengths'].append(text_length)
            results['perplexity_scores'].append(perplexity)
        
        # Calculate average metrics
        avg_length = sum(results['text_lengths']) / len(results['text_lengths'])
        avg_perplexity = sum(results['perplexity_scores']) / len(results['perplexity_scores'])
        
        results['average_length'] = avg_length
        results['average_perplexity'] = avg_perplexity
        
        logger.info(f"Evaluation completed. Avg length: {avg_length:.2f}, Avg perplexity: {avg_perplexity:.2f}")
        
        return results

def main():
    """
    Main function to demonstrate the GPT-2 text generator.
    """
    print("=== GPT-2 Text Generation Task ===\n")
    
    # Initialize the generator
    generator = GPT2TextGenerator()
    
    # Check if fine-tuned model exists with required files; otherwise skip training
    model_path = "fine_tuned_model"
    model_files_exist = (
        os.path.exists(model_path) and 
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) and
        os.path.exists(os.path.join(model_path, "config.json"))
    )

    if model_files_exist:
        print("Loading existing fine-tuned model...")
        generator.load_fine_tuned_model(model_path)
    else:
        print("No fine-tuned model found. Using pre-trained GPT-2 (skipping training).")
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Python programming",
        "Natural language processing",
        "Deep learning models"
    ]
    
    print("\n=== Generating Text ===\n")
    
    # Generate text for each prompt
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        generated_texts = generator.generate_text(
            prompt, 
            max_length=80, 
            temperature=0.7,
            num_return_sequences=1
        )
        
        for i, text in enumerate(generated_texts):
            print(f"Generated: {text}")
        print("-" * 50)
    
    # Evaluate the model
    print("\n=== Model Evaluation ===\n")
    evaluation_results = generator.evaluate_model(test_prompts[:3])
    
    print(f"Average text length: {evaluation_results['average_length']:.2f} words")
    print(f"Average perplexity: {evaluation_results['average_perplexity']:.2f}")
    
    print("\n=== Task Completed Successfully! ===")

if __name__ == "__main__":
    main()
