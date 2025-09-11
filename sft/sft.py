
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import json
import os
from typing import Dict, List
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChemDFMProtocolDataset(Dataset):
    """Dataset class for ChemDFM protocol generation with conversation format"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load and format data
        logger.info(f"Loading data from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Format as conversation for ChemDFM
                    formatted_text = self.format_as_conversation(item['instruction'], item['response'])
                    self.data.append(formatted_text)
                    
                except Exception as e:
                    logger.warning(f"Error loading line {line_num}: {e}")
        
        logger.info(f"Loaded {len(self.data)} examples")
        
        # Show sample
        if self.data:
            sample = self.data[0]
            logger.info(f"Sample formatted text length: {len(sample)} chars")
            logger.info(f"Sample preview: {sample[:200]}...")
    
    def format_as_conversation(self, instruction: str, response: str) -> str:
        """Format data as conversation for ChemDFM"""
        
        conversation = f"[Round 0]\nHuman: {instruction}\nAssistant: {response}"
        return conversation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
       
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # For causal LM, input_ids and labels are the same
        input_ids = encoding['input_ids']
        
        # Create attention mask
        attention_mask = encoding.get('attention_mask', [1] * len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)  # For causal LM training
        }

class ChemDFMLoRATrainer:
    """Main training class for ChemDFM with LoRA"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Setup logging to file
        log_file = os.path.join(config['output_dir'], 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def setup_model_and_tokenizer(self):
        """Setup ChemDFM model with LoRA"""
        logger.info(f"Loading ChemDFM model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with appropriate precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
        
        # self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_r'],  # Rank
            lora_alpha=self.config['lora_alpha'],  # Scaling parameter
            lora_dropout=self.config['lora_dropout'],  # Dropout
            target_modules=self.config['lora_target_modules'],  # Which layers to apply LoRA
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer
    
    def create_datasets(self):
        """Create training and validation datasets"""
        logger.info("Creating datasets...")
        
        train_dataset = ChemDFMProtocolDataset(
            self.config['train_file'],
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        val_dataset = ChemDFMProtocolDataset(
            self.config['val_file'],
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        return train_dataset, val_dataset
    
    def create_training_args(self):
        """Create training arguments optimized for LoRA"""
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            
            # Training schedule - conservative for small dataset
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            
            # Learning rate - higher for LoRA
            learning_rate=self.config['learning_rate'],
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,  # Small warmup for small dataset
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_dir=f"{self.config['output_dir']}/logs",
            logging_steps=20,  # More frequent logging for small dataset
            report_to=[],  # Disable wandb for simplicity
            
            # Performance
            dataloader_num_workers=2,
            fp16=True,  # Use mixed precision
            gradient_checkpointing=False,  # Save memory
            
            # Misc
            seed=42,
            remove_unused_columns=False,
            push_to_hub=False,
            
            # Save strategy
            save_strategy="steps",
        )
    
    def train(self):
        """Main training function"""
        logger.info("Starting LoRA training setup...")
        
        # Setup model
        self.model, self.tokenizer = self.setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets()
        
        # Create data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not masked LM
            return_tensors="pt",
            pad_to_multiple_of=8  
        )
        
        # Training arguments
        training_args = self.create_training_args()
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Log training info
        effective_batch_size = self.config['batch_size'] * self.config['gradient_accumulation_steps']
        total_steps = len(train_dataset) // effective_batch_size * self.config['num_epochs']
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Total training steps: {total_steps}")
        
        # Start training
        logger.info("Starting LoRA training...")
        train_result = self.trainer.train()
        
        # Save final LoRA adapter
        self.model.save_pretrained(self.config['output_dir'])
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Log results
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def test_inference(self, test_instruction: str):
        """Test model inference with the trained LoRA adapter"""
        logger.info("Testing model inference...")
        
        # Format as conversation
        input_text = f"[Round 0]\nHuman: {test_instruction}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=20,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = generated_text[len(input_text):].strip()
        
        logger.info(f"Test input: {test_instruction}")
        logger.info(f"Generated protocol: {prediction}")
        
        return prediction

def create_chemdfm_config():
    """Create configuration for ChemDFM LoRA training"""
    return {
        # Model
        'model_name': 'OpenDFM/ChemDFM-v1.5-8B',
        
       
        'train_file': 'train_short_instructions.jsonl',
        'val_file': 'val_short_instructions.jsonl',
        
        # Output
        'output_dir': './chemdfm_protocol_lora',
        
        # LoRA parameters
        'lora_r': 16,  
        'lora_alpha': 32, 
        'lora_dropout': 0.1,
        'lora_target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
       
        'num_epochs': 5,  
        'batch_size': 1, 
        'gradient_accumulation_steps': 8,  
        'learning_rate': 2e-4,  
        
        # Token length
        'max_length': 1024,  
        
        # Evaluation
        'eval_steps': 50,  
        'save_steps': 50,
    }

def main():
    """Main execution function"""
    
    # Create config
    config = create_chemdfm_config()
    
    # Check if files exist
    for key in ['train_file', 'val_file']:
        if not os.path.exists(config[key]):
            logger.error(f"File not found: {config[key]}")
            logger.error("Please update the file paths in create_chemdfm_config()")
            return
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This model requires GPU.")
        return
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Log config
    logger.info("LoRA Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer
    trainer = ChemDFMLoRATrainer(config)
    
    # Train
    train_result = trainer.train()
    
    # Test inference
    test_instruction = """Generate experimental protocol:

Reaction: benzaldehyde + sodium borohydride → benzyl alcohol
Type: Ketone to alcohol reduction

Protocol:"""
    
    trainer.test_inference(test_instruction)
    
    logger.info(f"LoRA training complete! Adapter saved to: {config['output_dir']}")
    logger.info("To use the trained model, load the base model and apply the LoRA adapter.")

if __name__ == "__main__":
    main()