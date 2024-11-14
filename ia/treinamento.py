from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def treinar_modelo(model, tokenizer, dataset_path):
    dataset = load_dataset('text', data_files={'train': dataset_path})
    
    def tokenizar_exemplo(exemplo):
        return tokenizer(exemplo['text'], truncation=True, padding='max_length')
    
    dataset = dataset.map(tokenizar_exemplo, batched=True)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
    )
    
    trainer.train()
    model.save_pretrained('./modelo_treinado')
