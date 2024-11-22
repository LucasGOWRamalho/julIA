from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import TrainingArguments

def treinar_modelo(model, tokenizer, dataset_path):
    dataset = load_dataset('text', data_files={'train': dataset_path})
    
    def tokenizar_exemplo(exemplo):
        return tokenizer(exemplo['text'], truncation=True, padding='max_length')
    
    dataset = dataset.map(tokenizar_exemplo, batched=True)
    
    training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',  # Diretório para os logs
    logging_steps=100,     # Log a cada 100 steps
)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
    )
    
    trainer.train()
    model.save_pretrained('./modelo_treinado')


def configurar_treinamento(output_dir='./results', epochs=3, batch_size=8, log_dir='./logs'):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=log_dir,
        evaluation_strategy='no',
        weight_decay=0.01
    )

