from transformers import Trainer, TrainingArguments
from utils.pre_processamento import carregar_tokenizer, preparar_dataset
from transformers import AutoModelForCausalLM

class AprendizadoContinuo:
    def __init__(self, modelo_caminho, tokenizer_caminho):
        self.tokenizer = carregar_tokenizer(tokenizer_caminho)
        self.model = AutoModelForCausalLM.from_pretrained(modelo_caminho)
    
    def treinar_incremental(self, caminho_dados, epochs=1, output_dir='./results_continuo'):
        tokenized_dataset = preparar_dataset(caminho_dados, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            logging_dir='./logs_continuo'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            data_collator=None
        )
        trainer.train()
        trainer.save_model(output_dir)

    def aprendizado_continuo(novos_dados, pipeline):
    pipeline.data_processor.prepare_data(novos_dados)
    pipeline.train_model(epochs=1)  # Treina apenas 1 época para ajuste incremental
