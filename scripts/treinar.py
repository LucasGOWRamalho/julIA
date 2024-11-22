from modelo import carregar_modelo
from treinamento import treinar_modelo
from transformers import Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from ia.treinamento import configurar_treinamento
from utils.pre_processamento import carregar_tokenizer, preparar_dataset

def treinar_modelo(caminho_dados, model_name='gpt2', output_dir='./results'):
    tokenizer = carregar_tokenizer(model_name)
    tokenized_dataset = preparar_dataset(caminho_dados, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = configurar_treinamento(output_dir=output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def iniciar_treinamento():
    tokenizer, model = carregar_modelo('gpt2')
    treinar_modelo(model, tokenizer, 'dataset/initial_corpus.txt')

if __name__ == "__main__":
    iniciar_treinamento()
