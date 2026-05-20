"""
Utilitários para treinamento de modelos.

Este módulo fornece classes e funções para facilitar o treinamento
dos modelos BERTimbau de forma padronizada.
"""

import os
import torch
import torch.nn as nn
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer
    # EarlyStoppingCallback removido para evitar conflito no Windows
)
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CUSTOM TRAINERS - Para suporte a Class Weights e Focal Loss
# ============================================================

class WeightedCETrainer(Trainer):
    """Trainer com Cross-Entropy ponderada por classe.

    Aplica os class_weights diretamente na loss durante o treinamento,
    forcando o modelo a prestar mais atencao a classes minoritarias.
    """
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """Trainer com Focal Loss (Lin et al. 2017).

    FL(p) = -(1-p)^gamma * log(p)
    Reduz peso de exemplos faceis e foca em exemplos dificeis.
    """
    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.alpha.to(logits.device) if self.alpha is not None else None
        ce_loss = nn.functional.cross_entropy(logits, labels, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        loss = focal_loss.mean()
        return (loss, outputs) if return_outputs else loss

class TrainingHelper:
    """Classe auxiliar para treinamento de modelos."""
    
    def __init__(
        self,
        task_name: str,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        output_base_dir: str = "./models"
    ):
        """
        Inicializa o helper de treinamento.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            model_name: Nome do modelo base
            output_base_dir: Diretório base para salvar modelos
        """
        self.task_name = task_name
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        
        # Cria diretório de saída se não existir
        os.makedirs(output_base_dir, exist_ok=True)
        
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepara os datasets para treinamento.
        
        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            val_texts: Textos de validação
            val_labels: Labels de validação
            test_texts: Textos de teste
            test_labels: Labels de teste
            tokenizer: Tokenizador BERT
            max_length: Comprimento máximo das sequências
            
        Returns:
            Tuple com datasets tokenizados
        """
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Cria datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'labels': test_labels
        })
        
        # Tokeniza
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Datasets preparados para {self.task_name}")
        logger.info(f"Treino: {len(train_dataset)} amostras")
        logger.info(f"Validação: {len(val_dataset)} amostras")
        logger.info(f"Teste: {len(test_dataset)} amostras")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_f1",
        greater_is_better: bool = True
    ) -> TrainingArguments:
        """
        Cria argumentos de treinamento padronizados.
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=None,
            seed=42
        )
    
    def compute_metrics(self, eval_pred):
        """
        Computa métricas durante o treinamento.
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_labels: int,
        training_args: TrainingArguments,
        model_path: Optional[str] = None,
        loss_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[BertForSequenceClassification, Trainer]:
        """
        Treina o modelo.

        Args:
            ...
            loss_config: Configuracao opcional de loss customizada. Formatos:
                - None: usa CrossEntropy padrao
                - {'type': 'weighted_ce', 'weights': [w0, w1, w2]}: CE ponderada
                - {'type': 'focal', 'gamma': 2.0, 'alpha': [a0, a1, a2]}: Focal Loss
        """
        # Carrega o modelo
        if model_path and os.path.exists(model_path):
            model = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )

        # Cria o trainer apropriado baseado em loss_config
        if loss_config is None:
            # Comportamento padrao: CrossEntropy
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=None
            )
            logger.info(f"Trainer: padrao (CrossEntropy)")
        elif loss_config.get('type') == 'weighted_ce':
            weights_list = loss_config['weights']
            class_weights = torch.tensor(weights_list, dtype=torch.float32)
            trainer = WeightedCETrainer(
                class_weights=class_weights,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=None
            )
            logger.info(f"Trainer: WeightedCE com weights={weights_list}")
        elif loss_config.get('type') == 'focal':
            gamma = loss_config.get('gamma', 2.0)
            alpha_list = loss_config.get('alpha')
            alpha = torch.tensor(alpha_list, dtype=torch.float32) if alpha_list else None
            trainer = FocalLossTrainer(
                gamma=gamma,
                alpha=alpha,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=None
            )
            logger.info(f"Trainer: FocalLoss com gamma={gamma}, alpha={alpha_list}")
        else:
            raise ValueError(f"loss_config['type'] desconhecido: {loss_config.get('type')}")

        logger.info(f"Iniciando treinamento para {self.task_name}")

        # Treina o modelo
        trainer.train()

        logger.info(f"Treinamento concluido para {self.task_name}")

        return model, trainer
    
    def save_model_with_metadata(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        output_dir: str,
        training_args: TrainingArguments,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Salva o modelo com metadados.
        """
        # Salva modelo e tokenizador
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Cria metadados
        metadata = {
            'task': self.task_name,
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'training_args': {
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'warmup_steps': training_args.warmup_steps,
                'weight_decay': training_args.weight_decay
            },
            'final_metrics': metrics
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Salva metadados
        with open(os.path.join(output_dir, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo salvo em {output_dir} com metadados")
    
    def get_output_dir(self, experiment_name: Optional[str] = None) -> str:
        """
        Gera diretório de saída para o modelo.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            dir_name = f"{self.task_name}_{experiment_name}_{timestamp}"
        else:
            dir_name = f"{self.task_name}_{timestamp}"
        
        return os.path.join(self.output_base_dir, dir_name)


def create_training_config(
    task: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Cria configuração padrão para treinamento.
    """
    return {
        'task': task,
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'num_train_epochs': epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'eval_steps': 500,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True
    }