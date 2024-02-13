from transformers import Trainer, TrainingArguments

from selfweed.data import get_dataloaders
from selfweed.models import MODEL_REGISTRY


class Run:
    def __init__(self, params):
        self.params = params
        self.model = MODEL_REGISTRY[params['model']['name']](**params['model']['params'])
        self.data = get_dataloaders(params['data'])
        self.train_params = params['train']      

    def train(self):
        arguments = TrainingArguments(
            report_to="wandb",
        )
            
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=self.data.train_dataset,
            eval_dataset=self.data.eval_dataset,
        )
        trainer.train()
        return trainer