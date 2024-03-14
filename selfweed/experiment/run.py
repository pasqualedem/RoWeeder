from transformers import Trainer, TrainingArguments

from selfweed.data import get_trainval
from selfweed.models import MODEL_REGISTRY


class Run:
    def __init__(self):
        pass
    
    def init(self, params):
        self.params = params
        self.model = MODEL_REGISTRY[params['model']['name']](**params['model']['params'])
        
    def _prep_for_train(self):
        self.data = get_trainval(self.params['dataset'], self.params['dataloader'])
        self.train_params = self.params['train']
    
    def launch(self):
        return self.train()

    def train(self):
        arguments = TrainingArguments(
            report_to="wandb",
            output_dir=self.params['tracker']['output_dir'],
            per_device_train_batch_size=self.params['dataloader']['batch_size'],
            dataloader_num_workers=self.params['dataloader']['num_workers'],
        )
            
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=self.data[0],
            eval_dataset=self.data[1],
        )
        trainer.train()
        return trainer