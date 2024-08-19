import torch

from torchmetrics import Accuracy, Metric, Precision, Recall, F1Score, JaccardIndex, ConfusionMatrix


class MetricCollection(torch.nn.ModuleDict):
    def __init__(self, metrics):
        super().__init__()
        for k, v in metrics.items():
            self[k] = v
        
    def forward(self, *args, **kwargs):
        return {name: metric(*args, **kwargs) for name, metric in self.items()}
    
    def compute(self, *args, **kwargs):
        return {name: metric.compute() for name, metric in self.items()}
    
    def reset(self):
        for metric in self.values():
            metric.reset()
            
    def update(self, *args, **kwargs):
        for metric in self.values():
            metric.update(*args, **kwargs)
            
            
class RowF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.metrics = MetricCollection({
            "IntraRowF1": F1Score(num_classes=3, task="multiclass", ignore_index=-100, average=None),
            "InterRowF1": F1Score(num_classes=3, task="multiclass", ignore_index=-100, average=None)
        })
        
    def update(self, preds, target, on_row_plants, off_row_plants):
        preds = preds.clone().flatten()
        target = target.clone().flatten()
        bg = target == 0
        
        on_row_plants = on_row_plants.flatten().bool()
        off_row_plants = off_row_plants.flatten().bool()
        
        # on_row_target = target[on_row_plants]
        # on_row_preds = preds[on_row_plants]
        
        # off_row_target = target[off_row_plants]
        # off_row_preds = preds[off_row_plants]
        
        # on_row_target = target * on_row_plants
        # off_row_target = target * off_row_plants
        
        # on_row_preds = preds * on_row_plants
        # off_row_preds = preds * off_row_plants
        
        on_row_target = target.clone()
        on_row_target[off_row_plants] = -100
        
        off_row_target = target.clone()
        off_row_target[on_row_plants] = -100 
        
        if on_row_target.numel() > 0:
            self.metrics["IntraRowF1"](preds, on_row_target)
        if off_row_target.numel() > 0:
            self.metrics["InterRowF1"](preds, off_row_target)
        
    def compute(self):
        return self.metrics.compute()
    
    def reset(self):
        self.metrics.reset()

def build_metrics(params):
    metrics = {}
    for key, value in params.items():
        metric = globals()[key](**value)
        metrics[key] = metric
    metrics = MetricCollection(metrics)
    return metrics