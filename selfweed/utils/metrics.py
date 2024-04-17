import torch

from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex, ConfusionMatrix


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


def build_metrics(params):
    metrics = {}
    for key, value in params.items():
        metric = globals()[key](**value)
        metrics[key] = metric
    metrics = MetricCollection(metrics)
    return metrics