import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat


class ContrastiveLoss(nn.Module):
    def __init__(self):
        """
        Computes the contrastive loss of the class prompts generated for each example in the support set. 
        """
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor([torch.log(torch.tensor(10))]))
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        
    def forward(self, scores: torch.Tensor):
        """
        Arguments:
            scores (torch.Tensor): B x 4 score matrix
        """
        scores = [[score * torch.exp(self.t_prime) for score in score_stage] for score_stage in scores]
        stage_losses = []
        
        for score_stage in scores:
            crop_logits, weed_logits = score_stage
            crop_batch_size = crop_logits.shape[0]
            weed_batch_size = weed_logits.shape[0]
            if crop_logits.numel():
                crop_labels = torch.stack([torch.ones_like(crop_logits[:, 0]), torch.zeros_like(crop_logits[:, 1])], dim=1)
                crop_loss = F.binary_cross_entropy_with_logits(crop_logits, crop_labels) / crop_batch_size
            else:
                crop_loss = torch.tensor(0.0)
            
            if weed_logits.numel():
                weed_labels = torch.stack([torch.zeros_like(weed_logits[:, 0]), torch.ones_like(weed_logits[:, 1])], dim=1)
                weed_loss = F.binary_cross_entropy_with_logits(weed_logits, weed_labels) / weed_batch_size
            else:
                weed_loss = torch.tensor(0.0)
            stage_losses.append(crop_loss + weed_loss)
            
        return sum(stage_losses) / len(stage_losses)

        
        
        
