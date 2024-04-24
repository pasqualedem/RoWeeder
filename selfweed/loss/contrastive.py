import torch
import torch.nn as nn

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
        B, C, R = scores.shape
        scores = scores * torch.exp(self.t_prime) + self.bias
        
        contrastive_matrix = torch.eye(C, device=scores.device)
        contrastive_matrix = 2 * contrastive_matrix - 1
        contrastive_matrix = repeat(contrastive_matrix, "r c -> b r c", b=B)
        loss = -torch.log(torch.sigmoid(scores * contrastive_matrix))
        return loss.sum() / B

        
        
        
