import torch
import torch.nn as nn

from torch.nn.functional import normalize
from einops import rearrange


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
        B = scores.shape[0]
        
        class_embeddings = rearrange(class_embeddings, "b m c d -> b (m c) d")
        class_embeddings = normalize(class_embeddings, p=2, dim=-1)
        dot_products = class_embeddings @ rearrange(class_embeddings, " b c d -> b d c")
        scores = scores * torch.exp(self.t_prime) + self.bias
        
        contrastive_matrix = torch.eye(4, device=class_embeddings.device)
        contrastive_matrix = 2 * contrastive_matrix - 1
        loss = -torch.log(torch.sigmoid(dot_products * contrastive_matrix))
        return loss / B

        
        
        
