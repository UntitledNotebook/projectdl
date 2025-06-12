import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size: int, emb_dimension: int, hidden_dimension: int = None):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        
        if hidden_dimension:
            self.fc1 = nn.Linear(emb_dimension, hidden_dimension)
            self.output = nn.Linear(hidden_dimension, emb_size)
            self.use_hidden_layer = True
        else:
            self.output = nn.Linear(emb_dimension, emb_size)
            self.use_hidden_layer = False

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
        if self.use_hidden_layer:
            init.kaiming_uniform_(self.fc1.weight.data, nonlinearity='relu')
            init.zeros_(self.fc1.bias.data)
            init.xavier_uniform_(self.output.weight.data)
            init.zeros_(self.output.bias.data)
        else:
            init.xavier_uniform_(self.output.weight.data)
            init.zeros_(self.output.bias.data)


    def forward(self, target, context):
        emb_target = self.target_embeddings(target)
        
        if self.use_hidden_layer:
            hidden_output = F.relu(self.fc1(emb_target)) 
            score = self.output(hidden_output)           
        else:
            score = self.output(emb_target)              
            
        log_probs = F.log_softmax(score, dim=-1)

        total_loss = 0
        num_valid_contexts = 0
        
        for i in range(context.shape[1]):
            context_at_i = context[:, i]
            
            valid_mask = (context_at_i != -1)
            
            if valid_mask.any():
                loss_at_i = F.nll_loss(
                    log_probs[valid_mask],
                    context_at_i[valid_mask],
                    ignore_index=-1 
                )
                total_loss += loss_at_i
                num_valid_contexts += 1
        
        if num_valid_contexts > 0:
            return total_loss / num_valid_contexts
        else:
            return torch.tensor(0.0, device=target.device, requires_grad=True)