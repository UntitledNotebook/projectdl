import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size: int, emb_dimension: int, hidden_dimension: int = None): # Added hidden_dimension
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        
        if hidden_dimension:
            # More complex: Embedding -> Hidden Layer -> Output Layer
            self.fc1 = nn.Linear(emb_dimension, hidden_dimension)
            self.output = nn.Linear(hidden_dimension, emb_size)
            self.use_hidden_layer = True
        else:
            # Original: Embedding -> Output Layer
            self.output = nn.Linear(emb_dimension, emb_size)
            self.use_hidden_layer = False

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
        if self.use_hidden_layer:
            init.kaiming_uniform_(self.fc1.weight.data, nonlinearity='relu')
            init.zeros_(self.fc1.bias.data)
            init.xavier_uniform_(self.output.weight.data) # Or kaiming if output activation was ReLU
            init.zeros_(self.output.bias.data)
        else:
            # Initialize the original output layer (if not using hidden)
            # You might want a specific initialization for self.output if not using hidden layer
            init.xavier_uniform_(self.output.weight.data)
            init.zeros_(self.output.bias.data)


    def forward(self, target, context):
        emb_target = self.target_embeddings(target) # Shape: (batch_size, emb_dimension)
        
        if self.use_hidden_layer:
            hidden_output = F.relu(self.fc1(emb_target)) # Shape: (batch_size, hidden_dimension)
            score = self.output(hidden_output)           # Shape: (batch_size, emb_size)
        else:
            score = self.output(emb_target)              # Shape: (batch_size, emb_size)
            
        log_probs = F.log_softmax(score, dim=-1) # Shape: (batch_size, emb_size)

        # The context tensor has shape (batch_size, num_context_words)
        # We want to calculate the loss for each context word
        
        total_loss = 0
        num_valid_contexts = 0
        
        # Iterate over each context word position
        for i in range(context.shape[1]):
            context_at_i = context[:, i] # Shape: (batch_size,)
            
            # Create a mask for valid (non-padding) context words
            valid_mask = (context_at_i != -1)
            
            if valid_mask.any():
                # Calculate NLL loss only for valid context words
                loss_at_i = F.nll_loss(
                    log_probs[valid_mask],  # Log probabilities for samples with valid context words
                    context_at_i[valid_mask], # The valid context words themselves
                    ignore_index=-1 # Should not be strictly necessary here due to mask, but good practice
                )
                total_loss += loss_at_i
                num_valid_contexts += 1
        
        if num_valid_contexts > 0:
            return total_loss / num_valid_contexts
        else:
            # Handle cases where there are no valid context words in the batch (e.g., all padding)
            return torch.tensor(0.0, device=target.device, requires_grad=True)