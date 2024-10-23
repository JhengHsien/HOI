import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, key_dim)
        self.key_layer = nn.Linear(input_dim, key_dim)
        self.value_layer = nn.Linear(input_dim, value_dim)

    def forward(self, queries, keys, values):
        # Generate queries, keys, and values
        Q = self.query_layer(queries)
        K = self.key_layer(keys)
        V = self.value_layer(values)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output