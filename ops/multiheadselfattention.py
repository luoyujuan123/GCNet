import torch
import torch.nn as nn
import math
import pdb
class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # pdb.set_trace()
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

# attention = SelfAttention(dropout=0.5)
# batch_size, num_queries, num_hiddens  = 2, 4, 10
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans.shape)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# batch_size, num_queries, num_hiddens, num_heads  = 32, 8, 6272, 4
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# X = torch.randn(batch_size, num_queries, num_hiddens)
# ans = attention(X, X, X)
# print(ans.shape)
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        # Squeeze operation: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation operation: Fully connected layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()
        
        # Squeeze: Global Average Pooling (across the temporal dimension)
        y = self.global_avg_pool(x).view(batch_size, channels)
        
        # Excitation: Fully connected layers and Sigmoid activation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1)
        
        # Re-scale the input features by the learned channel-wise weights
        return x * y.expand_as(x)
class SEBlock_four(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock_four, self).__init__()
        
        # Squeeze: Global Average Pooling to get channel-wise statistics
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size is (N, C, 1, 1)

        # Excitation: Fully connected layers with a reduction to control model complexity
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()  # Input shape is (N, C, H, W)

        # Squeeze: Global average pooling
        y = self.global_avg_pool(x).view(batch_size, channels)  # Shape (N, C)

        # Excitation: Fully connected layers
        y = self.fc1(y)  # Shape (N, C // reduction)
        y = self.relu(y)
        y = self.fc2(y)  # Shape (N, C)
        y = self.sigmoid(y)  # Shape (N, C)

        # Scale: Reshape and multiply the input tensor by the scaling factors
        y = y.view(batch_size, channels, 1, 1)  # Shape (N, C, 1, 1)
        return x * y  # Element-wise multiplication with original input