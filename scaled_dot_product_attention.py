import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = NotImplementedError
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        self.Q, self.K, self.V = Q, K, V
        # TODO: Implement forward pass
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        K_transposed = np.swapaxes(K, -2, -1)
        scaled_dot_product = np.matmul(Q, K_transposed) / np.sqrt(Q.shape[-1])

        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        max_scores = np.max(scaled_dot_product, axis=-1, keepdims=True)
        exp_scores = np.exp(scaled_dot_product - max_scores)
        sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
        self.attention_scores = exp_scores / sum_exp_scores

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)
        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.matmul(np.swapaxes(self.attention_scores, -2, -1), d_output)

        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -2, -1))
        sum_d_attention = np.sum(d_attention_scores * self.attention_scores, axis=-1, keepdims=True)
        d_scaled_dot_product = self.attention_scores * (d_attention_scores - sum_d_attention)

        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.Q.shape[-1])

        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = np.matmul(d_scaled_dot_product, self.K)

        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -2, -1), self.Q)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

