import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)

        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.A = exp_Z / sum_exp_Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        dLdZ = np.zeros_like(dLdA)
           
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = self.A.reshape(-1, C)
            dLdA = dLdA.reshape(-1, C)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = np.moveaxis(self.A.reshape(shape), -1, self.dim)
            dLdZ = np.moveaxis(dLdZ.reshape(shape), -1, self.dim)

        return dLdZ
 

    