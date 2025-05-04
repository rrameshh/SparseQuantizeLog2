# llm_quantizers/llm_logarithm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantizers._ste import round_ste

class AdaLogLLMQuantizer(nn.Module):
    """
    Enhanced AdaLog Quantizer specialized for LLM activations.
    
    This quantizer uses a mixture of logarithmic bases to better fit
    the multimodal distributions commonly found in LLM activations.
    """
    def __init__(self, n_bits=4, symmetric=False, channel_wise=False, num_clusters=2):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.num_clusters = num_clusters
        self.inited = False
        self.training_mode = False
        
        # Base AdaLog parameters
        self.r = 37.0  # Fixed parameter as in original AdaLog
        self.register_buffer('q', torch.tensor([int(self.r)]))
        
        # New parameters for mixture of logarithmic bases
        self.register_buffer('cluster_assignments', torch.zeros((num_clusters,)))
        self.register_buffer('mixture_weights', torch.ones((num_clusters,)) / num_clusters)
        self.register_parameter('q_values', nn.Parameter(torch.ones(num_clusters) * int(self.r)))
        
        # Lookup tables for each base
        self.register_buffer('tables', torch.zeros((num_clusters, self.n_levels * 2)))
        self.register_buffer('bias_reparamed', torch.tensor(False))
        
        # For handling negative values in activations
        self.shift = nn.Parameter(torch.zeros((1)))
        
    def update_tables(self):
        """Update lookup tables for each logarithmic base in the mixture."""
        for c in range(self.num_clusters):
            for i in range(0, self.n_levels * 2):
                # Compute quantized value using the specific logarithmic base for this cluster
                val = round((2 ** (-((self.q_values[c].item() * i) % self.r) / self.r)) * (4 * self.n_levels - 2)) / (4 * self.n_levels - 2)
                self.tables[c, i].data.copy_(torch.tensor(val))
    
    def cluster_activations(self, x, num_samples=1000):
        """
        Cluster activation values to determine optimal logarithmic bases.
        This is a key innovation for LLMs where activation distributions
        are often multimodal.
        """
        # Sample activation values
        if x.numel() > num_samples:
            indices = torch.randperm(x.numel())[:num_samples]
            samples = x.view(-1)[indices]
        else:
            samples = x.view(-1)
        
        # Filter out zeros and very small values
        samples = samples[samples > 1e-5]
        if samples.numel() == 0:
            return
        
        # Take log2 of samples to work in log space
        log_samples = -torch.log2(samples)
        
        # Simple k-means clustering
        centroids = torch.linspace(log_samples.min(), log_samples.max(), self.num_clusters)
        
        for _ in range(10):  # 10 iterations of k-means
            # Assign samples to nearest centroid
            dists = torch.abs(log_samples.unsqueeze(1) - centroids.unsqueeze(0))
            assignments = dists.argmin(dim=1)
            
            # Update centroids
            new_centroids = centroids.clone()
            for c in range(self.num_clusters):
                if (assignments == c).sum() > 0:
                    new_centroids[c] = log_samples[assignments == c].mean()
            
            if torch.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        # Compute optimal q values based on centroids
        # The idea: we want 2^(-q/r) to approximate the distribution well
        for c in range(self.num_clusters):
            # Convert centroid from log2 space to linear space
            center = 2 ** (-centroids[c])
            # Find q that makes 2^(-q/r) close to this value
            optimal_q = -math.log2(center) * self.r
            self.q_values.data[c] = torch.tensor(max(10, min(50, optimal_q)))
            
        # Compute mixture weights based on cluster sizes
        for c in range(self.num_clusters):
            self.mixture_weights[c] = (assignments == c).float().mean()
            
        self.update_tables()

    def forward(self, x):
        """
        Forward pass with mixture of logarithmic quantizers.
        For each value, we determine which cluster it belongs to and
        use the corresponding logarithmic base.
        """
        if self.n_bits == 32:
            return x
            
        assert self.inited
        
        # Apply shift for handling negative values
        shifted_x = (x + self.shift).clamp(min=1e-15, max=1.0)
        
        # Initialize output tensor
        x_dequant = torch.zeros_like(x)
        
        # Apply each quantizer according to its mixture weight
        log_x = -shifted_x.log2() 
        
        for c in range(self.num_clusters):
            # Determine which values are closest to this cluster
            if c < self.num_clusters - 1:
                # For all but the last cluster, use values that are closer to this centroid
                # than the next one
                mask = (log_x <= self.cluster_assignments[c+1]) & (log_x > self.cluster_assignments[c])
            else:
                # For the last cluster, use all remaining values
                mask = log_x > self.cluster_assignments[c]
            
            # Skip if no values belong to this cluster
            if not mask.any():
                continue
                
            # Quantize using the specific logarithmic base for this cluster
            x_quant = torch.round(log_x[mask] * self.r / self.q_values[c])
            x_quant = x_quant.clamp(0, 2 * self.n_levels - 1).long()
            
            # Dequantize using lookup table
            x_dequant[mask] = (2 ** (-x_quant * self.q_values[c] / self.r)) * self.scale
            
        # Remove shift to get back to original range
        if not self.bias_reparamed:
            x_dequant = x_dequant - self.shift
            
        return x_dequant
        
    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, clusters={self.num_clusters}, q={self.q_values.tolist()})'