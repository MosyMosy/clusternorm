import random
import torch
import torch.nn as nn


class cluster_MixStyle(nn.Module):
    """cluster based MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x, cluster_map):
        if not self.training or not self._activated or (cluster_map is None):
            return x

        if random.random() > self.p:
            return x
        
        lmda = self.beta.sample((x.size(0), 1, 1, 1))
        lmda = lmda.to(x.device)

        # Turn the first head's probabilities to one-hot
        cluster_map_one_hot = torch.argmax(cluster_map[0], dim=1)
        _, cluster_map_count = torch.unique(cluster_map_one_hot, sorted=True, return_counts=True)
        cluster_map_split_ind = torch.cumsum(cluster_map_count, dim=0, dtype=torch.int32) - 1 # make index as zero-based
        cluster_map_sorted_ind = torch.argsort(cluster_map_one_hot, dim=0)        
        clustered_samples = torch.split(x[cluster_map_sorted_ind], cluster_map_split_ind)
            
        # Statistics over sample's spatial dimensions
        sample_mu = clustered_samples.mean(dim=[3, 4], keepdim=True).detach()
        sample_std = ((clustered_samples.var(dim=[3, 4], keepdim=True) + self.eps).sqrt()).detach()
        clustered_samples_normed = (clustered_samples - sample_mu) / sample_std
        
        # Statistics over each cluster
        cluster_mu = clustered_samples.mean(dim=[1, 3, 4], keepdim=True).detach()
        cluster_std = ((clustered_samples.var(dim=[1, 3, 4], keepdim=True) + self.eps).sqrt()).detach()

        mu_mix = sample_mu * lmda + torch.unsqueeze(cluster_mu, 1) * (1-lmda)
        std_mix = sample_std * lmda + torch.unsqueeze(cluster_std, 1) * (1-lmda)
        
        cluster_mixstyle = clustered_samples_normed * std_mix + mu_mix
        cluster_mixstyle = torch.flatten(cluster_mixstyle, end_dim=0)
        cluster_map_sorted_ind_inverse = torch.argsort(cluster_map_sorted_ind, dim=0)        
        # resort the samples as it was originally. This essential to have a valid cluster_map for the subsequent layers
        return cluster_mixstyle[cluster_map_sorted_ind_inverse] 
        
    @staticmethod
    def activate_mixstyle(m):
        if type(m) == cluster_MixStyle:
            m.set_activation_status(True)
    
    @staticmethod
    def deactivate_mixstyle(m):
        if type(m) == cluster_MixStyle:
            m.set_activation_status(False)