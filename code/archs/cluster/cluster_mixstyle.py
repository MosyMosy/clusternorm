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
        print(cluster_map_count)
        cluster_map_sorted_ind = torch.argsort(cluster_map_one_hot, dim=0)        
        clustered_samples = torch.split(x[cluster_map_sorted_ind], cluster_map_count.tolist())
        clustered_samples = torch.stack(clustered_samples).to(x.device)
        
        cluster_map_sorted_ind_inverse = torch.argsort(cluster_map_sorted_ind, dim=0)
            
        # Statistics over each cluster. Keep clusters and channels
        cluster_mu = clustered_samples.mean(dim=[1, 3, 4], keepdim=True).detach()        
        cluster_mu = torch.flatten(cluster_mu, end_dim=0)        
        cluster_mu = torch.repeat_interleave(cluster_mu, cluster_map_count)
        cluster_mu = cluster_mu[cluster_map_sorted_ind_inverse]
        
        cluster_std = ((clustered_samples.var(dim=[1, 3, 4], keepdim=True) + self.eps).sqrt()).detach()
        cluster_std = torch.flatten(cluster_std, end_dim=0)
        cluster_std = torch.repeat_interleave(cluster_std, cluster_map_count)
        cluster_std = cluster_std[cluster_map_sorted_ind_inverse]
        
        # Statistics over sample's spatial dimensions. Keep clusters, samples and channels
        sample_mu = x.mean(dim=[2, 3], keepdim=True).detach()
        sample_std = ((x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()).detach()
        
        x_normed = (x-sample_mu) / sample_std            
        mu_mix = sample_mu * lmda + cluster_mu, 1 * (1-lmda)
        std_mix = sample_std * lmda + cluster_std, 1 * (1-lmda)
                
        return x_normed * std_mix + mu_mix
                
    @staticmethod
    def activate_mixstyle(m):
        if type(m) == cluster_MixStyle:
            m.set_activation_status(True)
    
    @staticmethod
    def deactivate_mixstyle(m):
        if type(m) == cluster_MixStyle:
            m.set_activation_status(False)