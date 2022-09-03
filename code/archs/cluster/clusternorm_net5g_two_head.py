import torch.nn as nn

from code.archs.cluster.net5g import ClusterNorm_Net5gTrunk
from code.archs.cluster.residual import BasicBlock, BasicBlock_MixStyle, ResNet

# resnet34 and full channels

__all__ = ["ClusterNormNet5gTwoHead"]


class ClusterNormNet5gTwoHeadHead(nn.Module):
  def __init__(self, config, output_k, semisup=False):
    super(ClusterNormNet5gTwoHeadHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.semisup = semisup

    if not semisup:
      self.num_sub_heads = config.num_sub_heads

      self.heads = nn.ModuleList([nn.Sequential(
        nn.Linear(512 * BasicBlock.expansion, output_k),
        nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])
    else:
      self.head = nn.Linear(512 * BasicBlock.expansion, output_k)

  def forward(self, x, kmeans_use_features=False):
    if not self.semisup:
      results = []
      for i in range(self.num_sub_heads):
        if kmeans_use_features:
          results.append(x)  # duplicates
        else:
          results.append(self.heads[i](x))
      return results
    else:

      return self.head(x)


class ClusterNormNet5gTwoHead(ResNet):
  def __init__(self, config):
    # no saving of configs
    super(ClusterNormNet5gTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNorm_Net5gTrunk(config)

    self.head_A = ClusterNormNet5gTwoHeadHead(config, output_k=config.output_k_A)

    semisup = (hasattr(config, "semisup") and
               config.semisup)
    print("semisup: %s" % semisup)

    self.head_B = ClusterNormNet5gTwoHeadHead(config, output_k=config.output_k_B,
                                          semisup=semisup)

    self._initialize_weights()

  def forward(self, x, head="B", kmeans_use_features=False,
              trunk_features=False,
              penultimate_features=False,
              cluster_map = None):
    # default is "B" for use by eval code
    # training script switches between A and B

    x = self.trunk(x, penultimate_features=penultimate_features, cluster_map = cluster_map)

    if trunk_features:  # for semisup
      return x

    # returns list or single
    if head == "A":
      x = self.head_A(x, kmeans_use_features=kmeans_use_features)
    elif head == "B":
      x = self.head_B(x, kmeans_use_features=kmeans_use_features)
    else:
      assert (False)

    return x
