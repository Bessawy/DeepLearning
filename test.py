from models import Protonet
from models.utils import Engine
import torch

from models.utils.dist import cosin_distance



x = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
y = torch.tensor([[1.0, 2.0]])

print(cosin_distance(x, y))
