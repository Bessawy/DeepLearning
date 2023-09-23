from models import Protonet
from models.utils import Engine
import torch

engine = Engine()
## you can also pass your own encode Protonet(encoder)
model = Protonet.defualt_encoder()



engine.train(
    model = model,
    loader = train_loader,
    optim_method = torch.optim.SGD,
    optim_config = { 'lr': 0.1},
    max_epoch = 100
)