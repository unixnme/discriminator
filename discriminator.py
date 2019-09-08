import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class Discriminator(nn.Module):
    def __init__(self, config_file:str):
        '''
        given json config file, create a discriminator
        :param config_file: json config file path
        '''
        super().__init__()
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        dims = [int(d) for d in model_config['dims'].split(',')]
        drop = model_config['drop']
        layers = []
        for idx in range(len(dims) - 2):
            input_dim = dims[idx]
            output_dim = dims[idx + 1]
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.Dropout(drop))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.sequential = nn.Sequential(*layers)


    def forward(self, x):
        return self.sequential(x)


if __name__ == '__main__':
    discriminator = Discriminator('disc.json')
    optim = torch.optim.SGD(discriminator.parameters(), 1e-3, .95)

    for epoch in range(1000):
        optim.zero_grad()

        true_x = torch.randn(64,16)
        false_x = (torch.rand_like(true_x) - 0.5) * 12**0.5
        true_label = torch.ones(64, dtype=torch.long)
        false_label = torch.zeros_like(true_label)
        y1 = discriminator(true_x)
        y2 = discriminator(false_x)
        loss1 = F.cross_entropy(y1, true_label)
        loss2 = F.cross_entropy(y2, false_label)
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        print(loss.item())