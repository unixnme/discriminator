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
        self.input_dim = dims[0]
        dims[0] += 2 # max & min
        drop = model_config['drop']
        layers = []
        for idx in range(len(dims) - 2):
            input_dim = dims[idx]
            output_dim = dims[idx + 1]
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.Dropout(drop))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.sequential = nn.Sequential(*layers)


    def forward(self, x):
        max_x = torch.max(x, -1, keepdim=True)[0]
        min_x = torch.min(x, -1, keepdim=True)[0]
        x = torch.cat([x, max_x, min_x], -1)
        return self.sequential(x)


if __name__ == '__main__':
    discriminator = Discriminator('disc.json')
    optim = torch.optim.SGD(discriminator.parameters(), 1e-3, .95)
    batch_size = 4
    channel = discriminator.input_dim

    for epoch in range(10000):
        optim.zero_grad()

        true_x = torch.randn(batch_size, channel)
        false_x = (torch.rand_like(true_x) - 0.5) * 12**0.5
        true_label = torch.ones(batch_size, dtype=torch.long)
        false_label = torch.zeros_like(true_label)
        y1 = discriminator(true_x)
        y2 = discriminator(false_x)
        correct1 = y1.argmax(-1) == 1
        correct2 = y2.argmax(-1) == 0
        correct = (correct1.sum() + correct2.sum()).type(torch.float).item() / (len(y1) + len(y2))
        loss1 = F.cross_entropy(y1, true_label)
        loss2 = F.cross_entropy(y2, false_label)
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        print(loss.item(), correct * 100)