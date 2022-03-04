import typing

import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias.data, 0.0)
        nn.init.orthogonal_(m.weight.data, gain=1.41)


class PolicyNet(nn.Module):
    def __init__(self, obsv_dim, act_dim, hidden_size=(256, 256)):
        super().__init__()
        self.act_dim = act_dim
        self.net = [nn.Linear(obsv_dim, hidden_size[0]), nn.ReLU(inplace=True)]
        for i in range(len(hidden_size) - 1):
            self.net.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_size[-1], act_dim))
        self.net = nn.Sequential(*self.net)
        self.apply(weight_init)

        self.mask_number = -1e7

    def forward(self, x):
        logits = self.net(x)
        # if x[1].sum()==17:
        #     print('here')
        logits[x > 0.5] = self.mask_number
        prob = F.softmax(logits, dim=-1)
        return prob

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


class VNet(nn.Module):
    def __init__(self, obsv_dim, act_dim, hidden_size=(256, 256)):
        super().__init__()
        self.net = [nn.Linear(obsv_dim, hidden_size[0]), nn.ReLU(inplace=True)]
        for i in range(len(hidden_size) - 1):
            self.net.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_size[-1], 1))
        self.net = nn.Sequential(*self.net)
        self.apply(weight_init)

    def forward(self, obs):
        qs = self.net(obs).squeeze(-1)
        return qs

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)
