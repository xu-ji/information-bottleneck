from torch import nn
import torch
import numpy as np
from dnn.swag_repo.swag.models.preresnet import add_tag

DETACH = False
EPS = 1e-10

class StochasticMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.z_sz = arch[-2]

        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - (2)): # if there are 5 layers, will loop l = 1
            layers += [nn.ReLU(), nn.Linear(arch[l], arch[l+1])]

        print(layers)

        self.pred_mu = nn.Sequential(*layers)

        sigma_layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - (2)): # if there are 5 layers, will loop l = 1
            sigma_layers += [nn.ReLU(), nn.Linear(arch[l], arch[l+1])]

        self.pred_sigma = nn.Sequential(*(sigma_layers + [nn.Softplus()])) # nn.Sigmoid(), Limit()

        self.cls = nn.Sequential(
            nn.Linear(arch[-2], arch[-1]),
        )

        self.pred_mu.apply(lambda module: add_tag(module, 0))
        self.pred_sigma.apply(lambda module: add_tag(module, 0))

        self.cls.apply(lambda module: add_tag(module, 1))


    def forward(self, x, repr=False):
        means = self.pred_mu(x) # n, enc_sz
        stds = self.pred_sigma(x).clamp(min=EPS)

        eps = torch.randn_like(means)
        z = means + stds * eps

        if repr:
            distr = torch.distributions.normal.Normal(means, stds)  # batch_sz, L for each
            z_prob = z
            if DETACH: z_prob = z.detach()
            logprob = distr.log_prob(z_prob).sum(dim=1) # batch_sz
            return z, logprob

        return self.cls(z), stds.mean().item()


    def log_marg_prob(self, z, d_x, jensen):
        batch_sz, L = z.shape
        batch_sz2 = d_x.shape[0]

        means = self.pred_mu(d_x)  # n, enc_sz
        stds = self.pred_sigma(d_x).clamp(min=EPS)

        # for each target, pass through each mean
        means = means.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        stds = stds.unsqueeze(0).expand(batch_sz, batch_sz2, L)

        z = z.unsqueeze(1).expand(batch_sz, batch_sz2, L)

        distr = torch.distributions.normal.Normal(means, stds)
        z_prob = z
        if DETACH: z_prob = z.detach()
        logprob = distr.log_prob(z_prob)
        assert logprob.shape == (batch_sz, batch_sz2, L)

        logprob = logprob.sum(dim=2) # batch_sz, batch_sz2, logprob of each code, was missing before!
        if jensen:
            log_margprob = logprob.mean(dim=1) # est
        else:
            log_margprob = - np.log(batch_sz2) + torch.logsumexp(logprob, dim=1)

        assert log_margprob.shape == (batch_sz,)

        return log_margprob # batch_sz


class StochasticConvMLP(nn.Module):
    def __init__(self, arch, in_dim):
        super().__init__()
        self.arch = arch

        self.z_sz = arch[2]

        layers = [
            nn.Conv2d(arch[0], arch[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(arch[1] * int(in_dim[0] * in_dim[1] / 4), arch[2]),
            nn.ReLU()
        ]
        self.pre_feats = nn.Sequential(*layers)

        self.pred_mu = nn.Linear(arch[2], arch[3])

        self.pred_sigma = nn.Sequential(*([nn.Linear(arch[2], arch[3])] + [nn.Softplus()])) # nn.Sigmoid(), Limit()

        """

        self.pred_feat = nn.Sequential(
            nn.Linear(arch[3], arch[4]),
            nn.ReLU(),
            #nn.Linear(arch[3], arch[4]),
            #nn.ReLU(),
        )
        """
        self.cls = nn.Sequential(
            #nn.Linear(arch[4], arch[5]),
            nn.Linear(arch[3], arch[4]),
        )

        self.pre_feats.apply(lambda module: add_tag(module, 0))
        self.pred_mu.apply(lambda module: add_tag(module, 0))
        self.pred_sigma.apply(lambda module: add_tag(module, 0))
        #self.pred_feat.apply(lambda module: add_tag(module, 0))

        self.cls.apply(lambda module: add_tag(module, 1))


    def forward(self, x, repr=False):
        feats = self.pre_feats(x)
        means = self.pred_mu(feats) # n, enc_sz
        stds = self.pred_sigma(feats).clamp(min=EPS)

        eps = torch.randn_like(means)
        z = means + stds * eps

        if repr:
            distr = torch.distributions.normal.Normal(means, stds)  # batch_sz, L for each
            z_prob = z
            if DETACH: z_prob = z.detach()
            logprob = distr.log_prob(z_prob).sum(dim=1) # batch_sz
            return z, logprob

        return self.cls(z), stds.mean().item()


    def log_marg_prob(self, z, d_x, jensen):
        batch_sz, L = z.shape
        batch_sz2 = d_x.shape[0]

        feats = self.pre_feats(d_x)
        means = self.pred_mu(feats)  # n, enc_sz
        stds = self.pred_sigma(feats).clamp(min=EPS)

        # for each target, pass through each mean
        means = means.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        stds = stds.unsqueeze(0).expand(batch_sz, batch_sz2, L)

        z = z.unsqueeze(1).expand(batch_sz, batch_sz2, L)

        distr = torch.distributions.normal.Normal(means, stds)
        z_prob = z
        if DETACH: z_prob = z.detach()
        logprob = distr.log_prob(z_prob)
        assert logprob.shape == (batch_sz, batch_sz2, L)

        logprob = logprob.sum(dim=2) # batch_sz, batch_sz2, logprob of each code, was missing before!
        if jensen:
            log_margprob = logprob.mean(dim=1) # est
        else:
            log_margprob = - np.log(batch_sz2) + torch.logsumexp(logprob, dim=1)

        assert log_margprob.shape == (batch_sz,)

        return log_margprob # batch_sz


class BasicConvMLP(nn.Module):
    def __init__(self, arch, in_dim):
        super().__init__()
        self.arch = arch

        self.z_sz = arch[2]

        layers = [
            nn.Conv2d(arch[0], arch[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
        ]
        self.pred_feats1 = nn.Sequential(*layers)

        self.pred_feats2 = nn.Sequential(nn.Linear(arch[1] * int(in_dim[0] * in_dim[1] / 4), arch[2]), nn.ReLU())

        self.pred_feats3 = nn.Sequential(nn.Linear(arch[2], arch[3]), nn.ReLU())

        self.cls = nn.Sequential(
            nn.Linear(arch[3], arch[4]),
        )

        self.pred_feats1.apply(lambda module: add_tag(module, 0))
        self.pred_feats2.apply(lambda module: add_tag(module, 1))
        self.pred_feats3.apply(lambda module: add_tag(module, 2))
        self.cls.apply(lambda module: add_tag(module, 3))


    def forward(self, x, repr=False):
        feats1 = self.pred_feats1(x)
        feats2 = self.pred_feats2(feats1)
        feats3 = self.pred_feats3(feats2)

        if repr:
            # return 3 features
            return [feats1, feats2, feats3]

        return self.cls(feats3), None


class BasicMLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        self.layer1 = nn.Sequential(nn.Linear(arch[0], arch[1]), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(arch[1], arch[2]), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(arch[2], arch[3]), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(arch[3], arch[4]), nn.ReLU())

        self.cls = nn.Linear(arch[4], arch[5])

        self.layer1.apply(lambda module: add_tag(module, 0))
        self.layer2.apply(lambda module: add_tag(module, 1))
        self.layer3.apply(lambda module: add_tag(module, 2))
        self.layer4.apply(lambda module: add_tag(module, 3))
        self.cls.apply(lambda module: add_tag(module, 4))


    def forward(self, x, repr=False):
        feats1 = self.layer1(x)
        feats2 = self.layer2(feats1)
        feats3 = self.layer3(feats2)
        feats4 = self.layer4(feats3)

        if repr:
            return [feats1, feats2, feats3, feats4]

        return self.cls(feats4), None


