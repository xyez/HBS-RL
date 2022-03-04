import torch as t
import torch.optim as optim

from bitselect.PPO.networks import PolicyNet, VNet


class Agent():
    def __init__(self, obsv_dim, act_dim, lr, device='cpu', discount_factor=0.99, target_kl=0.01, ent_coef=0.01, rho=0.2, clip_v=True, is_train=True):
        self.obsv_dim = obsv_dim
        self.act_dim = act_dim
        self.lr = lr
        self.device = device
        self.discount_factor = discount_factor
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.rho = rho
        self.clip_v = clip_v
        self.is_train = is_train

        self.small_number = 1e-6

        self.pi = PolicyNet(obsv_dim, act_dim)
        self.pi.to(self.device)
        if self.is_train:
            self.v = VNet(obsv_dim, act_dim)
            self.v.to(device)

            self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr)
            self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr)

    def get_action(self, state):
        with t.no_grad():
            prob = self.pi(state)  # NE,NB
            if self.is_train:
                action = prob.multinomial(1)  # NE
                logprob = (prob + 1e-6).log()
                logp = logprob.gather(dim=1, index=action).squeeze(-1)
                v = self.v(state)
                return action, logp, v
            else:
                action = prob.argmax(dim=1, keepdim=True)
                return action

    def compute_loss(self, inputs, train_pi_iters, train_v_iters):
        obsv, acts, logp_a_old, advs, rets, vals = inputs
        # shape
        # obsv: B,NE,S
        # acts: B,NE
        # logp_a_old: B,NE
        # advs: B,NE
        # rets: B,NE
        for i in range(train_pi_iters):
            prob = self.pi(obsv)  # B,NE,S
            logp = (prob + self.small_number).log()
            logp_a = logp.gather(dim=-1, index=acts.unsqueeze(-1)).squeeze(-1)  # B,NE
            ratio = (logp_a - logp_a_old).exp()  # B,NE
            loss_pi = -t.min(ratio * advs, ratio.clamp(1 - self.rho, 1 + self.rho) * advs).mean()

            if self.ent_coef > 0:
                loss_ent = self.ent_coef * (prob * logp).sum(-1).mean()
                loss_pi += loss_ent

            approx_kl = (logp_a_old - logp_a).mean().item()
            clip_frac = (ratio.gt(1 + self.rho) | ratio.lt(1 - self.rho)).float().mean().item()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()
            if i == 0:
                loss_pi_old = loss_pi.item()
                ent_old = -(prob * logp).sum(-1).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break
        for i in range(train_v_iters):
            v = self.v(obsv)  # B,NE
            if self.clip_v:
                v_clipped = vals[:-1] + (v - vals[:-1]).clamp(-self.rho, self.rho)
                v_loss_clipped = (rets - v_clipped).pow(2)
                v_loss = (rets - v).pow(2)
                loss_v = t.max(v_loss, v_loss_clipped).mean()
            else:
                loss_v = ((v - rets) ** 2).mean()
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()
            if i == 0:
                loss_v_old = loss_v.item()
        info = {'LossPi': loss_pi_old, 'LossV': loss_v_old, 'KL': approx_kl, 'Entropy': ent_old, 'ClipFrac': clip_frac,
                'DeltaLossPi': loss_pi.item() - loss_pi_old, 'DeltaLossV': loss_v.item() - loss_v_old, 'V': v.mean().item()}
        return info

    def compute_value(self, state):
        with t.no_grad():
            v = self.v(state)
        return v

    def load_param(self, state_dicts):
        self.pi.load_state_dict(state_dicts['pi'])
        if self.is_train:
            self.v.load_state_dict(state_dicts['v'])
            self.optimizer_pi.load_state_dict(state_dicts['pi'])
            self.optimizer_v.load_state_dict(state_dicts['v'])

    def get_param(self):
        state_dicts = {
            'pi': self.pi.state_dict(),
            'v': self.v.state_dict(),
            'optim_pi': self.optimizer_pi.state_dict(),
            'optim_v': self.optimizer_v.state_dict()
        }
        return state_dicts
