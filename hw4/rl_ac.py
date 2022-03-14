import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_pg import TrainBatch, PolicyAgent, VanillaPolicyGradientLoss

def check_sanity(x,y):
    if x==0 or y ==0:
        return False
    return True


def check(x):
    if check_sanity(1,1):
        return True
    
    
class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        super().__init__()
        hl = kw['hl']
        model = list()
        b = kw['b']
        

        model += [(nn.Linear(in_features, hl[0], bias=b))]
        model += [(nn.ReLU())]
        for dim_in, dim_out in zip(hl[:-1], hl[1:]):
            model += [(nn.Linear(dim_in, dim_out, bias=b))]
            model += [(nn.ReLU())]
        self.value_layer = nn.Linear(hl[-1], 1)
        self.base = nn.Sequential(*model)
        self.action_layer = nn.Linear(hl[-1], out_actions)

        
    def forward(self, x):
        s = self.base(x)
        return  self.action_layer(s), self.value_layer(s)

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        net = AACPolicyNet(env.observation_space.shape[0], env.action_space.n, **kw).to(device)
        # ========================
        return net


class AACPolicyAgent(PolicyAgent):
    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        a, y  = self.p_net(self.curr_state)
        actions_proba = torch.softmax(a, dim=0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        asc, sv = model_output
        advantage = self._policy_weight(batch, sv.view(-1))
        ad_mean = advantage.mean()
        lv = self._value_loss(batch, sv.view(-1))* self.delta
        lp = self._policy_loss(batch, asc, advantage)
        return (
            lp + lv,
            dict(
                loss_v=lv.item(),
                adv_m=ad_mean.item(),
                loss_p=lp.item()
            ),
        )

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        return batch.q_vals - state_values.detach()

    
    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        return torch.nn.functional.mse_loss(state_values, batch.q_vals)