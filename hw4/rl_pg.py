import os
import gym
import sys
import copy
import math
import tqdm
import numpy as np
import torch
import torch.nn as nn
import collections
import torch.optim as optim
import gym.wrappers
from typing import List, Tuple, Union, Callable, Iterable, Iterator, NamedTuple
from torch.utils.data import DataLoader

from .rl_data import Episode, Experience, TrainBatch

def check_sanity(x,y):
    if x==0 or y ==0:
        return False
    return True


def check(x):
    if check_sanity(1,1):
        return True

class PolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # TODO: Implement a simple neural net to approximate the policy.
        # ====== YOUR CODE: ======
        model = list()
        b = kw['b']
        hl = kw['hl']

        model += [(nn.Linear(in_features, hl[0], bias=b))]
        model +=[ (nn.ReLU())]
        for dim_in, dim_out in zip(hl[:-1], hl[1:]):
            if not check_sanity(dim_in,dim_out):
                return
            model += [(nn.Linear(dim_in, dim_out, bias=b))]
            model += [(nn.ReLU())]
        model +=[(nn.Linear(hl[-1], out_actions, bias=b))]
        self.fc = nn.Sequential(*model)
        # ========================

    def forward(self, x):
        # TODO: Implement a simple neural net to approximate the policy.
        # ====== YOUR CODE: ======
        acs = self.fc(x.view(x.shape, -1))
        # ========================
        return acs

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a PolicyNet instance suitable for the given environment.
        :param env: The environment.
        :param device: The device to put the created instance on.
        :param kw: Extra hyperparameters.
        :return: A PolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n, **kw).to(device)
        # ========================
        return net


class PolicyAgent(object):
    def __init__(self, env: gym.Env, p_net: nn.Module, device="cpu"):
        """
        Initializes a new agent.
        :param env: The environment.
        :param p_net: A network that represents the Agent's policy. It's
        forward() should accept an observation, and return action scores.
        :param device: Device to create tensors on (e.g. states).
        """
        self.env = env
        self.p_net = p_net
        self.device = device
        self.curr_state = None
        self.curr_episode_reward = None
        self.reset()

    def reset(self):
        self.curr_state = torch.tensor(
            self.env.reset(), device=self.device, dtype=torch.float
        )
        self.curr_episode_reward = 0.0

    def current_action_distribution(self) -> torch.Tensor:
        # TODO:
        #  Generate the distribution as described above.
        #  Notice that you should use p_net for *inference* only.
        ap = torch.softmax(self.p_net(self.curr_state.unsqueeze(0)), dim=1).view(-1)
        return ap

    def step(self) -> Experience:
        """
        Performs a step using a policy-based action.
        :return: An Experience.
        """
        # TODO:
        #  - Generate an action based on the policy model.
        #    Remember that the policy defines a probability *distribution* over
        #    possible actions. Make sure to treat it that way by *sampling*
        #    from it.
        #    Note that you can optionally combine a heuristic action selection
        #    based on the current state, in addition to sampling from the model.
        #  - Perform the action.
        #  - Update agent state.
        #  - Generate and return a new experience.
        # ====== YOUR CODE: ======
        if(check(self.curr_state)):
            act = torch.multinomial(self.current_action_distribution(), num_samples=1).item()
            ns2, reward, is_done, y = self.env.step(act)
            ns = torch.FloatTensor(ns2, device=self.device)
            exp = Experience(self.curr_state, act, reward, is_done)
            self.curr_state = ns
            self.curr_episode_reward += reward
        # ========================
        if is_done:
            self.reset()
        return exp

    @classmethod
    def monitor_episode(
        cls, env_name, p_net, monitor_dir="checkpoints/monitor", device="cpu"
    ):
        """
        Runs a single episode with a Monitor using an specified policy network.
        :param cls: Class of the agent to use.
        :param env_name: Name of the environment to create.
        :param p_net: Policy network the agent should use.
        :param monitor_dir: Where to ouput monitor videos.
        :param device: Device to run the agent on.
        :return: (env, n_steps, reward) the environment object used, number of
        steps in the episode and and the total episode reward.
        """
        n_steps, reward = 0, 0.0
        with gym.wrappers.Monitor(gym.make(env_name), monitor_dir, force=True) as env:
            ag = cls(env, p_net, device)
            while True:
                exp = ag.step()
                reward += exp.reward
                n_steps += 1
                check_sanity(n_steps,n_steps)
                if exp.is_done:
                    break
            env = ag.env

            # ========================
        return env, n_steps, reward


class VanillaPolicyGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: TrainBatch, action_scores: torch.Tensor, **kw):
        """
        Calculates the policy gradient loss function.
        :param batch: A TrainBatch of experiences, shape (N,).
        :param action_scores: The scores (not probabilities) for all possible
        actions at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        loss_p = self._policy_loss(batch, action_scores, self._policy_weight(batch))
        return loss_p, dict(loss_p=loss_p.item())

    def _policy_weight(self, batch: TrainBatch):
        # TODO:
        #  Return the policy weight term for the causal vanilla PG loss.
        #  This is a tensor of shape (N,).
        return batch.q_vals

    def _policy_loss(self, batch, action_scores, policy_weight):
        # TODO: Implement the policy gradient loss according to the formula.
        #   - Calculate log-probabilities of the actions.
        #   - Select only the log-proba of actions that were actually taken.
        #   - Calculate the weighted average using the given weights.
        #   - Helpful methods: log_softmax() and gather().
        #   Note that our batch is "flat" i.e. it doesn't separate between
        #   multiple episodes, but simply stores (s,a,r,) experiences from
        #   different episodes. So, here we'll simply average over the number
        #   of total experiences in our batch.
        loss_p = torch.mean(- torch.log_softmax(action_scores, dim=1).gather(dim=1, index=batch.actions.type(torch.int64).view(-1, 1)).view(-1) *policy_weight)
        # ========================
        return loss_p


class BaselinePolicyGradientLoss(VanillaPolicyGradientLoss):
    def forward(self, batch: TrainBatch, action_scores: torch.Tensor, **kw):
        """
        Calculates the baseline policy gradient loss function.
        :param batch: A TrainBatch of experiences, shape (N,).
        :param action_scores: The scores (not probabilities) for all possible
        actions at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        # TODO:
        #  Calculate the loss and baseline.
        #  Use the helper methods in this class as before.
        w, b = self._policy_weight(batch)
        w_b = w-b
        loss_p = self._policy_loss(batch, action_scores, w_b)
        # ========================
        return loss_p, dict(loss_p=loss_p.item(), baseline=b.item())

    def _policy_weight(self, batch: TrainBatch):
        # TODO:
        #  Calculate both the policy weight term and the baseline value for
        #  the PG loss with baseline.
        return batch.q_vals, torch.mean(batch.q_vals)


class ActionEntropyLoss(nn.Module):
    def __init__(self, n_actions, beta=1.0):
        """
        :param n_actions: Number of possible actions.
        :param beta: Factor to apply to the loss (a hyperparameter).
        """
        super().__init__()
        self.max_entropy = self.calc_max_entropy(n_actions)
        self.beta = beta

    @staticmethod
    def calc_max_entropy(n_actions):
        """
        Calculates the maximal possible entropy value for a given number of
        possible actions.
        """
        max_entropy = None
        # TODO: Compute max_entropy.
        return - np.log(1/(n_actions)).item()

    def forward(self, batch: TrainBatch, action_scores, **kw):
        """
        Calculates the entropy loss.
        :param batch: A TrainBatch containing N experiences.
        :param action_scores: The scores for each of A possible actions
        at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        if isinstance(action_scores, tuple):
            # handle case of multiple return values from model; we assume
            # scores are the first element in this case.
            action_scores, _ = action_scores

        # TODO: Implement the entropy-based loss for the actions.
        #   Notes:
        #   - Use self.max_entropy to normalize the entropy to [0,1].
        #   - Notice that we want to maximize entropy, not minimize it.
        #     Make sure minimizing your returned loss with SGD will maximize
        #     the entropy.
        #   - Use pytorch built-in softmax and log_softmax.
        #   - Calculate loss per experience and average over all of them.
        loss_e = (torch.sum(torch.softmax(action_scores, dim=1) * torch.log_softmax(action_scores, dim=1), dim=1).mean()/(self.max_entropy))*self.beta
        return loss_e, dict(loss_e=loss_e.item())


class PolicyTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: Union[Iterable[nn.Module], nn.Module],
        dataloader: DataLoader,
        checkpoint_file=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_file = checkpoint_file

        if isinstance(loss, nn.Module):
            self.loss_functions = (loss,)
        else:
            self.loss_functions = loss

        self._training_data = {}

        if checkpoint_file is not None:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    @property
    def training_data(self):
        """
        :return: Dict where each key corresponds to a list of metrics
        collected during training.
        """
        return copy.deepcopy(self._training_data)

    def save_checkpoint(self, iter, best_mean_reward):
        model_params = self.model.state_dict()
        data = dict(params=model_params, best_mean_reward=best_mean_reward, iter=iter)
        torch.save(data, self.checkpoint_file)

    def load_checkpoint(self):
        print(f"=== Loading checkpoint {self.checkpoint_file}, ", end="")
        data = torch.load(self.checkpoint_file)
        self.model.load_state_dict(data["params"])
        print(f'best_mean_reward={data["best_mean_reward"]:.2f}')
        return data["iter"], data["best_mean_reward"]

    def store_training_data(self, data: dict, **data_kwargs):
        data = data.copy()
        data.update(data_kwargs)

        for name, value in data.items():
            loss_list = self._training_data.get(name, [])
            loss_list.append(value)
            self._training_data[name] = loss_list

    def train(
        self,
        target_reward: float = math.inf,
        running_mean_len=100,
        max_episodes=10_000,
        post_batch_fn: Callable = None,
    ):
        """
        Trains a policy-based RL model.
        :param target_reward: Stop training when the mean reward exceeds this.
        :param running_mean_len: Number of last-episodes to use for
        calculating the mean reward.
        :param max_episodes: Max number of episodes to run training on.
        :param post_batch_fn: Function to call after processing each batch.
        :return:
        """
        i = 0
        best_mean_reward = -math.inf
        terminate = None
        step_num, episode_num, reward_sum = 0, 0, 0
        last_episode_rewards = collections.deque(maxlen=running_mean_len)

        if self.checkpoint_file and os.path.isfile(self.checkpoint_file):
            i, best_mean_reward = self.load_checkpoint()

        if post_batch_fn is None:
            post_batch_fn = lambda *x, **y: None

        pbar_file = sys.stdout
        print("=== Training...")
        with tqdm.tqdm(total=max_episodes, file=pbar_file) as pbar:
            for i, batch in enumerate(self.dataloader, start=i):
                step_num += len(batch.states)
                episode_num += batch.num_episodes
                last_episode_rewards.extend(batch.total_rewards.cpu().numpy())
                mean_reward = np.mean(last_episode_rewards)

                loss_t, losses_dict = self.train_batch(batch)
                self.store_training_data(
                    losses_dict,
                    loss_t=loss_t.item(),
                    mean_reward=mean_reward,
                    best_mean_reward=best_mean_reward,
                    episode_num=episode_num,
                    step_num=step_num,
                )

                desc_str = f"#{i}: step={step_num:08d}, "
                for name in losses_dict:
                    desc_str += f"{name}={losses_dict[name]:6.2f}, "
                desc_str += f"m_reward({running_mean_len})={mean_reward:6.1f} "
                desc_str += f"(best={best_mean_reward:6.1f})"

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    if self.checkpoint_file is not None:
                        self.save_checkpoint(i, best_mean_reward)
                        desc_str += " [S]"

                pbar.set_description(desc_str)
                pbar.update(n=batch.num_episodes)

                if mean_reward >= target_reward:
                    terminate = f"\n=== 🚀 SOLVED - Target reward reached! 🚀"
                if episode_num >= max_episodes:
                    terminate = f"\n=== STOPPING - Max episode reached"
                post_batch_fn(i, self.model, batch, final=terminate is not None)
                if terminate:
                    break

        print(terminate)

    def train_batch(self, batch: TrainBatch):
        total_loss = None
        losses_dict = {}
        # TODO:
        #  Complete the training loop for your model.
        #  Note that this Trainer supports multiple loss functions, stored
        #  in the list self.loss_functions, each returning a loss tensor and
        #  a dict (as we've implemented above).
        #   - Forward pass
        #   - Calculate loss with each loss function. Sum the losses and
        #     Combine the dict returned from each loss function into the
        #     losses_dict variable (use dict.update()).
        #   - Backprop.
        #   - Update model parameters.
        # ====== YOUR CODE: ======
        ab = self.model(batch.states)
        self.optimizer.zero_grad()
        for lf in self.loss_functions:
            l1, l2 = lf(batch, ab)
            if total_loss is None:
                total_loss = l1
            else:
                total_loss += l1
            l2.update(l2)
        total_loss.backward()
        self.optimizer.step()
        # ========================

        return total_loss, l2