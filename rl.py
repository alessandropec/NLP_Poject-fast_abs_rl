""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]
            all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=2)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch = next(loader)
    for raw_arts in art_batch:
        (inds, ms), bs = agent(raw_arts)
        # inds = indices of the extracted sentences
        # ms = probability distribution over the sentences
        # bs = baseline, so the predicted expected reward for all possible action
        # the baseline is obtained through regression by adding a linear layer on the logits of the PointerNet
        
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts)]
    reward=None
    
    with torch.no_grad():
        summaries = abstractor(ext_sents)
    print("Sum",summaries,"\n")
    i = 0 # start of the sliding window per batch, this will have to be updated with the length of the current element in the batch
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch): # indices of the extracted sentences and golden summaries sentences
        rs = ([reward_fn(summaries[i+j], abss[j])
            for j in range(min(len(inds)-1, len(abss)))]
            + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
            + [stop_coeff*stop_reward_fn(
                list(concat(summaries[i:i+len(inds)-1])),
                list(concat(abss)))])
        
        # in rs we have a lot of stuff summed, lets understand why:
        # first of all note that rs is the same length as the number of extracted sentences
        # it is the sum of three different lists:
            # 1) for each extracted sentence calculate the rouge reward with the corresponding golden summary sentence.
            # 2) if the extracted sentences are more than the golden summary sentences, then pad with 0.
            # 3) finally the stop reward, where we look at the reward associated with stopping at that particular number of sentences
            #    calculated as the total rouge score between extracted and golden summary sentences multiplied by the stop coefficient.

        assert len(rs) == len(inds)

        avg_reward += rs[-1]/stop_coeff  # as avg reward we use the rouge calculated between all the extracted and golden summary sentences

        i += len(inds)-1 # move the window to the next batch

        # compute discounted rewards -> discount factor essentially determines how much the reinforcement learning agents cares about rewards in the distant future relative to those in the immediate future
        R = 0
        disc_rs = []
        for r in rs[::-1]: # take each element in rs starting from the last in reverse order
            R = r + gamma * R # sum the current reward and multiply the previous ones by the discount factor, since we care less about rewards in the future
            disc_rs.insert(0, R) # put R at the first index of the list disc_rs
        rewards += disc_rs # concat to the rewards list (list because one for each element in the batch)
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        # action is the extracted sentence
        # p is the probability distribution at that step of extracting sentences
        # r is the discont reward we calculated
        # b is the baseline (i.e. the predicted expected reward for all possible actions at a certain time step) needed to calculate the advantage
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action) # log_prob returns the log of the probability density/mass function evaluated at value
                    * (advantage/len(indices))) # for the reward in the case of A2C we use the avg advantage
    critic_loss = F.mse_loss(baseline, reward) # to train the critic we use the MSE loss (quadratic)
    autograd.backward(
    [critic_loss.unsqueeze(0)] + losses,
    [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    
    
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
