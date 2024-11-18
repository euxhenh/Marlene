"""
MIT License

Copyright (c) 2019 Jackie Loong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Based on the implementation of https://github.com/dragen1860/MAML-Pytorch
"""


from collections import OrderedDict
from typing import Callable

import torch
from torch import nn
from torch.func import functional_call


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(
        self,
        model: nn.Module,
        meta_optim: torch.optim.Optimizer,
        loss: Callable,
        update_lr: float,
        update_step: int = 5,
        update_step_test: int = 10,
        gradient_clip: float = 0.1
    ):
        super().__init__()

        self.update_lr = update_lr
        self.update_step = update_step
        self.update_step_test = update_step_test

        self.net = model
        self.meta_optim = meta_optim
        self.loss = loss
        self.gradient_clip = gradient_clip

    def forward(self, x_support_batch, y_support_batch, x_query_batch, y_query_batch):
        losses_q = [0 for _ in range(self.update_step + 1)]

        accuracy = 0.0

        for x_support, y_support, x_query, y_query in zip(
            x_support_batch, y_support_batch, x_query_batch, y_query_batch
        ):
            # forward
            y_support_pred, _ = self.net(x_support)
            loss = self.loss(y_support_pred, y_support)

            grad = torch.autograd.grad(loss, self.net.parameters())
            self.clip_grad_by_norm_(grad, self.gradient_clip)
            fast_weights = self.update_weights(self.net.named_parameters(), grad)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                y_query_pred, _ = functional_call(
                    self.net,
                    OrderedDict(self.net.named_parameters()),
                    x_query,
                )
                # reconstruction loss
                loss_q = self.loss(y_query_pred, y_query)
                losses_q[0] += loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                y_query_pred, _ = functional_call(
                    self.net,
                    fast_weights,
                    x_query,
                )
                loss_q = self.loss(y_query_pred, y_query)
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                y_support_pred, _ = functional_call(
                    self.net,
                    fast_weights,
                    x_support,
                )
                loss = self.loss(y_support_pred, y_support)
                grad = torch.autograd.grad(loss, fast_weights.values())
                self.clip_grad_by_norm_(grad, self.gradient_clip)
                fast_weights = self.update_weights(fast_weights.items(), grad)

                y_query_pred, _ = functional_call(
                    self.net,
                    fast_weights,
                    x_query,
                )
                # loss_q will be overwritten and just keep the loss_q on
                # last update step.
                loss_q = self.loss(y_query_pred, y_query)
                losses_q[k + 1] += loss_q

            if y_query_pred.argmax() == y_query:
                accuracy += 1

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / len(x_support_batch)
        accuracy /= len(x_support_batch)

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip)
        self.meta_optim.step()

        return loss_q.item(), accuracy

    def update_weights(self, weights, grads, lr: float | None = None):
        lr = lr or self.update_lr

        weights = OrderedDict(
            (name, param - lr * g)
            for (name, param), g in zip(weights, grads)
        )

        return weights

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
