# import torch
# import torch.nn as nn

# class NeuralMemory(nn.Module):
#     def __init__(self, dim=128):
#         super().__init__()
#         self.dim = dim

#         self.memory = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.SiLU(),
#             nn.Linear(dim * 2, dim)
#         )

#         # Chỉ lưu trạng thái surprise để tham khảo, không update parameter inplace trong forward
#         self.suprise_metrics = [torch.zeros_like(p) for p in self.memory.parameters()]

#         self.Q_mem_prj = nn.Linear(dim, dim)
#         self.K_mem_prj = nn.Linear(dim, dim)
#         self.V_mem_prj = nn.Linear(dim, dim)

#         self.theta = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.SiLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),
#         )
#         self.eta = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.SiLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),
#         )
#         self.alpha = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.SiLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),
#         )

#     def hyperparameters(self, x):
#         theta = self.theta(x).mean()  # theta là một scalar, lấy mean của batch
#         eta = self.eta(x).mean()  # eta là một scalar, lấy mean của batch
#         alpha = self.alpha(x).mean()  # alpha là một scalar, lấy mean của batch
#         return theta, eta, alpha
    
#     def write(self, y):
#         # detach y để memory loss không phá graph chính của attention
#         y_detached = y.detach()

#         K = self.K_mem_prj(y_detached)
#         V = self.V_mem_prj(y_detached)

#         pred = self.memory(K)
#         loss_mem = ((pred - V) ** 2).mean()

#         theta, eta, alpha = self.hyperparameters(y_detached)

#         # Không được sửa self.memory.parameters() bằng p.mul_ / p.add_ ở đây.
#         # Vì forward chưa backward xong, inplace update parameter sẽ gây lỗi:
#         # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#         if self.training and torch.is_grad_enabled():
#             grads_mem = torch.autograd.grad(
#                 loss_mem,
#                 self.memory.parameters(),
#                 create_graph=False,
#                 retain_graph=False,
#                 allow_unused=True
#             )

#             new_S = []
#             for s, g in zip(self.suprise_metrics, grads_mem):
#                 s = s.to(y.device)
#                 if g is None:
#                     new_S.append(s.detach())
#                 else:
#                     new_S.append((eta * s - theta * g.detach()).detach())

#             self.suprise_metrics = new_S
#             with torch.no_grad():
#                 for p, s in zip(self.memory.parameters(), self.suprise_metrics):
#                     p.mul_(1-alpha).add_(s)

#         return loss_mem.detach()

#     def read(self, x):
#         Q = self.Q_mem_prj(x)
#         out = self.memory(Q)
#         return out
# # Chạy

import torch
import torch.nn as nn


class NeuralMemory(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.dim = dim

        self.memory = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        self.suprise_metrics = [
            torch.zeros_like(p)
            for p in self.memory.parameters()
        ]

        self.Q_mem_prj = nn.Linear(dim, dim)
        self.K_mem_prj = nn.Linear(dim, dim)
        self.V_mem_prj = nn.Linear(dim, dim)

        self.theta = nn.Sequential(
            nn.Linear(dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.eta = nn.Sequential(
            nn.Linear(dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Sequential(
            nn.Linear(dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def hyperparameters(self, x):
        theta = self.theta(x).mean()
        eta = self.eta(x).mean()
        alpha = self.alpha(x).mean()

        return theta, eta, alpha

    def read(self, x):
        Q = self.Q_mem_prj(x)
        return self.memory(Q)

    def memory_loss(self, y):
        """
        Chỉ tính loss.
        KHÔNG update memory parameter ở đây.
        """

        y = y.detach()

        K = self.K_mem_prj(y)
        V = self.V_mem_prj(y)

        pred = self.memory(K)

        loss_mem = ((pred - V) ** 2).mean()

        return loss_mem.detach()

    def update_memory(self, y):
        """
        Manual update Neural Memory.
        Được gọi SAU optimizer.step().
        """

        y = y.detach()

        K = self.K_mem_prj(y)
        V = self.V_mem_prj(y)

        pred = self.memory(K)

        loss_mem = ((pred - V) ** 2).mean()

        theta, eta, alpha = self.hyperparameters(y)

        grads_mem = torch.autograd.grad(
            loss_mem,
            self.memory.parameters(),
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )

        new_S = []

        for s, g in zip(self.suprise_metrics, grads_mem):

            s = s.to(y.device)

            if g is None:
                new_S.append(s.detach())

            else:
                new_S.append(
                    (eta * s - theta * g.detach()).detach()
                )

        self.suprise_metrics = new_S

        # Manual parameter update
        with torch.no_grad():

            for p, s in zip(
                self.memory.parameters(),
                self.suprise_metrics
            ):
                p.mul_(1 - alpha)
                p.add_(s)