import numpy as np
import torch
from torch import nn
import torch.nn.init as init

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from src.bandit import BasicBandit
from typing import List

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from copy import deepcopy


tkwargs = {
    "device": torch.device("cuda:0"),
    "dtype": torch.double,
}


class Network(nn.Module):
    def __init__(self, input_dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
        if init_params is None:
            ## use initialization
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            ### manually set the initialization vector
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y




class NeuralUCB(BasicBandit):
    def __init__(self, T, contexts, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):

        self.diagonalize = diagonalize
        self.num_arms = len(contexts)   
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.context_list = init_x.to(dtype=torch.float32)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(dtype=torch.float32)
        else:
            self.reward = None
            
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            ### diagonalization
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            ### no diagonalization
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.nu = nu
        self.style = style
        self.contexts = np.array(contexts)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
        self.T = T
        self.interval = np.floor(self.T / self.num_arms)
        if self.interval == 0:
            self.interval = 32
        self.X = []
        self.y = []
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None


    def update(self, arm, reward):
        self.X.append(self.contexts[arm])
        self.y.append(reward)
        self.len += 1

        self.t += 1

        if self.t % self.interval == 0:
            self.train()
            
            
    def choose_action(self):
        
        return self.select()
    
    
    
    def best_arm(self):

        # return self.select()
        context = torch.from_numpy(self.contexts).to(**tkwargs)    
        best_id = np.argmax(self.func(context).detach().cpu().numpy())
        # breakpoint()
        return best_id
    
    
    def reset(self):
        
        self.X = []
        self.y = []
        self.t = 0


    def select(self, batch_size=64):
        context = torch.from_numpy(self.contexts).to(**tkwargs)     
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context
        # batch computing of jacobian
        # batch_size = 300
        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_list.append(g_list_.cpu())
            mu.append(mu_.cpu())
        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)
        # mu = self.func(context).cpu()
        # sum_mu = torch.sum(mu)
        # with backpack(BatchGrad()):
        #     sum_mu.backward()

        # g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1).cpu()

        if self.diagonalize:
#             ### diagonalization
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        else:
            ### no diagonalization
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(self.nu * self.lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if self.diagonalize:
            ### diagonalization
            self.U += g_list[arm] * g_list[arm]
        else:
            ### no diagonalization
            self.U += torch.outer(g_list[arm], g_list[arm])

        return arm


    def train(self, local_training_iter=100):

        context, reward = np.array(self.X), np.array(self.y)
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        self.context_list = torch.from_numpy(context).to(**tkwargs)
        self.reward = torch.from_numpy(reward).to(**tkwargs)
        # if context is not None:
        #     if self.context_list is None:
        #         self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**tkwargs)
        #         self.reward = torch.tensor([reward]).to(**tkwargs)
        #     else:
        #         self.context_list = torch.cat((self.context_list, context.reshape(1, -1).to(**tkwargs)))
        #         self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(1,-1).to(**tkwargs)))

        self.len = self.context_list.shape[0]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba / self.len)

        # if self.len % self.delay != 0:
        #     return 0
        # torch.save({"context_list": self.context_list, "reward": self.reward}, 'train_data.pt')

        self.std = self.context_list.std(dim=0) + 1e-30
        self.mean = self.context_list.mean(dim=0)
        standardized_context = (self.context_list - self.mean) / self.std 
        # standardized_reward = ((self.reward - self.reward.mean(dim=0)) / (self.reward.std(dim=0) + 1e-30)).reshape(-1)
        standardized_reward = self.reward.reshape(-1)
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            # breakpoint()
            pred = self.func(standardized_context).view(-1)
        
            loss = self.loss_func(pred, standardized_reward)
            loss.backward()
            optimizer.step()
        print("Training Loss : ", loss.item())
        return self.func.state_dict()