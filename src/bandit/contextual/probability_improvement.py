import numpy as np
import torch
import torch.nn.init as init
import time
import random

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_model
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.kernels.kernel import Kernel
from torch.optim import Adam
from src.bandit import BasicBandit
from typing import List


class SimpleMLP(torch.nn.Module):
    # Initialize the model with nomalized weights
    # set parameters to dtype float32
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        init.normal_(self.fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return output
    
    


class CombinedStringKernel(Kernel):
        def __init__(self, base_latent_kernel, instruction_kernel, latent_train, instruction_train, **kwargs):
            super().__init__(**kwargs)
            self.base_latent_kernel = base_latent_kernel # Kernel on the latent space (Matern Kernel)
            self.instruction_kernel = instruction_kernel # Kernel on the latent space (Matern Kernel)
            self.latent_train = latent_train # normalized training input
            self.lp_dim = self.latent_train.shape[-1]
            self.instruction_train = instruction_train # SMILES format training input #self.get_smiles(self.latent_train)#.clone())

        
        def forward(self, z1, z2, **params):
            # z1 and z2 are unnormalized
            check_dim = 0
            if len(z1.shape) > 2:
                check_dim = z1.shape[0]
                z1 = z1.squeeze(1)
            if len(z2.shape) > 2:
                check_dim = z2.shape[0]
                z2 = z2[0]
            latent_train_z1 = z1[:, :self.lp_dim] 
            latent_train_z2 = z2[:, :self.lp_dim]
            
            K_train_instruction = self.instruction_kernel.forward(self.instruction_train, self.instruction_train, **params)
            latent_space_kernel = self.base_latent_kernel.forward(self.latent_train, self.latent_train, **params)
            K_z1_training = self.base_latent_kernel.forward(latent_train_z1, self.latent_train, **params)
            K_z2_training = self.base_latent_kernel.forward(latent_train_z2, self.latent_train, **params)
            latent_space_kernel_inv = torch.inverse(latent_space_kernel + 0.0001 * torch.eye(len(self.latent_train)).to(latent_space_kernel.device))

            kernel_val = K_z1_training @ latent_space_kernel_inv @ (K_train_instruction) @ latent_space_kernel_inv @ K_z2_training.T
            if check_dim > 0:
                kernel_val = kernel_val.unsqueeze(1)
            return kernel_val


    



class ProbabilityImprovement(BasicBandit):
    """
    General Successive Elimination: https://arxiv.org/pdf/2106.04763.pdf
    """
    def __init__(self, num_arms: int = 0, T: int = 100, contexts: List[float] = []) -> None:
        super().__init__(num_arms)
        assert len(contexts) == num_arms, "Number of arms and contexts must be equal"
        # Save the data collected during the bandit process
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X = []
        self.y = []
        self.contexts = np.array(contexts)
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.logpi = np.zeros(self.num_arms)
        self.projector = SimpleMLP(self.contexts[0].shape[0], 256, 64).to(self.device)
        self.projector.eval()
        self.T = T
        self.t = 0
    
    def update(self, 
               arm: int = 0, 
               reward: float = 0.0) -> None:
        
        self.num_pulls[arm] += 1
        # n = self.num_pulls[arm]
        self.X.append(self.contexts[arm])
        self.y.append(reward)
        self.t += 1
        
        if self.contexts.shape[1] > 1000:
            eval_X = self.projector(torch.tensor(self.contexts).float().to(self.device)).unsqueeze(1).double()
            X = np.array(self.X)
            train_X = self.projector(torch.tensor(X).float().to(self.device)).double()
        else:
            eval_X = torch.tensor(self.contexts).float().to(self.device)
            train_X = torch.tensor(self.X).float().to(self.device)
            
        len_batch = len(self.X)
        
        if len_batch % 16 == 0 or self.t == self.T:
            train_y = torch.tensor(self.y).double().view(-1, 1).to(self.device)
            # train_y = torch.tensor(self.y).float().view.to(self.device)
            # Fit the GP model
            
            matern_kernel = MaternKernel(nu=2.5,
                                        ard_num_dims=train_X.shape[-1],
                                        lengthscale_prior=GammaPrior(3.0, 6.0))
            matern_kernel_y = MaternKernel(nu=2.5,
                                        ard_num_dims=train_y.shape[-1],
                                        lengthscale_prior=GammaPrior(3.0, 6.0))
            
            covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_y, latent_train=train_X.double(), instruction_train=train_y.double()))
            outcome_transform = Standardize(m=1)
            gp_model = SingleTaskGP(train_X, train_y,outcome_transform=outcome_transform, covar_module=covar_module)
            gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            # gp_mll = gp_mll.to(train_X)
            # optimizer = Adam(gp_model.parameters(), lr=1e-3)

            start_time = time.time()
            fit_gpytorch_model(gp_mll)
                
            
            print(f"Time to fit the model: {time.time() - start_time}")
            gp_model.eval()
            # Define the acquisition function
            LogPI = LogProbabilityOfImprovement(gp_model, train_y.max().item(), maximize=True)
            # Get the best arm
            self.logpi = LogPI(eval_X).detach().cpu().numpy()
            # breakpoint()
            self.sample_mean = gp_model(eval_X).mean.detach().cpu().numpy().flatten()
            
        
    
    def choose_action(self):
        
        # ind = random.randint(0, self.num_arms - 1)
        ind = np.argmax(-self.logpi)
        
        return ind
                
    def best_arm(self):
       
        best_ind = np.argmax(self.sample_mean)
       
        return best_ind
    
    def reset(self):
        self.X = []
        self.y = []
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean

