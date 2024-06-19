import numpy as np
import torch
import torch.nn.init as init

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
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
    



class GSE(BasicBandit):
    """
    General Successive Elimination: https://arxiv.org/pdf/2106.04763.pdf
    """
    def __init__(self, num_arms: int = 0, T: int = 100, contexts: List[float] = []) -> None:
        super().__init__(num_arms)
        assert len(contexts) == num_arms, "Number of arms and contexts must be equal"
        # self.model = BayesianRidge()
        # self.model = GradientBoostingRegressor(loss="huber", learning_rate=0.01, max_depth=20, random_state=42)
        self.models = [BayesianRidge(), GradientBoostingRegressor(loss="huber", learning_rate=0.01, max_depth=20, random_state=42),
                       MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=100, learning_rate="adaptive", random_state=42)]
        # self.model = MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=100, learning_rate="adaptive", random_state=42)
        self.X = []
        self.y = []
        self.contexts = np.array(contexts)
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.active_arms = np.arange(self.num_arms)
        self.non_active_arms = np.array([])
        self.sample_mean = np.zeros(self.num_arms)
        self.T = T
        self.L = np.ceil(np.log2(self.num_arms))
        self.interval = np.floor(self.T/(self.L * self.active_arms.shape[0]))
        if self.T <= self.active_arms.shape[0]:
            self.interval = 1
        self.batch_size = 32
        self.t = 0
    
    def update(self, 
               arm: int = 0, 
               reward: float = 0.0) -> None:
        
        self.num_pulls[arm] += 1
        n = self.num_pulls[arm]
        self.X.append(self.contexts[arm])
        self.y.append(reward)
        self.t += 1
        
        # After pulling 2 * active_arms, train the evaluation model
        if (((self.t + 1) % self.batch_size == 0) or (self.t == self.T)) and self.active_arms.shape[0] > 1:

            if self.X[0].shape[0] > 1000:
                # The dimension is too high, use a MLP projection to reduce the dimension
                projector = SimpleMLP(self.X[0].shape[0], 256, 64)
                projector.eval()
                with torch.no_grad():
                    X = projector(torch.tensor(self.X, dtype=torch.float32)).numpy()
            #     # The dimension of the context is too high, use PCA to reduce the dimension
            #     n_components = min(len(self.X), 2*self.num_arms)
            #     pca = PCA(n_components=100)
            #     self.X = pca.fit_transform(self.X)
            # Train the evaluation model
            print(f"Reached step {self.t}, training the evaluation model...")
            train_X, test_X, train_y, test_y = train_test_split(X, self.y, test_size=0.2, random_state=42)
            # Mixture of the models to train the evaluation model
            r2s = np.zeros(len(self.models))
            pred_ys = np.zeros((len(self.models), len(self.contexts[self.active_arms])))
            mses = np.zeros(len(self.models))
            for i, model in enumerate(self.models):
                model.fit(train_X, train_y)
                eval_y = model.predict(test_X)
                with torch.no_grad():
                    eval_X = projector(torch.tensor(self.contexts[self.active_arms], dtype=torch.float32)).numpy()
                pred_y = model.predict(eval_X)
                pred_ys[i] = pred_y
                mse = mean_squared_error(test_y, eval_y)
                mses[i] = mse
                r2 = r2_score(test_y, eval_y)
                r2s[i] = r2
            # Compute a weight of using mse and r2
            weights = np.exp(r2s) / np.sum(np.exp(r2s))
            
                
            
            print("Evaluation model trained!")
            print("Mean squared error: %.2f" % mses.mean())
            print('Variance score: %.2f' % r2s.mean())
            self.sample_mean = weights.dot(np.array(pred_ys))
            # breakpoint()
            if r2s.max() >= 0 or mse.min() <= 0.1:
            # Reject the arms with the lowest sample mean
                active_num = self.active_arms.shape[0] // 2
                active_inds = (-self.sample_mean).argsort()[:active_num]
                self.active_arms = self.active_arms[active_inds]
                # # Sort the active arms based on their sample mean  
                self.sample_mean = self.sample_mean[active_inds]
                self.active_arms = self.active_arms[np.argsort(self.sample_mean)]
                best_arm_idx = np.argmax(self.sample_mean)
                best_arm_sample_mean = self.sample_mean[best_arm_idx]
                print(f"The evaluated sample mean is {best_arm_sample_mean}, Current best arm {self.active_arms[best_arm_idx]}")
                
                self.interval = np.floor(self.T/(self.L * self.active_arms.shape[0]))
            self.t = 0
    
    def choose_action(self):
        
        inx = np.argmin(self.num_pulls[self.active_arms])
        
        return self.active_arms[inx]
                
    def best_arm(self):
       
        best_ind = np.argmax(self.sample_mean)
       
        return self.active_arms[best_ind]
    
    def reset(self):
        self.model = BayesianRidge()
        self.X = []
        self.y = []
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean

