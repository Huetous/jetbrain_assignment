import torch, random
import torch.optim
import torch.nn.functional as F
from models import MLP, DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, obs_dim, act_dim, scheduler, lr=1e-3, gamma=0.99,
                 double=False, dueling=False, n_filters=[64, 128]):
        """
        :param obs_dim: observation space dimensions
        :param act_dim: action space dimensions
        :param scheduler: epsilon scheduler for epsilon-greedy policy
        :param lr: learning rate for main action-value function
        :param gamma: discounting rate
        :param double: specifies the way the main value function is updated
        :param dueling: specifies usage of dueling architecture for value functions
        :param n_filters: list of filter for the value functions architectures
        """
        self.double = double
        self.gamma = gamma
        self.act_dim = act_dim

        self.eps_history, self.loss_history = [], []
        self.scheduler = scheduler

        Net = DuelingMLP if dueling else MLP  # specifies NN architecture
        self.Q = Net(obs_dim, act_dim, n_filters).to(device)  # main value function
        self.Q_target = Net(obs_dim, act_dim, n_filters).to(device)  # target value function

        # Copy and freeze the target net parameters
        self.update_target()
        for p in self.Q_target.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def act(self, obs, t):
        """
        Returns an action that is selected according to the main value function
        :param obs: observation
        :param t: timestep
        :return: selected action
        """
        eps = self.scheduler(t)  # get new scheduled epsilon value
        self.eps_history.append(eps)
        
        if random.random() < eps:
            # choose a random action
            return random.randrange(self.act_dim), None
        
        with torch.no_grad():
            # Returns one action
            max_action_data = self.Q(obs).max(1)
            # action and its action value
            return max_action_data[1].item(), max_action_data[0].item()

    def update_main(self, batch):
        """
        Updates the main value function parameters
        """
        o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"]

        # Get action values for each (state, action) pair in the batch
        q = self.Q(o).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # Get action values for each action in each next state in the batch
            out = self.Q_target(o2)

            if self.double:  # Double DQN
                # Get actions with maximum action values in the next state, according to the main value function
                actions = self.Q(o2).argmax(1).unsqueeze(1)
                # Get action value for these actions, according to the target value function
                q_target = out.gather(1, actions).squeeze()
            else:
                # Get actions with maximum action values, according to the main value function
                # and their action values according to the MAIN value function
                q_target = out.max(1)[0]

            backup = r + self.gamma * (~d) * q_target

        # Use Huber loss instead of MSE, since the former is less sensitive to outliers (e.g., big errors)
        loss = F.smooth_l1_loss(q, backup)

        self.opt.zero_grad()
        loss.backward()

        # Gradient clipping improves performance (paper - https://arxiv.org/pdf/1511.06581.pdf)
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1., 1.)

        self.opt.step()
        self.loss_history.append(loss.item())

    def update_target(self):
        """
        Updater the target value function parameters
        """
        self.Q_target.load_state_dict(self.Q.state_dict())  # Copy parameters
