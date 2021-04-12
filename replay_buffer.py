import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def combined_shape(size, shape=None):
    if shape is None:
        return (size,)
    return (size, shape) if np.isscalar(shape) else (size, *shape)


class UniformReplayBuffer:
    """
    Replay buffer that selects data randomly (with equal probability)
    """

    def __init__(self, size, obs_dim):
        """
        :param size: buffer size
        :param obs_dim: observation space dimensions
        """
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)  # observations
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)  # next state observations
        self.act_buf = np.zeros(size, dtype=np.int64)  # actions
        self.rew_buf = np.zeros(size, dtype=np.float32)  # rewards
        self.done_buf = np.zeros(size, dtype=np.bool)  # an episode ended or not

        # the maximum size for the buffer, current size and pointer
        self.max_size, self.size, self.ptr = size, 0, 0

    def __len__(self):
        return self.size

    def store(self, obs, act, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)  # randomly select indices
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
