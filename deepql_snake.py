
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

UP = (-1, 0)
DOWN = (1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)
actions = {
    0: UP,
    1: DOWN, 
    2: RIGHT,
    3: LEFT
}
State = namedtuple('State', 
                  ['grid', 'positions', 'direction'])

Transition = namedtuple('Transition', 
                       ['state', 'action', 'next_state', 'reward', 'done'])

MEMORY_SIZE = 10_000
EPS = 10e-2

class Env():
    def __init__(self, size):
        assert size >= 5, 'Size bellow 5 unsuported'
        self.size = size
    
    def _set_fruit(self):
        ix1, ix2 = np.where(self.grid ==0)
        i = np.random.randint(ix1.size)
        self.grid[ix1[i], ix2[i]] = 2
    
    def reset(self):
        self.grid = torch.zeros((self.size, self.size))
        self.positions = np.vstack((np.zeros(3, dtype=np.int64), np.arange(3)))
        self.grid[self.positions] = 1
        self.grid[self.positions[0][-1], self.positions[1][-1]] = -1
        self.direction = RIGHT
        self.fruit = self._set_fruit()
        return State(self.grid.clone(), self.positions, self.direction)
    
    def _verify_action_compatibility(self, action):
        x, y = action
        x_old, y_old = self.direction
        if (x * y_old + x_old * y == 0) and (x * x_old + y * y_old < 0):
            return False
        return True
    
    def step(self, action):
        try:
            action = actions[action]
        except:
            pass
        done = False
        found_fruit = False
        current_direction = self.direction
        reward = 0
        if not self._verify_action_compatibility(action):
            reward -= 1
            action = current_direction
        dx, dy = action
        pos_x, pos_y = self.positions
        head_x, head_y = pos_x[-1], pos_y[-1]
        new_head_x, new_head_y = ((head_x + dx) % self.size,
                          (head_y + dy) % self.size)
        if self.grid[new_head_x, new_head_y] == 2:
            reward += 5
            found_fruit = True
            new_pos_x, new_pos_y = pos_x, pos_y
            delete_tail = False
        else:
            new_pos_x, new_pos_y = pos_x[1:], pos_y[1:]
            delete_tail = True
        tail_x, tail_y = pos_x[0], pos_y[0]    
        if ((new_head_x == new_pos_x) &
            (new_head_y == new_pos_y)).any():
            reward -= 10
            done = True
            return State(self.grid.clone(), self.positions, action), reward, done
        pos_x = np.hstack((new_pos_x, [new_head_x]))
        pos_y = np.hstack((new_pos_y, [new_head_y]))
        if delete_tail:
            self.grid[tail_x, tail_y] = 0
        self.grid[head_x, head_y] = 1
        self.grid[new_head_x, new_head_y] = -1
        if found_fruit:
            self._set_fruit()
        self.direction = action
        self.positions = (pos_x, pos_y)
        return State(self.grid.clone(), np.array((pos_x, pos_y)), 
                     action), reward, done        
        
 
class MemoryBuffer:
    def __init__(self, size):
        self._size = size
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = [] 
        self._head = None
    def _append_place_holder(self):    
        self._states.append(None)
        self._actions.append(None)
        self._next_states.append(None)
        self._rewards.append(None)
        self._dones.append(None)
        if self._head is None:
            self._head = 0
        else:
            self._head += 1
    def add(self, transition):
        if len(self) < self._size:
            self._append_place_holder()
        else:
            self._head = (self._head + 1) % self._size
        self._append_transition(transition)

    def _append_transition(self, transition):
        head = self._head 
        self._states[head] = transition.state
        self._actions[head] = transition.action
        self._next_states[head] = transition.next_state
        self._rewards[head] = transition.reward
        self._dones[head] = transition.done

    def sample(self, size):
        idx = np.random.randint(len(self), size=size)
        
        return (np.array(self._states)[idx], np.array(self._actions)[idx],
               np.array(self._next_states)[idx], np.array(self._rewards)[idx],
               np.array(self._dones)[idx])
    
    def batch_sample(self, size):
        states, actions, next_states, rewards, dones = self.sample(size)
        start_grid = torch.stack(states[:, 0].tolist())
        end_grids = torch.stack(next_states[:, 0].tolist())
        end_directions = torch.from_numpy(np.stack(next_states[:,-1]))
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones + 0)
        actions = torch.from_numpy(actions)
        directions = torch.from_numpy(np.stack(states[:,-1]))
        return ((start_grid, directions), (end_grids, end_directions), 
                actions, rewards, dones)
    
    def __len__(self):
        return len(self._states)
      

class AgentModel(nn.Module):
    def __init__(self, sizes, output_size):
        super().__init__()
        models = []
        for i in range(len(sizes)-1):
            models.append(nn.Linear(sizes[i], sizes[i+1]))
            models.append(nn.LeakyReLU())
        models.append(nn.Linear(sizes[-1], output_size))
        
        self.grid_model = nn.Sequential(*models)
    
    def forward(self, grid, directions):
        grid = grid.flatten(start_dim=1)
        grid = torch.cat((grid, directions), dim=1)
        return self.grid_model(grid)
        
        
class Agent:
    def __init__(self, state_size, action_size):
        sizes = [state_size, 50, 25, 10]
        self.action_size = action_size
        self.local_model = AgentModel(sizes, action_size).double()
        self.target_model = AgentModel(sizes, action_size).double()
        self.memory = MemoryBuffer(size=MEMORY_SIZE)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.01)
        self.tstep = 0
        
    def step(self, gamma, state, action, reward, next_state, done):
        transition = Transition(state=state, action=action, 
                               next_state=next_state, reward=reward, done=done)
        self.memory.add(transition)
        if len(self.memory) >= 100:
            (state, next_state, actions, 
               rewards, dones) = self.memory.batch_sample(64)
            self.learn(gamma, state, next_state, actions, rewards, dones)
            
    def learn(self, gamma, states, next_states, actions, rewards, dones):
        creterion = torch.nn.MSELoss()
        self.local_model.train()
        self.target_model.eval()
        predicted_targets = self.local_model(*(x.double() for x in states)).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            labels_next = self.target_model(*(x.double() for x in next_states)).detach().max(1)[0].unsqueeze(1)
        labels = rewards + (gamma* labels_next.squeeze()) 
        loss = creterion(predicted_targets, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.tstep += 1
        if self.tstep % 100 == 0:
            self.target_model.load_state_dict(self.local_model.state_dict())
        
    
    def act(self, state):
        if np.random.random() <= EPS:
            return np.random.randint(self.action_size)
        if isinstance(state, State):
            state = (state.grid.unsqueeze(0).double(),
                     torch.as_tensor(state.direction).unsqueeze(0).double())
        with torch.no_grad():
            return self.local_model(*state).squeeze().argmax().item()
          
# train model 

env = Env(10)
agent = Agent(102, 4) # 102 = 10 * 10 + 2 (grid size + direction dimensionality)

def episode(max_t = 10000):
    rewards = []
    state = env.reset()
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.step(0.9, state, action, reward, next_state, done)
        state = next_state
        rewards.append(reward)
        if done:
            break
    return np.sum(rewards)
  

rews = []
for i in range(20):
    print('peisode {}'.format(i))
    mean_reward = episode(1000)
    print(mean_reward)
    rews.append(mean_reward)
         
