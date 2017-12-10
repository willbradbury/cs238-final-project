import itertools
import numpy as np
from collections import defaultdict

class Learner(object):
  grid_mesh_count = 10
  epsilon = 1e-4

  def __init__(self, simulator, alpha, gamma, exploration_param, decay_param,
                  local_approx=False):
    """ Learns a policy using Sarsa-Lambda given a State instance.
    Params:
      simulator -- A Simulator instance.
      n_epochs -- The number of epochs the entire simulation should be run for.
      alpha -- A float that specifies the learning rate.
      gamma -- A float between 0 and 1 that specifies the discount factor.
      exploration_param -- A float specifying the amount of exploration, depends
        on the exploration strategy.
      decay_param -- A float between 0 and 1 that specifies the exponential
        decay of counts in Sarsa-Lambda
    """
    self.simulator = simulator
    self.alpha = alpha
    self.gamma = gamma
    self.exploration_param = exploration_param
    self.decay_param = decay_param
    self.local_approx = local_approx

    if local_approx:
      state_size = len(simulator.get_reduced_state(as_np_array=local_approx))
      self.theta = np.zeros((self.grid_mesh_count,)*state_size + (simulator.num_actions,))
      self.N = np.zeros((self.grid_mesh_count,)*state_size + (simulator.num_actions,))
      self.shift_array = np.array([0] + [simulator.max_perf]*(state_size-1))
      self.scale_array = np.array([float(self.grid_mesh_count-1)/simulator.remaining_iters] + [float(self.grid_mesh_count-1)/(2*simulator.max_perf)]*(state_size-1))
      self.rounding_matrix = np.array(list(itertools.product([-1,1], repeat=state_size)))
    else:
      self.theta = defaultdict(float)
      self.N = defaultdict(float)

  def theta_index(self, state, action, idx):
    if self.local_approx:
      return np.hsplit(np.insert(idx, idx.shape[1], float(action+self.simulator.max_aggressiveness)*(self.simulator.num_actions-1)/(2*self.simulator.max_aggressiveness), axis=1), idx.shape[1]+1)
    else:
      return (state, action)

  def theta_utility(self, state, actions, idx, weights):
    if self.local_approx:
      weights = weights.flatten()
      # return [np.dot(self.theta[np.hsplit(np.insert(idx, idx.shape[1], action_idx, axis=1), idx.shape[1]+1)].flatten(), weights)
                # for action_idx in np.round((actions+self.simulator.max_aggressiveness)*(self.simulator.num_actions-1)/(2*self.simulator.max_aggressiveness))]
      return [np.dot(self.theta[self.theta_index(state, action, idx)].flatten(), weights) for action in actions]
    else:
      return [self.theta[(state, action)] for action in actions]

  def beta(self, state):
    if self.local_approx:
      # print state
      translated_state = (state + self.shift_array) * self.scale_array
      idx = (np.ceil(translated_state * self.rounding_matrix) * self.rounding_matrix).astype(np.int32)
      weights = 1 / (np.sum((translated_state - idx)**2, axis=1) + self.epsilon)
      weights /= np.sum(weights)
      # print translated_state
      # print idx
      # print weights

      return (idx, weights.reshape(-1,1))
    else:
      return (None, 1)

  def explore_action(self, state, idx, weights):
    actions = self.simulator.possible_actions()
    Q_s = np.array(self.theta_utility(state, actions, idx, weights))
    # print(Q_s)
    probs = np.exp(self.exploration_param * Q_s)
    probs /= np.sum(probs)
    return np.random.choice(actions, p=probs)

  def optimal_action(self, state, idx=None, weights=None):
    actions = self.simulator.possible_actions()
    return max(zip(self.theta_utility(state, actions, idx, weights), actions))[1]

  def update_step(self, s_t, a_t, idx_t, weights_t):
    s_prime = self.simulator.get_reduced_state(as_np_array=self.local_approx)
    idx_prime, weights_prime = self.beta(s_prime)
    r_t = self.simulator.reward()
    a_prime = self.explore_action(s_prime, idx_prime, weights_prime)
    self.N[self.theta_index(s_t, a_t, idx_t)] += weights_t
    delta = r_t + self.gamma * np.dot(self.theta[self.theta_index(s_prime, a_prime, idx_prime)].flatten(), weights_prime.flatten()) - np.dot(self.theta[self.theta_index(s_t, a_t, idx_t)].flatten(), weights_t.flatten())
    if self.local_approx:
      self.theta += self.alpha * delta * self.N
      self.N *= self.gamma * self.decay_param
    else:
      for s, a in self.N.keys():
        idx, weights = self.beta(s)
        self.theta[self.theta_index(s,a,idx)] += self.alpha * delta * self.N[self.theta_index(s,a,idx)]
        self.N[self.theta_index(s,a,idx)] *= self.gamma * self.decay_param
    return s_prime, a_prime, idx_prime, weights_prime

  def run_step(self, s_t, a_t):
    s_prime = self.simulator.get_reduced_state(as_np_array=self.local_approx)
    idx_prime, weights_prime = self.beta(s_prime)
    r_t = self.simulator.reward()
    a_prime = self.optimal_action(s_prime, idx_prime, weights_prime)
    return r_t, s_prime, a_prime

  def train(self, n_epochs):
    print self.theta.size
    print "training:"
    for epoch in xrange(n_epochs):
      print 'epoch', epoch
      s_t = self.simulator.get_reduced_state(as_np_array=self.local_approx)
      idx_t, weights_t = self.beta(s_t)
      a_t = self.explore_action(s_t, idx_t, weights_t)
      while not self.simulator.is_finished():
        self.simulator.iterate(a_t)
        s_t, a_t, idx_t, weights_t = self.update_step(s_t, a_t, idx_t, weights_t)
      self.update_step(s_t, a_t, idx_t, weights_t)
      self.simulator.reset()
      self.N = np.zeros(self.N.shape)
    if self.local_approx:
      print np.count_nonzero(self.theta)
    else:
      print len(self.theta)
    # self.simulator.visualize() # Final outcome visualization

  def evaluate(self, n_epochs):
    print "evaluating:"
    total_reward = 0.0
    for epoch in xrange(n_epochs):
      print epoch
      s_t = self.simulator.get_reduced_state(as_np_array=self.local_approx)
      idx_t, weights_t = self.beta(s_t)
      a_t = self.optimal_action(s_t, idx_t, weights_t)
      while not self.simulator.is_finished():
        self.simulator.iterate(a_t)
        r_t, s_t, a_t = self.run_step(s_t, a_t)
        total_reward += r_t
      r_t, _, _ = self.run_step(s_t, a_t)
      total_reward += r_t
      self.simulator.reset()
    return total_reward / n_epochs
