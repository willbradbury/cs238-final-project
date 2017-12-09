import itertools
import numpy as np
from collections import defaultdict

class Learner(object):
  grid_mesh_count = 50

  def __init__(self, simulator, n_epochs, alpha, gamma, exploration_param,
                    decay_param):
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
    self.n_epochs = n_epochs
    self.alpha = alpha
    self.gamma = gamma
    self.exploration_param = exploration_param
    self.decay_param = decay_param

    state_size = len(simulator.get_reduced_state())
    self.shift_array = np.array([0] + [simulator.max_perf]*(state_size-1))
    self.scale_array = np.array([1] + [grid_mesh_count]*(state_size-1))
    self.rounding_matrix = np.array(list(itertools.product([-1,1], repeat=state_size)))

    # TODO(shivaal): make Q an np.matrix
    self.Q = defaultdict(float)
    self.N = defaultdict(int)

  def explore_action(self, state):
    actions = self.simulator.possible_actions()
    Q_s = np.array([self.Q[(state, action)] for action in actions])
    probs = np.exp(self.exploration_param * Q_s)
    probs /= sum(probs)
    return np.random.choice(actions, p=probs)

  def optimal_action(self, state):
    return max((self.Q[(state, action)], action)
        for action in self.simulator.possible_actions())[1]

  def update_step(self, s_t, a_t):
    s_prime = self.simulator.get_reduced_state()
    r_t = self.simulator.reward()
    a_prime = self.explore_action(s_prime)
    self.N[(s_t, a_t)] += 1
    delta = r_t + self.gamma * self.Q[(s_prime, a_prime)] - self.Q[(s_t, a_t)]
    for s, a in self.N.keys():
      self.Q[(s,a)] += self.alpha * delta * self.N[(s,a)]
      self.N[(s,a)] *= self.gamma * self.decay_param
    return s_prime, a_prime

  def beta(self, state, action):
    # floor and ceiling every dimension in the state to every
    # shift numpy vector
    translated_state = (state + self.shift_array) * self.scale_array
    print(translated_state.shape)
    # multiply state
    rounded_matrix = np.ceil(translated_state * self.rounding_matrix) * self.rounding_matrix
    print(rounded_matrix)

  def run(self):
    for _ in xrange(self.n_epochs):
      s_t = self.simulator.get_reduced_state()
      a_t = self.explore_action(s_t)
      while not self.simulator.is_finished():
        self.simulator.iterate(a_t)
        s_t, a_t = self.update_step(s_t, a_t)
      self.update_step(s_t, a_t)
