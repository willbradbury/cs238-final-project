'''
simulator takes in state, action
rl algorihtm lives elsewhere
main manages the two

simulator is a class:
  print out representation of state
  get small state
  get next iteration given action
  extract reward given state,action
  get list of possible actions
  reset simulator
  check if horizon reached

  superintendent constraints:
    penalized for aggressiveness
    penalized if school perf drops too low

Learner:
  takes in a simulator
  pass in learning params:
    alpha if learning rate
    gamma for discount
    exploration params
    num_iterations
  choose action given state
  fn approximation?
'''
from learn import Learner
from simulator import State
import numpy as np

if __name__ == '__main__':
  # initialize simulator (add params)
  state = State(80, [[(100, 1, 2, 0.05, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(200, 0, 3, 0, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(300, 1, 1, 0.05, 0.5)]], (0.8, 1), debug=False)
  # initialize learner (add params)
  learner = Learner(state, alpha=0.05, gamma=0.9, exploration_param=0.5, decay_param=0.2, local_approx=True)

  # learner.beta(np.array([4, -1.11, 0.011, 0.457]))
  learner.train(100)
  print learner.evaluate(10)
