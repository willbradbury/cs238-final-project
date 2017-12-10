from learn import Learner
from simulator import State
import numpy as np

if __name__ == '__main__':
  state = State(80, [[(100, 1, 2, 0.05, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(200, 0, 3, 0, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(300, 1, 2, 0.0, 0.5)]], (0.95, 1), debug=False)
  learner = Learner(state, alpha=0.05, gamma=0.9, exploration_param=1.0, decay_param=0.2, local_approx=True)

  learner.train(1000)
  print learner.evaluate(100)
