from learn import Learner
from simulator import State
import numpy as np

if __name__ == '__main__':
  state = State(80, [[(100, 1, 2, 0.05, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(200, 0, 3, 0, 0.5), (200, -1, 1, -0.05, 0.5)],
                      [(300, 1, 2, 0.0, 0.5)]], (0.95, 1), debug=False)
  learner = Learner(state, alpha=0.05, gamma=0.9, exploration_param=1.0, decay_param=0.2, local_approx=True)

  # 10
  # learner.train(10)
  # print learner.evaluate(100)
  # # 100
  # learner.train(90)
  # print learner.evaluate(100)
  # # 250
  # learner.train(150)
  # print learner.evaluate(100)
  # # 500
  # learner.train(250)
  # print learner.evaluate(100)
  # # 750
  # learner.train(250)
  # print learner.evaluate(100)
  # # 1000
  # learner.train(250)
  # print learner.evaluate(10)

  # print learner.evaluate_random(100)
  # print learner.evaluate_aggressive(100)
  # print learner.evaluate_status_quo(1)
  print learner.evaluate_conservative(1)
