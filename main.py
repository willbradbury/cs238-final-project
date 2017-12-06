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

if __name__ == '__main__':
  # initialize simulator (add params)
  state = State(50, [(0, 2000), (1, 2000), (2, 1800)],
                    [(0, 0, 500, 1, 2), (1, 0, 500, -1, 1), (2, 1, 500, 0, 3),
                     (3, 1, 1000, -1, 1), (4, 2, 1000, 1, 1)],
                    (3000, 1))
  # initialize learner (add params)
  learner = Learner(state)
  learner.run()
