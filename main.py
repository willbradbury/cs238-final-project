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
  state = State(80, [(0, 2000), (1, 2000), (2, 1800)],
                    [(0, 0, 100, 1, 2), (1, 0, 200, -1, 1), (2, 1, 200, 0, 3),
                     (3, 1, 200, -1, 1), (4, 2, 300, 1, 1)],
                    (.8, 1))
  # initialize learner (add params)
  learner = Learner(state, 10, 0.05, 0.9, 1.0, 0.2)
  learner.run()
