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
from simulator import Simulator

if __name__ == '__main__':
  # initialize simulator (add params)
  simulator = Simulator()
  # initialize learner (add params)
  learner = Learner(simulator)
  learner.run()
