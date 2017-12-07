import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator


class State(object):
  perf_scale_factor = 0.001

  def __init__(self, n_iters, school_configs, district_configs):
    """ Initializes a new State object, for use in a Learner.
    Params:
      n_iters -- The number of iterations the simulation should be run for.
      school_configs -- A list containing one list per school, where each list
        contains one tuple per region, and each tuple is of the form
        (n_residents, income_avg, income_sd, school_perf_avg, school_perf_sd)
      district_configs -- A tuple of the form (budget_per_student,
        aggressiveness)
    """
    self.remaining_iters = n_iters
    self.district_state = np.array(
        np.array([[np.random.normal(i_avg,i_sd),
                   np.random.normal(sp_avg, sp_sd),
                   np.random.random_integers(1,18)] \
                       for n_res, i_avg, i_sd, sp_avg, sp_sd in school
                         for _ in range(n_res)]) \
                           for school in school_configs])
    self.school_budgets = np.zeros(len(school_configs))
    budget_per_student, aggressiveness = district_configs
    self.total_budget = budget_per_student
    self.allocate_budget(aggressiveness)

    self.initial_params = {'n_iters':n_iters, 'school_configs':school_configs,
                           'district_configs':district_configs}

  def allocate_budget(self, aggressiveness):
    school_sizes = np.vectorize(lambda s: s.shape[0])(self.district_state) 
    num_students = np.sum(school_sizes)
    school_perfs = np.vectorize(lambda s: np.mean(s,axis=0)[1])(self.district_state)
    print "aggressiveness", aggressiveness
    print "school sizes:", school_sizes
    print "avg school perfs:", school_perfs
    allocations = school_sizes*np.exp(-aggressiveness*school_perfs)
    print "softmax allocations: ", allocations
    allocations_sum = np.sum(allocations)
    self.school_budgets = self.budget_per_student*num_students*allocations/allocations_sum
    print "budget per student:", self.school_budgets/school_sizes

  def revert_old_students(student):
    if student[2] > 18:
      student[1] = 0
      student[2] = 0
    return student

  def iterate(self, aggressiveness):
    self.allocate_budget(aggressiveness)
    self.remaining_iters -= 1
    for i,school in enumerate(list(self.district_state)):
      school_size = school.shape[0]
      school_perf = np.mean(school, axis=0)[1]
      budget_per_student = self.school_budgets[i]/school_size
      school[(:,1)] += np.ones(school_size)
      expected_changes = self.perf_scale_factor * (school_perf + (budget_per_student-1) + school[(:,0)]) / np.log1p(school[(:,1)])
      school[(:,1)] += np.random.normal(expected_changes, .1)
      school = np.vectorize(revert_old_students)(school)

  def possible_actions(self):
    return list(np.linspace(-0.2,0.2,20))

  def is_finished(self):
    return self.remaining_iters == 0

  def reward(self):
    if self.is_finished():
      # TODO(wbradbur): Add a measure of inequality
      return sum(self.school_perf.values())/len(self.school_perf.values())
    else:
      return 0

  def get_reduced_state(self):
    return (self.remaining_iters, tuple(self.school_perf.values()))

  def __repr__(self):
    repr_str = "schools: " + str(self.school_states) \
                + ", school perfs: " + str(self.school_perf)

  def reset(self):
    self.__init__(**self.initial_params)

  def visualize(self, metric_id=0):
    """ Generate visualization in which:
          Rows = schools
          Boxes = regions (neighborhoods)
          Sub-boxes = students
            -> which are colored according to the specified metric (student performance, income, etc.)
    """
    # metric_ids:
    # 0 = income
    # 1 = performance

    test_region_states = {
      0: (0, [[1],[1],[1],[2],[3]]),
      1: (0, [[1],[1],[1],[1],[1]]),
      2: (0, [[1],[1],[1]]),
      3: (1, [[3],[3],[3],[4],[4]])
    }
    self.region_states = test_region_states

    # Collect school metrics
    metrics = {}
    max_students_per_region = 0
    for region_id in self.region_states:
      school_id = self.region_states[region_id][0]
      students = self.region_states[region_id][1]
      num_students = len(students)
      if num_students > max_students_per_region:
        max_students_per_region = num_students # update max students
      region_metrics = sorted([student[metric_id] for student in students], reverse=True)
      if school_id in metrics:
        metrics[school_id].append(region_metrics)
      else:
        metrics[school_id] = [region_metrics]
    max_regions_per_school = np.max([len(school_metrics) for _, school_metrics in metrics.iteritems()])

    # Convert metrics to grid layout
    n_super_rows = len(metrics) # num schools
    n_super_cols =  max_regions_per_school # max num regions assigned to a school
    box_dim = int(math.ceil(max_students_per_region**(0.5))) # square dimensions to contain max num students in a region
    print "n_super_rows: %d, n_super_cols: %d, box_dim: %d" % (n_super_rows, n_super_cols, box_dim)

    grid = np.zeros((n_super_rows * box_dim, n_super_cols * box_dim))
    for school_id, school_metrics in metrics.iteritems():
      for region_id in range(len(school_metrics)):
        cur_box = np.array(school_metrics[region_id])
        cur_box.resize(box_dim, box_dim)
        print "school_id: %d, region_id: %d" % (school_id, region_id)
        start_row = school_id * box_dim
        start_col = region_id * box_dim
        grid[start_row:(start_row+box_dim), start_col:(start_col+box_dim)] = cur_box
    print grid

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = plt.imshow(grid, cmap=plt.cm.plasma, interpolation='nearest', origin='upper')
    minor_locator = IndexLocator(3, 0)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which = 'minor', axis='y', color='w', linewidth=5)
    ax.grid(which = 'minor', axis='x', color='w', linewidth=2)
    plt.show()

if __name__ == '__main__':
  # TODO(michellelam): remove testing code
  state = State(1, [], [], [1,2])
  state.visualize()
