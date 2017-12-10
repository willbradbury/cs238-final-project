import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator


class State(object):
  budget_scale_factor = 20
  perf_scale_factor = 0.1
  max_perf = 5.0
  num_actions = 10
  max_aggressiveness = 10.0

  def __init__(self, n_iters, school_configs, district_configs, debug=False):
    """ Initializes a new State object, for use in a Learner.
    Params:
      n_iters -- The number of iterations the simulation should be run for.
      school_configs -- A list containing one list per school, where each list
        contains one tuple per region, and each tuple is of the form
        (n_residents, income_avg, income_sd, school_perf_avg, school_perf_sd)
      district_configs -- A tuple of the form (budget_per_student,
        aggressiveness)
    """
    self.debug = debug
    self.remaining_iters = n_iters
    self.district_state = np.array([
        np.array([[np.random.normal(i_avg,i_sd),
                   np.random.normal(sp_avg, sp_sd),
                   np.random.random_integers(1,18)] \
                       for n_res, i_avg, i_sd, sp_avg, sp_sd in school
                         for _ in range(n_res)]) \
                           for school in school_configs])
    self.school_budgets = np.zeros(len(school_configs))
    budget_per_student, aggressiveness = district_configs
    self.budget_per_student = budget_per_student
    self.school_perfs = np.vectorize(lambda s: np.mean(s,axis=0)[1])(self.district_state)
    self.allocate_budget(aggressiveness)

    self.initial_params = {'n_iters':n_iters, 'school_configs':school_configs,
                           'district_configs':district_configs, 'debug':debug}

  def allocate_budget(self, aggressiveness):
    school_sizes = np.vectorize(lambda s: s.shape[0])(self.district_state)
    num_students = np.sum(school_sizes)
    allocations = school_sizes*np.exp(-aggressiveness*self.school_perfs)
    self.school_budgets = self.budget_per_student*num_students*allocations/np.sum(allocations)
    if self.debug:
      print "aggressiveness", aggressiveness
      print "school sizes:", school_sizes
      print "avg school perfs:", self.school_perfs
      print "softmax allocations:", allocations
      print "budget per student:", self.school_budgets/school_sizes

  def iterate(self, aggressiveness):
    self.allocate_budget(aggressiveness)
    self.remaining_iters -= 1
    for i,school in enumerate(list(self.district_state)):
      school_size = school.shape[0]
      school_perf = np.mean(school, axis=0)[1]
      budget_per_student = self.school_budgets[i]/school_size
      school[:,2] += np.ones(school_size)
      expected_changes = self.perf_scale_factor * \
          (school_perf + self.budget_scale_factor*(budget_per_student-1) + school[:,0]) / np.log1p(school[:,2])
      school[:,1] += np.random.normal(expected_changes, .1)
      idx = school[:,2]>18
      school[idx, 1] = 0
      school[idx, 2] = 1
    self.school_perfs = np.vectorize(lambda s: np.mean(s,axis=0)[1])(self.district_state)

  def possible_actions(self):
    return np.linspace(-self.max_aggressiveness, self.max_aggressiveness, self.num_actions)

  def is_finished(self):
    return self.remaining_iters == 0

  def reward(self):
    return np.mean(self.school_perfs) - np.std(self.school_perfs)
    # if self.remaining_iters == 0:
    #   return np.mean(self.school_perfs) - np.std(self.school_perfs)
    #   # return np.mean(self.school_perfs)
    # else:
    #   return 0

  def get_reduced_state(self, as_np_array=False):
    clamped_school_perfs = np.minimum(self.school_perfs, self.max_perf)
    clamped_school_perfs = np.maximum(clamped_school_perfs, -self.max_perf)
    reduced_state = np.hstack((self.remaining_iters, clamped_school_perfs))
    if as_np_array:
      return reduced_state
    else:
      return tuple(reduced_state)

  def __repr__(self):
    repr_str = "schools: " + str(self.school_budgets) \
                + ", school perfs: " + str(self.school_perfs)

  def reset(self):
    self.__init__(**self.initial_params)

  def visualize(self, metric_id=0, debug=False):
    """ Generate visualization in which:
          Rows = schools
          Boxes = regions (neighborhoods)
          Sub-boxes = students
            -> which are colored according to the specified metric (student performance, income, etc.)


          school configs has list of schools
          ea school has list of tuples of stats for each region

          district_state: [schools: [regions: [residents: []]]]
    """

    # test_district_state = [
    #   [[1],[1],[1],[2],[3], [1],[1],[1],[1],[1], [1],[1],[1]],
    #   [[3],[3],[3],[4],[4]]
    # ]
    # district_state = np.array(test_district_state)

    # school_configs = [[(5,0), (5,0), (3,0)], [(5,0)]]
    # self.initial_params = {'n_iters':2, 'school_configs':school_configs,
    #                        'district_configs':[]}

    # Collect school metrics
    metrics = []
    max_students_per_region = 0
    school_i = 0
    for school in self.initial_params['school_configs']:
      #(n_residents, income_avg, income_sd, school_perf_avg, school_perf_sd)
      school_metrics = []
      student_i = 0
      for region in school:
        # n_residents = len(region)
        n_residents = region[0]
        if debug:
          print "n_residents", n_residents
        if n_residents > max_students_per_region:
          max_students_per_region = n_residents # update max students
        region_metrics = sorted([student[metric_id]
            for student in self.district_state[school_i][student_i:student_i+n_residents]], reverse=True)
        school_metrics.append(region_metrics)
        if debug:
          print "region_metrics", region_metrics
        student_i += n_residents
      metrics.append(school_metrics)
      if debug:
        print "school_metrics", school_metrics
      school_i += 1
    max_regions_per_school = np.max([len(school_metrics) for school_metrics in metrics])

    # Convert metrics to grid layout
    n_super_rows = len(metrics) # num schools
    n_super_cols =  max_regions_per_school # max num regions assigned to a school
    box_dim = int(math.ceil(max_students_per_region**(0.5))) # square dimensions to contain max num students in a region
    if debug:
      print "n_super_rows: %d, n_super_cols: %d, box_dim: %d" % (n_super_rows, n_super_cols, box_dim)

    # grid = np.zeros((n_super_rows * box_dim, n_super_cols * box_dim))
    grid = np.empty((n_super_rows * box_dim, n_super_cols * box_dim))
    grid[:] = np.nan
    for school_id in range(n_super_rows):
      school_metrics = metrics[school_id]
      for region_id in range(len(school_metrics)):
        cur_box = np.array(school_metrics[region_id])
        cur_box.resize(box_dim, box_dim)
        if debug:
          print "school_id: %d, region_id: %d" % (school_id, region_id)
        start_row = school_id * box_dim
        start_col = region_id * box_dim
        grid[start_row:(start_row+box_dim), start_col:(start_col+box_dim)] = cur_box
    # grid[grid == 0.0] = np.nan # Set empty entries to NaN (to leave un-colored)
    if debug:
      print "grid:", grid

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = plt.imshow(grid, cmap=plt.cm.coolwarm, interpolation='nearest', origin='upper', vmin=-6, vmax=6)
    minor_locator = IndexLocator(box_dim, 0)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    plt.colorbar()
    ax.grid(which = 'minor', axis='y', color='w', linewidth=5)
    ax.grid(which = 'minor', axis='x', color='w', linewidth=2)
    plt.show()

# if __name__ == '__main__':
#   # TODO(michellelam): remove testing code
#   state = State(1, [], [])
#   state.visualize()
