import numpy as np
import random
import math

class State(object):
  def __init__(self, n_iters, school_configs, region_configs, district_configs):
    """ Initializes a new State object, for use in a Learner.
    Params:
      n_iters -- The number of iterations the simulation should be run for.
      school_configs -- A list containing one tuple per school, where each
        tuple is of the form (school_id, capacity)
      region_configs -- A list containing one tuple per region, where each
        tuple is of the form (region_id, assigned_school_id, n_residents,
        income_avg, income_sd)
      district_configs -- A tuple of the form (total_budget, aggressiveness)
    """
    self.remaining_iters = n_iters
    self.school_states = {id:[cap, 0] for id,cap in school_configs}
    self.region_states = {id:(school, [[np.random.normal(i_avg,i_sd),0,0] \
                                        for _ in range(n_res)]) \
                                          for id, school, n_res, i_avg, i_sd \
                                            in region_configs}
    self.students_per_year = {id:n_res for id,_,n_res,_,_ in region_configs}
    self.school_perf = {id:0 for id in self.school_states}
    total_budget, aggressiveness = district_configs
    self.total_budget = total_budget
    self.allocate_budget(aggressiveness)

    self.initial_params = {'n_iters':n_iters, 'school_configs':school_configs,
                           'region_configs':region_configs,
                           'district_configs':district_configs}

  def allocate_budget(self, aggressiveness):
    school_sizes = {id:0 for id in self.school_states}
    for region in self.region_states:
      school_sizes[self.region_states[region][0]] += \
          len(self.region_states[region][1])
    allocations = {id:school_size[id] * \
                        math.exp(-aggressiveness*self.school_perf[id]) \
                          for id in self.school_states}
    allocations_sum = sum(allocations.values())
    for school in self.school_states:
      self.school_states[school][1] = float(allocations[school])/allocations_sum

  def student_perf_change(self, income, school_id, age, cache):
    if school_id in cache:
      school_avg_perf, budget_per_student = cache[school_id]
    else:
      sum_perf = 0
      num_students = 0
      for region in self.region_states:
        if self.region_states[region][0] == school_id:
          students = self.region_states[region][1]
          num_students += len(students)
          sum_perf += sum(map(lambda student: student[1],students))
      school_avg_perf = float(sum_perf)/num_students
      budget_per_student = float(self.school_states[school_id][1])/num_students
      cache[school_id] = school_avg_perf, budget_per_student
    expected_change = (school_avg_perf+(budget_per_student-1)+income) \
                        / math.log1p(age)
    return np.random.normal(expected_change, .5)

  def iterate(self, aggressiveness):
    self.allocate_budget(aggressiveness)
    self.remaining_iters -= 1
    cache = {}
    for region in self.region_states:
      for student in self.region_states[region][1]:
        student[2] += 1 # Increment age
        student[1] += self.student_perf_change(student[0],
                                               self.region_states[region][0],
                                               student[2], cache)
      """for _ in self.students_per_year[region]:
        copy = random.sample(self.region_states[region][1],1)
        copy[2] = 0 # Set age back to 0
        self.region_states[region][1].append(copy)"""
      self.region_states[region][1] = filter(lambda student: student[2]<=18,
                                        self.region_states[region][1])
    self.school_perf = {id:(cache[id][0] if id in cache else 0) \
                          for id in self.school_states}

  def possible_actions(self):
    return range(-2,2.2,0.2)

  def is_finished(self):
    return self.remaining_iters == 0

  def reward(self):
    if self.is_finished():
      # TODO(wbradbur): Add a measure of inequality
      return sum(self.school_perf.values())/len(self.school_perf.values())
    else:
      return 0

  def get_reduced_state(self):
    return (self.remaining_iters, self.school_perf)

  def __repr__(self):
    repr_str = "schools: " + str(self.school_states) \
                + ", school perfs: " + str(self.school_perf)

  def reset(self):
    self.__init__(**self.initial_params)
