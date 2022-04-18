

class Wod:
  
  GPP_TYPES = {
  0 : 'NOPERATOR',
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
  }

  def __init__(self, type):
    if type not in self.GPP_TYPES:
            raise ValueError("%s is not a valid type. Please enter a number 1, 2, 3, or 4" % type)
    self.type = type
    self.workout_dates = []

  def add_workout_date(self, workout_date):
        self.workout_dates.append(workout_date)
