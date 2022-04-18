from datetime import timedelta, date

class Gpp:
  
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

class Cycle():
  # date format
  DT_STR_FORMAT = "%Y-%m-%d"
  def __init__(self, start_dt, end_dt, *args, **kwargs):
    self.start_dt = start_dt
    self.end_dt = end_dt
    self.workout_dates = []
    super(Cycle, self).__init__(*args, **kwargs) 

    # Iterator/Generator to return += date + days inclusive from start to end date of cycle
  def date_range(date1: date, date2: date):
    for n in range(int ((date2 - date1).days) + 1): # + 1 because range is exclusive
        yield date1 + timedelta(n)
  
  def add_workout_date(self, workout_date):
    self.workout_dates.append(workout_date) 

  # Generate array of workout dates simulating a 30-day trial
  def generate_workout_dates():
    for dt in date_range(self.start_dt, self.end_dt):
        # deucegym.com only posts workouts on weekdays so we need to exclude weekends
        # integer values corresponding to ISO weekday Sat & Sun.
        weekend = [6,7]
        if dt.isoweekday() not in weekend: 
            # workout_dates is formatted to correspond to the deucegym.com URL pattern
            add_workout_date(dt.strftime(DT_STR_FORMAT))
    #print(workout_dates) 

class Workout():
  STRENGTH, METCON = range(2)
  
  def __init__(self, workout_type, gpp=None, cycle=None):
      self.workout_type = workout_type
      super(Workout, self).__init__(*args, **kwargs)      

class MetCon(Workout):
  # AC => 30-60 liss @RPE1-3 Active Recovery
  # HII => Work :30 Rest 2:00 Repeat 5x Cardio-Respiratory
  # SE => Heavy :30 Rest 1:00 Repeat 8x
  # PI => Work :06 Rest 2:00 Repeat 10x
  # TI_PC => Work :15 Easy :45, PC Rest 2:00
  # MM => 3x Exercises UB/no rest, Rest 1:1, Repeat 5x
  AEROBIC_CAPACITY, HI_INTENSITY_INTERVAL, STRENGTH_ENDURANCE, POWER_INTERVAL, TEMPO_INTERVAL_POWER_CAPACITY, MIXED_MODAL = range(6)

  def __init__(self, metcon_type, work: int, rest: int, intervals: int, *args, **kwargs):
      self.metcon_type = metcon_type
      self.work = work
      self.rest = rest
      self.intervals = intervals
      super(MetCon, self).__init__(*args, **kwargs)