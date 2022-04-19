from datetime import timedelta, date
import pandas as pd

class Gpp:  
  TYPES = {
  0 : 'NOPERATOR',
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
  }

  def __init__(self, type):
    if type not in self.TYPES:
            raise ValueError("%s is not a valid type. Please enter a number 1, 2, 3, or 4" % type)
    self.type = type

class Cycle():
  # date format
  DT_STR_FORMAT = "%Y-%m-%d"
  def __init__(self, start_dt: date, end_dt: date):
    self.start_dt = start_dt
    self.end_dt = end_dt
    self.workout_dates = []
    self.generate_workout_dates()

    # Iterator/Generator to return += date + days inclusive from start to end date of cycle
  def date_range(date1: date, date2: date):
    for n in range(int ((date2 - date1).days) + 1): # + 1 because range is exclusive
        yield date1 + timedelta(n)
  
  def add_workout_date(self, workout_date: str):
    self.workout_dates.append(workout_date) 

  # Generate array of workout dates simulating a 30-day trial
  def generate_workout_dates(self):
    for dt in self.date_range(self.start_dt, self.end_dt):
        # deucegym.com only posts workouts on weekdays so we need to exclude weekends
        # integer values corresponding to ISO weekday Sat & Sun.
        weekend = [6,7]
        if dt.isoweekday() not in weekend: 
            # workout_dates is formatted to correspond to the deucegym.com URL pattern
            self.add_workout_date(dt.strftime(Cycle.DT_STR_FORMAT))
    #print(workout_dates) 

class Tsac(Cycle):
  def __init__(self, *args, **kwargs):
      super(Sslp, self).__init__(*args, **kwargs)


class Sslp(Cycle):
  # csv to store the SSLP Macro Cycle
  SSLP_CSV = "sslp.csv"

  def __init__(self, *args, **kwargs):
      super(Sslp, self).__init__(*args, **kwargs)

  # % reference => https://www.t-nation.com/training/know-your-ratios-destroy-weaknesses/
  # Bench Press: 75% of back squat
  # Powerlifting Deadlift: 120% of back squat 
  # Military Press (strict): 45% of back squat
  # Power Clean: 68% of back squat
  def calc_sslp_ph1(self):
    one_rm_bs = int(input("What is your 1RM Back Squat?\n"))
    one_rm_bp = (one_rm_bs * .75)
    one_rm_sp = (one_rm_bs * .45)
    one_rm_dl = (one_rm_bs * 1.20)
    bs_pct_inc = 2.5
    bp_pct_inc = 2.0
    sp_pct_inc = 1.5
    dl_pct_inc = 3.5
    pct_1rm = .80
    ph1_rx_bs_loading = []
    ph1_rx_sp_bp_loading = []
    ph1_rx_dl_loading = []
    for i in range(1, len(self.workout_dates), 5):
      # starting at 80% to allow reasonable linear progression
      bs = str((one_rm_bs * pct_1rm) + (i * bs_pct_inc))
      bp = str((one_rm_bp * pct_1rm) + (i * bp_pct_inc))
      sp = str((one_rm_sp * pct_1rm) + (i * sp_pct_inc))
      dl = str((one_rm_dl * pct_1rm) + (i * dl_pct_inc))
      ph1_rx_bs_loading.append(bs)
      ph1_rx_sp_bp_loading.append(sp + '/' + bp)
      ph1_rx_dl_loading.append(dl)
    return [ph1_rx_bs_loading, ph1_rx_sp_bp_loading, ph1_rx_dl_loading, 'Phase 2-TBD', 'Phase 2-TBD', 'Phase 2-TBD', 'Phase 3-TBD', 'Phase 3-TBD', 'Phase 3-TBD']

  def load_csv(self, file_path: str) -> pd.DataFrame:
    emptyDF = pd.DataFrame()
    try:
      dfSSLP = pd.read_csv(file_path)
      return dfSSLP
    except FileNotFoundError:
      return emptyDF

  def load_data_sslp_ph1(self) -> pd.DataFrame:
    dfSSLP = self.load_csv(Sslp.SSLP_CSV)
    if len(dfSSLP.index) != 0:
      dfSSLP = dfSSLP.assign(Phase_1_RX_Loads=self.calc_sslp_ph1())
      return dfSSLP
    else:
        webscrape(gpp.SSLP)
        self.load_data_sslp_ph1()

class Workout():
  STRENGTH, METCON = range(2)
  
  def __init__(self, workout_type, gpp=None, cycle=None):
      self.workout_type = workout_type
      self.gpp = gpp
      self.cycle = cycle    

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