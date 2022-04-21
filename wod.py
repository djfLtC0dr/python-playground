from datetime import timedelta, date
import pandas as pd
from web_data import WebData
from pdf_data import PdfData
import json

class Gpp:  
  TYPES = {
  0 : 'NOPERATOR',
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
  }

  def __init__(self, type, allowed_types=TYPES):
    if type not in allowed_types:
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
  def date_range(self):
    for n in range(int ((self.end_dt - self.start_dt).days) + 1): # + 1 because range is exclusive
        yield self.start_dt + timedelta(n)
  
  def add_workout_date(self, workout_date: str):
    self.workout_dates.append(workout_date) 

  # Generate array of workout dates simulating a 30-day trial
  def generate_workout_dates(self):
    for dt in self.date_range():
        # deucegym.com only posts workouts on weekdays so we need to exclude weekends
        # integer values corresponding to ISO weekday Sat & Sun.
        weekend = [6,7]
        if dt.isoweekday() not in weekend: 
            # workout_dates is formatted to correspond to the deucegym.com URL pattern
            self.add_workout_date(dt.strftime(Cycle.DT_STR_FORMAT))
    #print(workout_dates) 

class Deuce(Cycle):
  def __init__(self, *args, **kwargs):
    super(Deuce, self).__init__(*args, **kwargs)

class Tsac(Cycle):
  MASH = "mash-evolution.pdf" # PDF
  # Pages of the mash-evolution TSAC macrocycle 
  PDF_SC_LV_PAGES = [*range(379, 420, 1)]
  
  def __init__(self, *args, **kwargs):
    super(Tsac, self).__init__(*args, **kwargs)
    self.pdf_data = PdfData(Tsac.MASH, Tsac.PDF_SC_LV_PAGES, self.workout_dates)  
    self.cylce_wods_json = self.pdf_data.load_pdf_to_json()
      

class Sslp(Cycle):
  # csv to store the SSLP Macro Cycle
  SSLP_CSV = "sslp.csv"

  def __init__(self, *args, **kwargs):
    super(Sslp, self).__init__(*args, **kwargs)
    self.cycle_wods_json = self.generate_workout_rx()

  def generate_workout_rx(self) -> str:
    return self.load_data_sslp_ph1()


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

  @staticmethod
  def load_csv(file_path: str) -> pd.DataFrame:
    df_empty = pd.DataFrame()
    try:
      df_sslp = pd.read_csv(file_path)
      return df_sslp
    except FileNotFoundError:
      return df_empty

  def load_data_sslp_ph1(self) -> str:
    json_formatted_str = ''
    df_sslp = self.load_csv(Sslp.SSLP_CSV)
    if len(df_sslp.index) != 0:
      df_sslp = df_sslp.assign(Phase_RX_Loads=self.calc_sslp_ph1())
      wods_json_str = df_sslp.to_json(orient='records')
      obj_data = json.loads(wods_json_str)
      json_formatted_str += json.dumps(obj_data, indent=4) 
      return json_formatted_str
    else:
        wd = WebData(WebData.SSLP, Gpp.TYPES[4]) #SSLP
        wd.webscrape(self.workout_dates)
        json_formatted_str = self.load_data_sslp_ph1()
        return json_formatted_str

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