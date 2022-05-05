from datetime import timedelta, date
import pandas as pd
from web_data import WebData
from pdf_data import PdfData
from rss_data import RssData
import json

class PushJerk:
  # Constant URL 
  PJ_URL = 'http://pushjerk.com/feed'# Webscrape 

  TYPES = {
  0 : '',
  1 : 'Hatch',
  2 : 'Juggernaut',
  3 : '7/13',
  4 : 'Texas'
  }
  
  def __init__(self, pj_type):
    _search_str = ''
    if pj_type == 'Hatch':
      _search_str = '12 – Hatch'
    elif pj_type == 'Juggernaut':
      _search_str = '/16: Inverted Juggernaut Method'
    elif pj_type == '7/13':
      _search_str = '(7/13'
    elif pj_type == 'Texas':
      _search_str = '(Texas'
    _params = {"s": _search_str, "orderby": "pubDate", "order": 'ASC'}
    self._rss_data = RssData(PushJerk.PJ_URL, pj_type, params=_params)
    self.wods_json = self._rss_data.wods_json

class Gpp:  
  TYPES = {
  0 : 'NOPERATOR',
  1 : 'ATHLETICS',
  2 : 'GARAGE',
  3 : 'TSAC',
  4 : 'SSLP'
  }

  def __init__(self, type=0, allowed_types=TYPES):
    if type not in allowed_types:
            raise ValueError("%s is not a valid type. Please enter a number 1, 2, 3, or 4" % type)
    self.type = type

class Workout():
  TYPES = {
  0 : 'STRENGTH',
  1 : 'METCON',
  }
  
  def __init__(self, workout_type=0, gpp=None):
    self.workout_type = workout_type
    self.gpp = gpp 

class Strength(Workout):
  
  def __init__(self, *args, **kwargs):
    self.one_rm_bs = self.get_one_rm_bs()  
    super(Strength, self).__init__(*args, **kwargs)

  def get_one_rm_bs(self) -> int:
    one_rm_bs = int(input("What is your 1RM Back Squat?\n"))
    return one_rm_bs 


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
  # Constant URL 
  DEUCE_URL = "http://www.deucegym.com/community/" # Webscrape
  
  def __init__(self, gpp_type, *args, **kwargs):
    super(Deuce, self).__init__(*args, **kwargs)
    self._web_data = WebData(Deuce.DEUCE_URL, gpp_type, self.workout_dates)
    self.cycle_wods_json = self._web_data.wods_json()

class Tsac(Cycle):
  MASH_PDF = "mash-evolution.pdf" # PDF
  # Pages of the mash-evolution TSAC macrocycle 
  PDF_SC_LV_PAGES = [*range(379, 420, 1)]
  
  def __init__(self, *args, **kwargs):
    super(Tsac, self).__init__(*args, **kwargs)
    self.pdf_data = PdfData(Tsac.MASH_PDF, Tsac.PDF_SC_LV_PAGES, self.workout_dates)
    self.cylce_wods_json = self.pdf_data.load_pdf_to_json()
      
class Sslp(Strength, Cycle):
  # csv to store the SSLP Macro Cycle
  SSLP_CSV = "sslp.csv"
  SSLP_URL = "https://startingstrength.com/get-started/programs" # Webscrape

  def __init__(self, *args, **kwargs):
    Strength.__init__(self, workout_type=0, gpp=Gpp.TYPES[4])
    Cycle.__init__(self, **kwargs)
    #super(Sslp, self).__init__(*args, **kwargs)    
    self._df_sslp = self.init_df_from_csv()
    self.cycle_wods_json = self.generate_cycle_wods_json()

  def init_df_from_csv(self) -> pd.DataFrame:
    try:
      csv_loaded = False
      df_sslp = self.load_csv(Sslp.SSLP_CSV)
      if len(df_sslp.index) == 0:
        wd = WebData(Sslp.SSLP_URL, Gpp.TYPES[4]) #SSLP
        csv_loaded = wd.webscrape_data_to_csv(self.workout_dates)
        if csv_loaded:
          df_sslp = self.load_csv(Sslp.SSLP_CSV)
      else:
        df_sslp = self.load_csv(Sslp.SSLP_CSV)
    except Exception as e:
      print(str(e))
    finally:
      return df_sslp  

  def generate_cycle_wods_json(self) -> str:
    return self.load_data_sslp_ph1()

  # % reference => https://www.t-nation.com/training/know-your-ratios-destroy-weaknesses/
  # Bench Press: 75% of back squat
  # Powerlifting Deadlift: 120% of back squat 
  # Military Press (strict): 45% of back squat
  # Power Clean: 68% of back squat
  def calc_sslp_ph1(self) -> list:
    one_rm_bp = (self.one_rm_bs * .75)
    one_rm_sp = (self.one_rm_bs * .45)
    one_rm_dl = (self.one_rm_bs * 1.20)
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
      bs = str((self.one_rm_bs * pct_1rm) + (i * bs_pct_inc))
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
    self._df_sslp = self._df_sslp.assign(Phase_RX_Loads=self.calc_sslp_ph1())
    wods_json_str = self._df_sslp.to_json(orient='records')
    obj_data = json.loads(wods_json_str)
    json_formatted_str += json.dumps(obj_data, indent=4) 
    return json_formatted_str