import pdfplumber
import pandas as pd
import json

class PdfData:
    def __init__(self, file_path: str, pages: list, workout_dates: list):
        self.file_path = file_path
        self.pages = pages
        self.workout_dates = workout_dates

    # Replace \n and Unicode right quote 
    @staticmethod
    def replace_chars(s: str) -> str:
        s = s.replace('\n', '').replace('\u2019', "'")
        return s  

    # Recursion to clean-up data
    def recursively_apply(self, l, f):
        for n, i in enumerate(l):
            if isinstance(i, list): # check if i is type list
                l[n] = self.recursively_apply(l[n], f)
            elif isinstance(i, str): # check if i is type str
                l[n] = f(i)
            elif i is None:
                l[n] = '' # nothing to replace there can be only one instance of None
        return l    

    def extract_data_from_tables(self) -> list:
        # Load the PDF
        pdf = pdfplumber.open(self.file_path)
        list_wods = []
        # Need to loop thru every other page of PDF_SC_LV_PAGES
        for page in self.pages[::2]:
            #print('pdf_page: ', page)
            # Save data to a pandas dataframe.
            # returns a list of lists, with each inner list representing a row in the table. 
            p = pdf.pages[page]
            list_wods.append(p.extract_tables())
        return list_wods
    
    def clean_parse_data(self, lst_pdf_data) -> list:
        # Iterate the lists and clean-up/parse strings
        lst_wods = self.recursively_apply(lst_pdf_data, self.replace_chars)
        # Get the data into a dataframe 
        return lst_wods

    def assign_weekly_date_to_wod_data(self, df_wods) -> list:
        lst_weekly_wod_data = []
        i = 0
        for wod_date in self.workout_dates[::5]: # 5-day "work week"
            # iterate through each row and assign date
            for idx in range(len(df_wods)):
                ret = wod_date, *df_wods.iloc[i]
                lst_weekly_wod_data.append(ret)
                i += 1
                break
        return lst_weekly_wod_data

    # Returns a formatted JSON string from data loaded from PDF
    def load_pdf_to_json(self) -> str: # JSON serialized string object
        json_formatted_str = ''
        # Get the data out of the PDF
        lst_pdf_data = self.extract_data_from_tables()
        # Clean data
        lst_pdf_data = self.clean_parse_data(lst_pdf_data)
        # Pass the list of lists into a DataFrame constructor
        df_wods = pd.DataFrame(lst_pdf_data)
        # print(df_wods)
        col_names = ['start_week', 'day_1', 'day_2', 'day_3']
        df_weekly_wods = pd.DataFrame(self.assign_weekly_date_to_wod_data(df_wods), columns=col_names)
        # print(df_weekly_wods)
        wods_json_str = df_weekly_wods.to_json(orient='records')
        obj_data = json.loads(wods_json_str)
        json_formatted_str += json.dumps(obj_data, indent=4)  
        return json_formatted_str