from operator import indexOf
import pdfplumber
import pandas as pd
import json

class PdfData:
    MASH = "mash-evolution.pdf" # PDF
    # Pages of the mash-evolution TSAC macrocycle 
    PDF_SC_LV_PAGES = [*range(379, 420, 1)]

    def __init__(self, pages: list, workout_dates: list):
        self.pages = pages
        self.workout_dates = workout_dates

    # Replace \n and Unicode right quote 
    def replace_chars(self, s: str) -> str:
        s = s.replace('\n', '').replace('\u2019', "'")
        return s  

    # TODO This still needs work
    def load_pdf_to_json(self) -> str: # JSON serialized string object
        json_formatted_str = ''
        # Load the PDF
        pdf = pdfplumber.open(PdfData.MASH)
        len_date_list = len(self.workout_dates)
        i = 0
        # Need to loop thru every other page of PDF_SC_LV_PAGES
        while i < len_date_list:
            for page in self.pages[::2]:
                #print('pdf_page: ', page)
                # Save data to a pandas dataframe.
                # returns a list of lists, with each inner list representing a row in the table. 
                p = pdf.pages[page]
                list_wods = p.extract_tables() 

                # Recursion to clean-up data
                def recursively_apply(l, f):
                    for n, i in enumerate(l):
                        if isinstance(i, list): # check if i is type list
                            l[n] = recursively_apply(l[n], f)
                        elif isinstance(i, str): # check if i is type str
                            l[n] = f(i)
                        elif i is None:
                            l[n] = '' # nothing to replace there can be only one instance of None
                    return l
                
                # Iterate the lists and clean-up/parse strings
                list_wods = recursively_apply(list_wods, self.replace_chars)
                # Get the data into a dataframe 
                lst_dfs = []
                for l_wod in list_wods:
                    while i < len_date_list:
                        #print('date: ', i)
                        wod_date = self.workout_dates[i]
                        for wod in l_wod:
                            # dictionary storing the data
                            wod_data = {
                                'Day': wod_date,
                                'ROE': wod[0],
                                'RPE': wod[1]
                            }
                            #create DataFrame by passing dictionary wrapped in a list
                            odf = pd.DataFrame.from_dict([wod_data])
                            lst_dfs.append(odf)
                        i += 1
                        df_wods = pd.concat(lst_dfs, ignore_index=True)
                    wods_json_str = df_wods.to_json(orient='records')
                obj_data = json.loads(wods_json_str)
            json_formatted_str += json.dumps(obj_data, indent=4)  
        return json_formatted_str