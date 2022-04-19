import pdfplumber
import pandas as pd

class PdfData:
    MASH = "mash-evolution.pdf" # PDF
    # Pages of the mash-evolution TSAC macrocycle 
    PDF_SC_LV_PAGES = [*range(379, 420, 1)]

    def __init__(self, pages: list, workout_dates: list):
        self.pages = pages
        self.workout_dates = workout_dates

    # Replace \n and * in code with empty string
    def replace_chars(self, s: str) -> str:
        s.replace('\n', '').replace('*', '') 
        return s  

    # TODO This still needs work
    def load_pdf(self) -> pd.DataFrame:
        # Load the PDF
        pdf = pdfplumber.open(PdfData.MASH)
        # Save data to a pandas dataframe.
        p0 = pdf.pages[self.pages[0]]
        # returns a list of lists, with each inner list representing a row in the table. 
        list_wods = p0.extract_tables()
        
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
            for wod in l_wod:
                # dictionary storing the data
                wod_data = {
                    'ROE': wod[0],
                    'RPE': wod[1]
                }
            #create DataFrame by passing dictionary wrapped in a list
            odf = pd.DataFrame.from_dict([wod_data])
            lst_dfs.append(odf)
        df_wods = pd.concat(lst_dfs, ignore_index=True)
        #print(df_wods)
        return df_wods