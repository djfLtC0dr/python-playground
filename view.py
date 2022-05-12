import tkinter as tk
from tkinter import Event, ttk
from tkhtmlview import HTMLLabel
from tkcalendar import DateEntry
from wod import PushJerk

class View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()
        # Set the theme
        self.tk.call("source", "azure.tcl")
        self.tk.call("set_theme", "dark")
        # set the controller
        self.controller = None 
        
    def create_widgets(self):
        # label
        self.wod_label = ttk.Label(self, text="Pick a WOD Type:")
        self.wod_label.grid(row=1, column=0, padx=5, pady=5)

        # option menu
        self.pj_type_var = tk.StringVar(self)
        self.pj_type_var.set(PushJerk.TYPES[0]) # default value no search criteria

        self.option_menu = ttk.OptionMenu(self, self.pj_type_var, *PushJerk.TYPES.values())
        self.option_menu.grid(column=1, row=1, padx=5, pady=10,  sticky=tk.W)    
        #self.option_menu.grid(row=1, column=1, sticky=tk.NSEW)

        # go button
        self.btn_go = ttk.Button(self, text='Get WODs')
        self.btn_go.grid(row=1, column=3, padx=10, sticky=tk.W)

        # tree view WODs scraped display for selection
        self.tree_view = ttk.Treeview(self, show="headings", columns=("WODs"), height=8)
        self.tree_view.heading("#1", text="WODs")
        self.tree_view.grid(row=2, columnspan=2, padx=10, sticky=tk.NW) 

        # HTMLLable view selected 
        self.html_label = HTMLLabel(self, background='#333330', height=12, width=28)
        self.html_label.grid(row=2, column=3, columnspan=2, pady=5)

        # Calendar Label
        self.lbl_dt = ttk.Label(self, text='Choose date:', width=11)
        self.lbl_dt.grid(row=3, column=0, sticky=tk.W)

        # Calendar
        self.cal = DateEntry(self, width=10, cursor='hand1')        
        self.cal.grid(row=3, column=1, padx=5, sticky=tk.W)
    
        # Squat Label
        self.lbl_sqt = ttk.Label(self, text='Enter 5RM Squat:', width=17)
        self.lbl_sqt.grid(row=3, column=3, sticky=tk.W)
        # Squat Entry
        # declaring string variable for storing entry
        self.sqt_var = tk.StringVar(self)
        vcmd = self.register(self.validate_digits_only)
        self.entry_sqt = ttk.Entry(self, width=3, textvariable=self.sqt_var, validate='key', validatecommand=(vcmd, '%P'))
        self.entry_sqt.grid(row=3, column=3, sticky=tk.E)        
        # Squat Save
        self.btn_sqt = ttk.Button(self, text='Save', width=5)
        self.btn_sqt.grid(row=3, column=4, sticky=tk.E)

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller

    def validate_digits_only(self, P) -> bool:
        try:
            if P.isdigit():
                v = int(P)
                if v < 0 or v > 999:
                    raise ValueError
                else:
                    # self.enable_save()
                    self.btn_sqt.config(state=(tk.NORMAL if P else tk.DISABLED))
                    return True                    
            else:
                v = str(P)
                if v != "\x08" and v != '':
                    return False
                else:
                    # self.enable_save()
                    self.btn_sqt.config(state=(tk.NORMAL if P else tk.DISABLED))
                    return True
        except ValueError as ve:
            print(ve)
            return False

    def get_input_date(self):
        return self.cal.get_date()
    
    def get_input_sqt(self):
        return self.sqt_var.get()
