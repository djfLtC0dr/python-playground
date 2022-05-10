import tkinter as tk
from tkinter import Event, ttk
from tkhtmlview import HTMLLabel
from tkcalendar import DateEntry
from wod import PushJerk

class View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # create widgets
        # label
        self.wod_label = ttk.Label(self, text="Pick a WOD Type:")
        self.wod_label.grid(row=1, column=0, padx=5, pady=5)

        # option menu
        self.pj_type_var = tk.StringVar(self)
        self.pj_type_var.set(PushJerk.TYPES[0]) # default value no search criteria

        self.option_menu = ttk.OptionMenu(self, self.pj_type_var, *PushJerk.TYPES.values(), command=self.handle_option_selected)
        self.option_menu.grid(column=1, row=1, padx=5, pady=10,  sticky=tk.W)    
        #self.option_menu.grid(row=1, column=1, sticky=tk.NSEW)

        # go button
        self.go_button = ttk.Button(self, text='Get WODs', state=tk.DISABLED, command=self.handle_go_button_clicked)
        self.go_button.grid(row=1, column=3, padx=10, sticky=tk.W)
        

        # tree view WODs scraped display for selection
        self.tree_view = ttk.Treeview(self, show="headings", columns=("WODs"), height=8)
        self.tree_view.heading("#1", text="WODs")
        self.tree_view.grid(row=2, columnspan=2, padx=10, sticky=tk.NW) 
        self.tree_view.bind("<Double-1>", self.handle_double_click)

        # HTMLLable view selected 
        self.html_label = HTMLLabel(self, background='#333330', height=12, width=28)
        self.html_label.grid(row=2, column=3, columnspan=2)

        # Calendar Label
        self.lbl_dt = ttk.Label(self, text='Choose date:', width=11)
        self.lbl_dt.grid(row=3, column=0, sticky=tk.E)

        # Calendar
        self.cal = DateEntry(self, width=10, cursor='hand1')        
        # self.cal = Calendar(self, selectmode='day', locale='en_US', selectbackground='#007fff', selectforeground='#000000',
        #            selection_callback=self.get_selection, cursor="hand1", year=2022, font='Helvetica 11', showweeknumbers=False)
        self.cal.grid(row=3, column=1, pady=5, sticky=tk.W)
        self.cal.bind("<<DateEntrySelected>>", self.get_date_selected)

        # Squat Label
        self.lbl_sqt = ttk.Label(self, text='Enter 5RM Squat:', width=15)
        self.lbl_sqt.grid(row=3, column=3, sticky=tk.W)
        # Squat Entry
        vcmd = self.register(self.validate_digits_only)
        self.entry_sqt = ttk.Entry(self, width=3, validate='key', validatecommand=(vcmd,'%P'))
        self.entry_sqt.grid(row=3, column=4, sticky=tk.W)        
        # Squat Save
        # TODO: Save button click to mongodb

        # set the controller
        self.controller = None

        # Set the theme
        self.tk.call("source", "azure.tcl")
        self.tk.call("set_theme", "dark")     
        
        # Hide widgets until show widgets called
        self.remove_widgets()   

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller

    def handle_option_selected(self, *args):
        btn_state = self.go_button.state()
        if len(btn_state) > 0:
            if btn_state[0] == tk.DISABLED:
                self.go_button['state'] = tk.NORMAL
            else:
                self.go_button['state'] = tk.DISABLED
        children = self.tree_view.get_children()
        if len(children) > 0:
            for child in children:
                self.tree_view.delete(child)                     
        self.remove_widgets()

    def handle_go_button_clicked(self):
        """
        Handle button click event
        :return:
        """
        if self.controller:
            pj_type_selected = self.pj_type_var.get()
            children = self.tree_view.get_children()
            if len(children) > 1:
                for child in children:
                    if self.tree_view.item(child)["values"][0].find(pj_type_selected) != -1:
                        pass # not going to re-populate treeview if type already selected
                    else: 
                        self.tree_view.delete(child)
            elif len(children) == 1: # There was an error of sorts scraping data
                for child in children:
                   self.tree_view.delete(child)
            self.remove_widgets()
            pj_wods_json = self.controller.scrape(pj_type_selected)
            if len(pj_wods_json) == 1: # CNX Error
                value = pj_type_selected + '_' + str(pj_wods_json[0]["wod_date"][0:3]) + '_Error--''retry'''
                self.tree_view.insert("", "end", values=(value), text=str(pj_wods_json[0]["wod_details"]))
                self.remove_widgets() 
            else:
                i:int = 0
                for row in pj_wods_json:
                    i += 1
                    value = pj_type_selected + '_' + str(row["wod_date"][0:3]) + '_' + str(i)
                    self.tree_view.insert("", "end", values=(value), text=str(row["wod_details"]))  

    def handle_double_click(self, event):
        item = self.tree_view.identify('item',event.x,event.y)
        div_style_open_tag = '<div style="Arial, Helvetica, sans-serif; font-size: 10px; color:white;">'
        div_style_close_tag = '</div>'
        html_text = self.tree_view.item(item,"text")
        html = div_style_open_tag + html_text + div_style_close_tag
        self.html_label.set_html(html)
        self.show_widgets(html_text)

    def remove_widgets(self):
        self.html_label.grid_remove()   
        self.lbl_dt.grid_remove()
        self.cal.grid_remove()
        self.lbl_sqt.grid_remove()
        self.entry_sqt.grid_remove()

    def show_widgets(self, html_text: str):
        if html_text.find('ConnectionResetError([54,104]') == -1: # no weired connection error
            self.html_label.grid()
            self.lbl_dt.grid()
            self.cal.grid()
            self.lbl_sqt.grid()
            self.entry_sqt.grid()       

    def validate_digits_only(self, key: str) -> bool:
        try:
            if key.isdigit():
                v = int(key)
                if v < 0 or v > 999:
                    raise ValueError
                else:
                    return True
            else:
                v = str(key)
                if v != "\x08" and v != '':
                    return False
                else:
                    return True
        except ValueError as ve:
            print(ve)
            return False
    
    def get_date_selected(self, e):
        print(self.cal.get_date())