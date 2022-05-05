import tkinter as tk
from tkinter import ttk
from tkhtmlview import HTMLLabel
from tkcalendar import Calendar
from wod import PushJerk

class View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # create widgets
        # label
        self.wod_label = ttk.Label(self, text="Pick a WOD Type:")
        self.wod_label.grid(row=1, column=0)

        # option menu
        self.pj_type_var = tk.StringVar(self)
        self.pj_type_var.set(PushJerk.TYPES[0]) # default value no search criteria

        self.option_menu = ttk.OptionMenu(self, self.pj_type_var, *PushJerk.TYPES.values())
        self.option_menu.grid(column=1, row=0, padx=10, pady=10,  sticky='w')    
        self.option_menu.grid(row=1, column=1, sticky=tk.NSEW)

        # go button
        self.go_button = ttk.Button(self, text='Go', command=self.go_button_clicked)
        self.go_button.grid(row=1, column=3, padx=10, sticky='w')

        # tree view WODs scraped display for selection
        self.tree_view = ttk.Treeview(self, show="headings", columns=("WODs"))
        self.tree_view.heading("#1", text="WODs")
        self.tree_view.grid(row=2, columnspan=2, padx=10, pady=10, sticky='nw') 
        self.tree_view.bind("<Double-1>", self.handle_double_click)

        # HTMLLable view selected 
        self.html_label = HTMLLabel(self)
        self.html_label.grid(row=2, column=3, columnspan=6, rowspan=3, sticky='w')

        self.cal = Calendar(self, font="Helvetica 11", selectmode='day', locale='en_US',
                   cursor="hand1", year=2022)
        self.cal.grid(row=2, column=5, padx=10, pady=10, sticky='ne')
        # TODO: hide unless handle_double_click
        

        # set the controller
        self.controller = None

        # style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', bgcolor='#282828', font=('Helvetica', 11), color='white')
        self.style.configure('TButton', bgcolor='#282828', font=('Helvetica', 11), color='white')
        self.style.configure('Treeview', bgcolor='#282828', font=('Helvetica', 11), color='white')    
        self.style.configure('HTMLLabel', bgcolor='#282828', font=('Helvetica', 11), color='white')    

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller

    def go_button_clicked(self):
        """
        Handle button click event
        :return:
        """
        if self.controller:
            pj_type_selected = self.pj_type_var.get()
            children = self.tree_view.get_children()
            if len(children) > 0:
                if children[0].find(pj_type_selected) != -1:
                    pass # not going to re-populate treeview is type already selected
                else: 
                    for child in children:
                        self.tree_view.delete(child)       
                        self.html_label.set_html('')    
            pj_wods_json = self.controller.scrape(pj_type_selected)
            if len(pj_wods_json) == 1: # CNX Error
                value = pj_type_selected + '_' + str(pj_wods_json[0]["wod_date"][0:3]) + '_Error--''re-GO'''
                self.tree_view.insert("", "end", values=(value), text=str(pj_wods_json[0]["wod_details"]))
            else:
                i:int = 0
                for row in pj_wods_json:
                    i += 1
                    value = pj_type_selected + '_' + str(row["wod_date"][0:3]) + '_' + str(i)
                    #label = HTMLLabel(self, html=str(row["content"]))
                    self.tree_view.insert("", "end", values=(value), text=str(row["wod_details"]))  

    def handle_double_click(self, event):
        item = self.tree_view.identify('item',event.x,event.y)
        div_style_open_tag = '<div style="Arial, Helvetica, sans-serif; font-size: 10px;>"'
        div_style_close_tag = "</div>"
        html = div_style_open_tag + self.tree_view.item(item,"text") + div_style_close_tag
        self.html_label.set_html(html)