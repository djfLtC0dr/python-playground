import tkinter as tk
from tkinter import ttk
from tkhtmlview import HTMLLabel
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

        self.optionmenu = ttk.OptionMenu(self, self.pj_type_var, *PushJerk.TYPES.values())
        self.optionmenu.grid(column=1, row=0, padx=10, pady=10,  sticky='w')    
        self.optionmenu.grid(row=1, column=1, sticky=tk.NSEW)

        # go button
        self.go_button = ttk.Button(self, text='Go', command=self.go_button_clicked)
        self.go_button.grid(row=1, column=3, padx=10)

        # tree view Wod display
        self.treeview = ttk.Treeview(self, show="headings", columns=("WODs"))
        self.treeview.heading("#1", text="WODs")
        self.treeview.grid(row=2, columnspan=6, sticky='nsew') 
        self.treeview.bind("<Double-1>", self.OnDoubleClick)

        # set the controller
        self.controller = None

        # style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', bgcolor='#282828', font=('Helvetica', 11), color='white')
        self.style.configure('TButton', bgcolor='#282828', font=('Helvetica', 11), color='white')
        self.style.configure('Treeview', bgcolor='#282828', font=('Helvetica', 11), color='white')        

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
            if len(self.treeview.get_children()) > 0:
                pass # TODO: figure out how to clear 
            pj_type_selected = self.pj_type_var.get()
            pj_wods_json = self.controller.scrape(pj_type_selected)
            i:int = 0
            for row in pj_wods_json:
                i += 1
                value = pj_type_selected + str(i)
                #label = HTMLLabel(self, html=str(row["content"]))
                self.treeview.insert("", "end", values=value, text=str(row["content"])) # values=(row["pubDate"],  

    def OnDoubleClick(self, event):
        item = self.treeview.identify('item',event.x,event.y)
        print("you clicked on", self.treeview.item(item,"text"))