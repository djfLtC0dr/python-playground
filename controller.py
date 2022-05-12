import tkinter as tk
from tkinter import Event, ttk
import traceback

class Controller:
    def __init__(self, model, view):
      self.model = model
      self.view = view
      self.view.option_menu.bind('<Button>', self.handle_option_selected)
      self.view.btn_go['state'] = tk.DISABLED
      self.view.btn_go.bind('<Button>', self.handle_go_button_clicked)
      self.view.tree_view.bind('<Double-1>', self.handle_double_click)
      self.view.btn_sqt['state'] = tk.DISABLED
      self.view.btn_sqt.bind('<Button>', self.add_data_to_plot) 
      # Hide widgets until show widgets called
      self.remove_widgets()        

    def handle_option_selected(self, *args):
        btn_state = self.view.btn_go.state()
        if len(btn_state) > 0:
            if btn_state[0] == tk.DISABLED:
                self.view.btn_go['state'] = tk.NORMAL
            else:
                self.view.btn_go['state'] = tk.DISABLED
        children = self.view.tree_view.get_children()
        if len(children) > 0:
            for child in children:
                self.view.tree_view.delete(child)                     
        self.remove_widgets()    

    def handle_go_button_clicked(self, *args):
        """
        Handle button click event
        :return:
        """
        pj_type_selected = self.view.pj_type_var.get()
        children = self.view.tree_view.get_children()
        if len(children) > 1:
            for child in children:
                if self.view.tree_view.item(child)["values"][0].find(pj_type_selected) != -1:
                    pass # not going to re-populate treeview if type already selected
                else: 
                    self.view.tree_view.delete(child)
        elif len(children) == 1: # There was an error of sorts scraping data
            for child in children:
                self.view.tree_view.delete(child)
        self.remove_widgets()
        if self.view.pj_type_var.get() != '': # Nothing Selected
          pj_wods_json = self.scrape(pj_type_selected)
          if len(pj_wods_json) == 1: # CNX Error
              value = pj_type_selected + '_' + str(pj_wods_json[0]["wod_date"][0:3]) + '_Error--''retry'''
              self.view.tree_view.insert("", "end", values=(value), text=str(pj_wods_json[0]["wod_details"]))
              self.remove_widgets() 
          else:
              i:int = 0
              for row in pj_wods_json:
                  i += 1
                  value = pj_type_selected + '_' + str(row["wod_date"][0:3]) + '_' + str(i)
                  self.view.tree_view.insert("", "end", values=(value), text=str(row["wod_details"]))  

    def handle_double_click(self, event):
        item = self.view.tree_view.identify('item',event.x,event.y)
        div_style_open_tag = '<div style="Arial, Helvetica, sans-serif; font-size: 10px; color:white;">'
        div_style_close_tag = '</div>'
        html_text = self.view.tree_view.item(item,"text")
        html = div_style_open_tag + html_text + div_style_close_tag
        self.view.html_label.set_html(html)
        self.show_widgets(html_text)

    def scrape(self, pj_type):
      """
      Save the email
      :param email:
      :return:
      """
      try:
        # scrape the model
        self.model.pj_type = pj_type
        return self.model.scrape()
      except:
        # show an error message
        traceback.print_exc()

    def remove_widgets(self):
        self.view.html_label.grid_remove()   
        self.view.lbl_dt.grid_remove()
        self.view.cal.grid_remove()
        self.view.lbl_sqt.grid_remove()
        self.view.entry_sqt.grid_remove()
        self.view.btn_sqt.grid_remove()

    def show_widgets(self, html_text: str):
        if html_text.find('ConnectionResetError([54,104]') == -1: # no weired connection error
            self.view.html_label.grid()
            self.view.lbl_dt.grid()
            self.view.cal.grid()
            self.view.lbl_sqt.grid()
            self.view.entry_sqt.grid()     
            self.view.btn_sqt.grid()
    
    def enable_save(self):
        btn_state = self.view.btn_sqt.state()
        if len(btn_state) > 0:
            if btn_state[0] == tk.DISABLED:
                self.view.btn_sqt['state'] = tk.NORMAL
            else:
                self.view.btn_sqt['state'] = tk.DISABLED

    def load_wod_data(self):
      try:
        clx = self.model.get_collections()
        print(clx)
        #TODO: matplotlib
      except:
        traceback.print_exc()

    def add_doc_to_db(self):
      try:
        date = self.view.get_input_date()
        sqt = self.view.get_input_sqt()
        doc = {'date': date, 'five_rm_sqt': sqt}
        if sqt != '':
          self.model.insert_doc(doc)
      except:
        traceback.print_exc()
    
    def add_data_to_plot(self, *args):
      self.add_doc_to_db()
      self.load_wod_data()