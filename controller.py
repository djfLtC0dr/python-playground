import tkinter as tk
import datetime
import pandas as pd
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
        for item in self.view.canvas.get_tk_widget().find_all():
           self.view.canvas.get_tk_widget().delete(item)

    def show_widgets(self, html_text: str):
        if html_text.find('ConnectionResetError([54,104]') == -1: # no web server connection error
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
    
    def add_data_to_plot(self, *args):
      if result := self.add_doc_to_db():
        self.load_data_to_df()

    def add_doc_to_db(self) -> bool:
        try:
          result = False
          date = self.view.get_input_date()
          # Need to use the datetime.combine method
          # pymongo does not accept date-encoded
          date_time = datetime.datetime.combine(date, datetime.time())
          sqt = self.view.get_input_sqt()
          doc = {'date': date_time, 'five_rm_sqt': sqt}
          if sqt != '':
            result = self.model.insert_doc(doc)
            return result.acknowledged
        except:
          traceback.print_exc()
    
    def load_data_to_df(self):
        try:
          # if we don't want to include id then pass _id:0
          query = {'_id': 0, 'date': 1, 'five_rm_sqt': 1} 
          #clx = self.model.get_collections()
          clx = self.model.get_collection()
          li = []
          for x in clx.find({}, query): 
            li.append(x)
          df = pd.DataFrame(li)
          df['date']= pd.to_datetime(df['date'], format="%Y,%m,%d %H:%M%z")
          df=df.set_index('date')
          self.view.plot_wod_data(df)
        except:
          traceback.print_exc()