from tkinter import *
from config import WIDTH, HEIGHT, WINDOW_NAME
from tkinter import filedialog
from logic.LinkPrediction import prediction_link_dictionary
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



METHODS = [k.value for k in prediction_link_dictionary.keys()]


class Application:
    def __init__(self, controller):
        self.c = controller
        self.window = Tk(className=f'{WINDOW_NAME}')
        self.window.attributes("-fullscreen", True)
        self.window.geometry(f'{WIDTH}x{HEIGHT}')
        self.window.columnconfigure(1, weight=int(WIDTH/2))
        self.window.columnconfigure(2, weight=int(WIDTH/2))


        self.title = Label(self.window, text="Aplikace na predikci linku", fg="red",  font=("Helvetica", 25))
        self.title.grid(row=1, column=1, columnspan=2)


        self.filename_entry_value = StringVar()
        self.filename_entry = Entry(self.window,
                                text = self.filename_entry_value,
                                fg = "blue",
                                state='disabled'   
                                )

    
        self.filename_entry.grid(row=2, column=1, sticky='nsew')
        self.button_explore = Button(
                        self.window,
                        text = "Browse files",
                        command = self.browseFilesSelect,
        )
        self.button_explore.grid(row=2, column=2, sticky='nsew')


        self.button_submit = Button(
                self.window,
                text = "Load",
                state=DISABLED,
                command=self.load_ds
        )
        self.button_submit.grid(row=3, column=2, columnspan=1, sticky='nsew')


        self.checkbox_value = IntVar()
        self.checkbox = Checkbutton(self.window, text='Visualization',variable=self.checkbox_value, onvalue=1, offvalue=0, state=DISABLED)
        self.checkbox.grid(row=3, column=1, columnspan=1, sticky='nsew')



    def mount_save_gui(self):

        self.title3 = Label(self.window, text="Výběr složky pro výstup", fg="red",  font=("Helvetica", 25))
        self.title3.grid(row=10, column=1, columnspan=2)

        self.folder_entry_value_save = StringVar()
        self.folder_entry_save = Entry(self.window,
                                text = self.folder_entry_value_save,
                                fg = "blue",
                                state='disabled'   
                                )

    
        self.folder_entry_save.grid(row=11, column=1, sticky='nsew')

        self.button_browse_save = Button(
                        self.window,
                        text = "Select output folder",
                        command = self.select_output_folder,
        )
        self.button_browse_save.grid(row=11, column=2, sticky='nsew')

        self.save_filename = StringVar()
        self.save_filename_entry = Entry(self.window,
                                text = self.save_filename,
                                fg = "blue",
                                state='disabled'   
        )

        self.save_filename_entry.grid(row=12, column=1, columnspan=2, sticky='nsew')



        self.button_save = Button(
                        self.window,
                        text = "Save prediction",
                        state=DISABLED,
                        command = self.save_prediction_action,
        )
        self.button_save.grid(row=13, column=1, columnspan=2, sticky='nsew')


    def save_prediction_action(self):
        path = f'{self.folder_entry_value_save.get()}/{self.save_filename.get()}_t{self.threshold_value.get()}.xlsx'
        self.save_prediction(self.predictions, path)
        print('saving')


    def unmount_save_gui(self):
        self.folder_entry_value_save.grid_remove()
        self.save_filename_entry.grid_remove()
        self.button_save.grid_remove()


    def select_output_folder(self):
        self.folder_save = filedialog.askdirectory(initialdir = ".//outputs",
                                            title = "Select folder")
        self.folder_entry_value_save.set(self.folder_save)

        print()
        f_i = self.filename.rindex('/')
        name_of_input_file = self.filename[f_i+1:].replace('.', '')
        method = list(self.predictions.keys())[0]
        value = f'{name_of_input_file}_{method}'
        self.save_filename.set(value)
        self.save_filename_entry['state'] = NORMAL
        self.button_save['state'] = NORMAL


    def unmount_selection_gui(self):
        self.checkbar.grid_remove()
        self.button_start.grid_remove()

    def disable_start_button(self, selected):
        if selected:
            self.button_start['state'] = NORMAL
            self.threshold_entry['state'] = NORMAL
        else:
            self.button_start['state'] = DISABLED
            self.threshold_entry['state'] = DISABLED
             

    def mount_properties_gui(self):
        properties = self.graph_analysis.get_analysis_dictionary() 
        print(properties)
        text_widget_input = ''
        for k, v in properties.items():
            text_widget_input += f'{k}: {v}\n'


        
        self.properties_text = Text()
        self.properties_text.grid(row=4, column=1, columnspan=1, sticky='nsew')
        self.properties_text.insert(END, text_widget_input) 
        self.properties_text.config(state=DISABLED)

        if self.checkbox_value.get():
            fig, ax = plt.subplots()
            nx.draw_spring(self.c.G, cmap=plt.get_cmap('jet'), with_labels = True, ax=ax)
            self.canvas = FigureCanvasTkAgg(fig, self.window) 
            self.canvas.get_tk_widget().grid(row=4, column=2, columnspan=1, sticky='nsew')


    def mount_selection_gui(self):
        self.title2 = Label(self.window, text="Výběr metody a parametrů pro predikci", fg="red",  font=("Helvetica", 25))
        self.title2.grid(row=6, column=1, columnspan=2)



        self.checkbar = Checkbar(self.window, METHODS, self.disable_start_button)
        self.checkbar.grid(row=7, column=1, columnspan=2, sticky='nsew')

        self.button_start = Button(
            self.window,
            text = "Start",
            state='disabled',
            command=self.start_prediction
        )
        self.button_start.grid(row=9, column=1, columnspan=2, sticky='nsew')

        self.threshold_value = StringVar()
        self.threshold_entry = Entry(self.window,
                                text = self.threshold_value,
                                fg = "red",
                                state='disabled'
                                )
        self.threshold_entry.grid(row=8, column=1,columnspan=2, sticky='nsew')



    def start_prediction(self):
        method_index = list(self.checkbar.state()).index(1)
        self.run_prediction(method_index)

        #Run prediction with save to file - outputs excel with sheets
    def create_prediction_df(self, prediction, columns=['TODO']):
        dic = {}
        confusion_labels = ['TRUE POSITIVE', 'TRUE NEGATIVE', 'FALSE POSITIVE', 'FALSE NEGATIVE']
        confusion_tuple, performance = prediction

        for label_index, label in enumerate(confusion_labels):
            value = confusion_tuple[label_index]
            dic[label] = value

        for k, v in performance.get_dictionary().items():
            dic[k] = v
        
        df = pd.DataFrame.from_dict(dic, orient='index', columns=columns)

        return df

    def save_prediction(self, predictions, path):
        predictions_keys = list(predictions.keys())
        dfs = []

        for k in predictions_keys:
            prediction = predictions[k]
            dfs.append(self.create_prediction_df(prediction, [k]))

        with pd.ExcelWriter(path) as writer:
            props_df = self.graph_analysis.get_df()
            props_df.to_excel(writer, sheet_name='GraphProperties')
            for index, df in enumerate(dfs):
                sheet_name = predictions_keys[index]
                df.to_excel(writer, sheet_name=sheet_name)
                writer.save()


    def run_prediction(self, index):
        threshold = int(self.threshold_value.get())
        print(threshold)
        self.predictions = self.c.run_prediction(index)
        if self.predictions is not None:
            self.mount_save_gui()
            #Show save button and browse


    def load_ds(self):
        if self.c.load_ds(self.filename_entry_value.get()):
            self.c.create_G()
            
            self.graph_analysis = self.c.make_graph_analysis()
            self.mount_properties_gui()
            self.mount_selection_gui()
            print('Loaded')
        else:
            self.unmount_selection_gui()
            print('Error')



    def start(self):
        self.window.mainloop()


    def browseFilesSelect(self):
        self.filename = filedialog.askopenfilename(initialdir = ".//",
                                            title = "Select a File",
                                            filetypes = (("all files",
                                                            "*.*"),
                                                        ("Text files",
                                                            "*.txt*"),
                                                        ))       
        self.filename_entry_value.set(self.filename)
        self.button_submit['state'] = NORMAL
        self.checkbox['state'] = NORMAL





class Checkbar(Frame):
    def __init__(self, parent=None, picks=[], cb=None, side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        self.checkbuttons = []
        self.picks = picks
        self.parent = parent
        self.side = side
        self.anchor = anchor
        self.cb = cb
        self.draw()

    def draw(self):
        for c in self.checkbuttons:
            c.pack_forget()
        self.checkbuttons = []
        self.vars = [] 
        for pick in self.picks:
            var = IntVar(self.parent)
            chk = Checkbutton(self, text=pick, variable=var, command=self.disable_rest)
            chk.pack(side=self.side, anchor=self.anchor, expand=YES)
            self.checkbuttons.append(chk)
            self.vars.append(var)

    def disable_rest(self):
        try:
            state = list(self.state())
            index = state.index(1)
            is_selected = np.argwhere(state == 1)

            if len(is_selected) == 0:
                self.cb(True)

            for i, c in enumerate(self.checkbuttons):
                if i != index:
                    c.config(state="disabled")
        except:
            self.cb(False)
            for c in self.checkbuttons:
                c.config(state="active")

    def state(self):
        return map((lambda var: var.get()), self.vars)