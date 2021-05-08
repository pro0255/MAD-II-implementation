import pandas as pd
import networkx as nx
from logic.LinkPrediction import apply_threshold, create_confusion_tuple, prediction_link_dictionary
from logic.CONSTANTS import STATIC_THRESHOLD
from logic.Performance import Performace
import time
from logic.GraphAnalysis import GraphAnalysis

class Controller:
    def __init__(self):
        pass


    def load_ds(self, path):
        """
        Data should be in format edge; edge csv file.
        """
        try:
            self.ds = pd.read_csv(path, sep=';', header=None)
            return True
        except:
            return False

    def create_G(self):
        """
        Creates graph from edges list.
        """
        self.G = nx.from_pandas_edgelist(self.ds, 0, 1)
        self.adj_A = nx.to_numpy_matrix(self.G, dtype='uint8')
        

    def run_prediction(self, index, threshold=STATIC_THRESHOLD):
        print('Start prediction')
        predictions = {}
        methods = list(prediction_link_dictionary.values())
        methods_name = list(prediction_link_dictionary.keys())

        method_name = methods_name[index].value
        selected_method = methods[index]

        prediction_results = self.make_calculation(self.adj_A, selected_method, threshold, method_name)
        prediction_results[1].threshold = threshold

        predictions[method_name] = prediction_results
        print('End prediction')
        return predictions



    def make_calculation(self, matrix, method, threshold=STATIC_THRESHOLD, method_type=""):
        start = time.time()
        matrix_method = method(matrix)
        matrix_method_thresholded = apply_threshold(matrix_method, threshold)
        title = f"{method_type}"
        calculated_tuple = create_confusion_tuple(matrix, matrix_method_thresholded, matrix_method, title)
        perf = Performace(title)
        perf.calculate(calculated_tuple)
        end = time.time()
        perf.time = end - start
        return calculated_tuple, perf

    def make_step_analysis(self, f, to, step, method_index):
        predictions = {}

        methods = list(prediction_link_dictionary.values())
        methods_name = list(prediction_link_dictionary.keys())

        method_name = methods_name[method_index].value
        selected_method = methods[method_index]

        i = f
        while i < to:
            prediction_results = self.make_calculation(self.adj_A, selected_method, i, method_name)
            prediction_results[1].threshold = i
            predictions[i] = prediction_results
            i+=step
            
        return predictions

    def make_graph_analysis(self):
        g = GraphAnalysis()
        g.make_calculation(self.G)
        return g
