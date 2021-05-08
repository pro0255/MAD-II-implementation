
class Performace:
    def __init__(self, title=""):
        pass
        self.sensitivity = 0
        self.specificity = 0
        self.precision = 0
        self.recall = 0
        self.fallout = 0
        self.accuracy = 0
        self.title = title
        self.time = 0
        self.threshold = 0


    def make_calculation(self, nom, den):
        if den == 0:
            return 0
        return nom/den

    def calc_sensitivity(self, true_pos, true_neg, false_pos, false_neg):
        nom = true_pos
        den = true_pos + false_neg
        self.sensitivity = self.make_calculation(nom, den)

    def calc_specificity(self, true_pos, true_neg, false_pos, false_neg):
        nom = true_neg
        den = false_pos + true_neg
        self.specificity = self.make_calculation(nom, den)
        
    def calc_precision(self, true_pos, true_neg, false_pos, false_neg):
        nom = true_pos
        den = true_pos + false_pos
        self.precision = self.make_calculation(nom, den)

    def calc_recall(self, true_pos, true_neg, false_pos, false_neg):
        nom = true_pos
        den = true_pos + false_neg
        self.recall = self.make_calculation(nom, den)

    def calc_fallout(self, true_pos, true_neg, false_pos, false_neg):
        nom = false_pos
        den = false_pos + true_neg
        self.fallout = self.make_calculation(nom, den)

    def calc_accuracy(self, true_pos, true_neg, false_pos, false_neg):
        nom = true_pos + true_neg
        p = true_pos + false_pos
        n = true_neg + false_neg
        den = p + n
        self.accuracy = self.make_calculation(nom, den)

    def calculate(self, tuple_confusion_matrix):
        self.calc_sensitivity(*tuple_confusion_matrix)
        self.calc_specificity(*tuple_confusion_matrix)
        self.calc_precision(*tuple_confusion_matrix)
        self.calc_recall(*tuple_confusion_matrix)
        self.calc_fallout(*tuple_confusion_matrix)
        self.calc_accuracy(*tuple_confusion_matrix)

    def get_dictionary(self):
        dic = {}
        dic['Sensitivity'] = self.sensitivity
        dic['Specificity'] = self.specificity
        dic['Precision'] = self.precision
        dic['Recall'] = self.recall
        dic['Fallout'] = self.fallout
        dic['Accuracy'] = self.accuracy
        dic['Time'] = self.time
        dic['Threshold'] = self.threshold
        return dic

    def __str__(self):
        res = f"{self.title.upper()}\n\n"
        res += f'\tSensitivity:{self.sensitivity}\n'
        res += f'\tSpecificity:{self.specificity}\n'
        res += f'\tPrecision:{self.precision}\n'
        res += f'\tRecall:{self.recall}\n'
        res += f'\tFallout:{self.fallout}\n'
        res += f'\tAccuracy:{self.accuracy}\n'
        return res

        


