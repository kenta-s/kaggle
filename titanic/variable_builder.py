import numpy as np
import pandas as pd
from IPython import embed

class VariableBuilder():
    def __init__(self):
        self.df = pd.read_csv('train.csv')

    def __call__(self):
        valid_data = self.build_variable()
        valid_data = np.array(valid_data).astype(np.float32).T
        return valid_data

    @staticmethod
    def convert_sex_to_int(str):
        if str == 'male':
            return 0
        elif str == 'female':
            return 1
        else:
            return 2

    @staticmethod
    def convert_embarked_to_int(str):
        if str == 'S':
            return 0
        elif str == 'C':
            return 1
        elif str == 'Q':
            return 2
        else:
            return 3

    def build_variable(self):
        sex_list = list(map(VariableBuilder.convert_sex_to_int, self.df.Sex))
        age_list = list(map(lambda x: 0.0 if np.isnan(x) else x, self.df.Age))
        embarked_list = list(map(VariableBuilder.convert_embarked_to_int, self.df.Embarked))
        return [
            self.df.Pclass,
            sex_list,
            age_list,
            self.df.SibSp,
            self.df.Parch,
            self.df.Fare,
            embarked_list
        ]
