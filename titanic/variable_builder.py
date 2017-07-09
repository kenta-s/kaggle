import numpy as np
import pandas as pd
from IPython import embed

class VariableBuilder():
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def __call__(self):
        valid_data = self.build_variable_x()
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

    def build_train_variable(self):
        sex_list = list(map(VariableBuilder.convert_sex_to_int, self.df.Sex))
        age_list = list(map(lambda x: 0.0 if np.isnan(x) else x, self.df.Age))
        embarked_list = list(map(VariableBuilder.convert_embarked_to_int, self.df.Embarked))
        valid_data = np.array([
            self.df.Pclass,
            sex_list,
            age_list,
            self.df.SibSp,
            self.df.Parch,
            self.df.Fare,
            embarked_list,
            self.df.Survived
        ]).astype(np.float32)
        data = list(map(lambda x: (np.array(x[0:7]), np.array(x[7]).astype(np.int32)), valid_data.T))
        return data

    def build_test_variable(self, file):
        sex_list = list(map(VariableBuilder.convert_sex_to_int, self.df.Sex))
        age_list = list(map(lambda x: 0.0 if np.isnan(x) else x, self.df.Age))
        embarked_list = list(map(VariableBuilder.convert_embarked_to_int, self.df.Embarked))
        df2 = pd.read_csv(file)
        survived = df2.Survived
        valid_data = np.array([
            self.df.Pclass,
            sex_list,
            age_list,
            self.df.SibSp,
            self.df.Parch,
            self.df.Fare,
            embarked_list,
            survived
        ]).astype(np.float32)
        data = list(map(lambda x: (np.array(x[0:7]), np.array(x[7]).astype(np.int32)), valid_data.T))
        return data
