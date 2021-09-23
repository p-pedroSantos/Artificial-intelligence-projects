import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_boston
boston = load_boston()

print(boston.DESCR)

data = pd.DataFrame(boston.data, columns=boston.feature_names)

data.head()

data.to_csv('data.CSV')

data['MEDV']= boston.target

data.head()

data.describe()

#!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
from pandas_profiling import ProfileReport

profile = ProfileReport(data, title ="Relatorio - Pandas Profiling")

profile

profile.to_file(output_file="relatorio01.html")
