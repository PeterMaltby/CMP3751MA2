import pandas
from sklearn.model_selection import KFold

#cross-validation of classifcation models.
#Peter Maltby - 2021-02-03

#reads data prints exert and produces stastical measure table output to "dataset_stats.xlsx"
data = pandas.read_excel (r'clinical_dataset.xlsx')

print("hello world");