import pandas
import numpy
import matplotlib.pyplot
import sklearn.preprocessing
from sys import getsizeof
print("numpy version: ", numpy.__version__)
print("pands version: ", pandas.__version__)

#Data analyses of clinical dataset.xlsx
#Peter Maltby - 2021-02-01

#reads data prints exert and produces stastical measure table output to "dataset_stats.xlsx"
data = pandas.read_excel (r'clinical_dataset.xlsx')
print(data)
data.describe().to_excel("dataset_stats.xlsx")
print("data total memory size: ", getsizeof(data), "bytes.")

#counts any missing values for each var.
missingValues = data[data.columns].isnull().sum()
print("Missing values by var: \n" , missingValues)

#spliting data into two classified datasets for visualisations.
data_healthy = data.loc[data['Status'] == "healthy"]
data_cancerous = data.loc[data['Status'] == "cancerous"]

#boxplot of ages classified by status
matplotlib.pyplot.boxplot([data_healthy["Age"],data_cancerous["Age"]])
matplotlib.pyplot.title("Boxplot of clinical_dataset.xlsx age, categorised by status.")
matplotlib.pyplot.xticks([1, 2], ['Healthy', "Cancerous"])
matplotlib.pyplot.savefig('s1Boxplot')

#histogram of BMI classfied by status.
matplotlib.pyplot.clf()
matplotlib.pyplot.hist([data_healthy["BMI"],data_cancerous["BMI"]])
matplotlib.pyplot.title("Density plot of clinical_dataset.xlsx BMI, categorised by status.")
matplotlib.pyplot.legend(['Healthy', "Cancerous"])
matplotlib.pyplot.savefig('s1Densityplot')


#catagorical data converted to bool where 1=healthy.
data['Status'] = pandas.get_dummies(data['Status'],drop_first=True)

#normalises data for machine learning
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

#data is now fullyt normilised and ready for machine learning.
print(data)

