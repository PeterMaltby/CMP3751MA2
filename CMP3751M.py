import pandas
import numpy
import matplotlib.pyplot
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
matplotlib.pyplot.xlabel("Status")
matplotlib.pyplot.ylabel("Age (years)")
matplotlib.pyplot.savefig('s1Boxplot')

#histogram of BMI classfied by status.
matplotlib.pyplot.clf()
matplotlib.pyplot.hist([data_healthy["BMI"],data_cancerous["BMI"]])
matplotlib.pyplot.title("Density plot of clinical_dataset.xlsx BMI, categorised by status.")
matplotlib.pyplot.legend(['Healthy', "Cancerous"])
matplotlib.pyplot.xlabel("BMI")
matplotlib.pyplot.ylabel("Density")
matplotlib.pyplot.savefig('s1Densityplot')

#catagorical data converted to bool where 0=healthy and placed in output array (y).
y = pandas.get_dummies(data['Status'],drop_first=True)
y = y.replace({0:1,1:0})
y = numpy.ravel(y,order='C')#solves sklearn compatability issue.
data = data.drop(['Status'],1)


#normalises data for machine learning and palces data in x.
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(data)
x = scaler.transform(data)

#data is now fully normilised and ready for machine learning.
print(x)
print(y)

#split data between test and train set 1:9 ratio.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

#tmp vars for accuracy plots.
accuracyScores_test = []
accuracyScores_train = []

#iter var states how many epochs of data to run.
#creates neurel network tiwh 2 hidden layers with 500 nodes each using logisitic activation function.
iter = 350
net = MLPClassifier((500,500),'logistic',max_iter=iter);

#loops through data and records accuracy for each epoch.
for n in range (1,iter):
    #single epoch run.
    net.partial_fit(x_train, y_train, classes=numpy.unique(y_train))

    #record accuracies and add to array for graph.
    predicted = net.predict(x_test)
    accuracyScores_test.append(accuracy_score(y_test, predicted))
    print ("accuracy of neural network = ", accuracy_score(y_test, predicted))
    predicted = net.predict(x_train)
    accuracyScores_train.append(accuracy_score(y_train, predicted))
    
print("output function: ", net.out_activation_)

#code related to plotting accuracy over epochs graph.
matplotlib.pyplot.clf()
matplotlib.pyplot.plot(range(1,iter),accuracyScores_test, 'r--',label = 'test dataset accuracy')
matplotlib.pyplot.plot(range(1,iter),accuracyScores_train,'b--',label = 'train dataset accuracy')
matplotlib.pyplot.legend()
matplotlib.pyplot.title("Accuracy of neural network prediction over data iterations.")
matplotlib.pyplot.xlabel("epochs")
matplotlib.pyplot.ylabel("accuracy (%)")
matplotlib.pyplot.savefig('S3AccuracyPlot')

#reset var resues here.
accuracyScores = []
#two loops one for min leaf size and one for number of trees.
for leaf_size in [5,50]:
    for ntrees in [10,50,100,1000,5000]:

        #code generates trains and calculates accuracy for all parameters.
        rfc = RandomForestClassifier(iter, min_samples_leaf=leaf_size)
        rfc.fit(x_train, y_train)

        predicted = rfc.predict(x_test)
        accuracyScores.append(accuracy_score(y_test, predicted))

#creates data frame table to show results more clearly from all runs.
table = pandas.DataFrame(data= {'5':accuracyScores[0:5], '50':accuracyScores[5:10]})
table.index=[10,50,100,1000,5000]
print(table)