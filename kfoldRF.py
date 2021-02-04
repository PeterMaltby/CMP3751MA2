import numpy
import pandas
import matplotlib.pyplot
import sklearn.preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#cross-validation of classifcation models.
#Peter Maltby - 2021-02-03

#reads data prints exert and produces stastical measure table output to "dataset_stats.xlsx"
data = pandas.read_excel (r'clinical_dataset.xlsx')

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

#creates kfold object that shuffles data and produces 10 splits.
kf = KFold(n_splits=10, shuffle=True, random_state= 1)

#decision forest tests.
trees = [10,25,50,75,100,250,500,750,1000]#how mnay trees to use for each pass.
accuracy = []

matplotlib.pyplot.clf()
for z in trees:

    rfc = RandomForestClassifier(z, min_samples_leaf=5)
    tmp = []#tmp array for storing results of each pass.

    print("trees : ", z)

    for Train, Test in kf.split(x):

        rfc.fit(x[Train],y[Train])

        predicted = rfc.predict(x[Test])
        tmp.append(accuracy_score(y[Test], predicted))

    print("Success!");
    accuracy.append(sum(tmp)/len(tmp))
    for xe in tmp:
        matplotlib.pyplot.scatter(z,xe, c="black", marker=".")

matplotlib.pyplot.plot(trees,accuracy, 'r--')
matplotlib.pyplot.title("Accuracy of random forest classifiers predictions over number of decision trees.")
matplotlib.pyplot.xlabel("number of decision trees.")
matplotlib.pyplot.ylabel("accuracy (%)")
matplotlib.pyplot.savefig('kfoldRFC')

print(trees)
print(accuracy)