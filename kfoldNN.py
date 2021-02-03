import numpy
import pandas
import matplotlib.pyplot
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
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

#neurel network tests.
neurons = [10,25,50,75,100,250,500,750,1000]#how mnay neurons to use for each pass.
accuracy = []

matplotlib.pyplot.clf()
for z in neurons:

    net = MLPClassifier((z,z),'logistic',max_iter=500);
    tmp = []#tmp array for storing results of each pass.

    print("Neurons : ", z)

    for Train, Test in kf.split(x):

        net.fit(x[Train],y[Train])

        predicted = net.predict(x[Test])
        tmp.append(accuracy_score(y[Test], predicted))

    print("Success!");
    accuracy.append(sum(tmp)/len(tmp))
    for xe in tmp:
        matplotlib.pyplot.scatter(z,xe, c="black", marker=".")


matplotlib.pyplot.plot(neurons,accuracy, 'r--')
matplotlib.pyplot.title("Average accuracy for neural net predictions over nodes per layer.")
matplotlib.pyplot.xlabel("nodes per layer")
matplotlib.pyplot.ylabel("accuracy (%)")
matplotlib.pyplot.savefig('KfoldNN')

print(neurons)
print(accuracy)

