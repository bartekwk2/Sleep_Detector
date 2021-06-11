
from speed_calculation import max_speed_for_second,speed_body_parts
from csv_operation import read_csv2,generate_labels,split_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm,metrics
import pickle



def plotTwoSeriesValues(xSleepData,ySleepData,xAwakeData,yAwakeData):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(xAwakeData,yAwakeData, s=10, c='r', marker="o", label='second')
    ax1.scatter(xSleepData, ySleepData, s=10, c='b', marker="s", label='first')
    plt.show()


def classifierTwo(maxSpeedPerSecond,openEyePixelsCount):
    allData = list(map(list, zip(maxSpeedPerSecond, openEyePixelsCount)))
    labels = generate_labels(721)
    X_train, X_test, y_train, y_test = train_test_split(allData,labels, test_size=0.2,)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def classifierAll(allPartsSpeed,openEyePixelsCount):
    LAnkle = allPartsSpeed[0]
    RAnkle = allPartsSpeed[1]
    LWrist = allPartsSpeed[2]
    RWrist = allPartsSpeed[3]

    allData = list(map(list, zip(LAnkle,RAnkle,LWrist,RWrist,openEyePixelsCount)))
    scaler = StandardScaler()
    allData = scaler.fit_transform(allData)
    labels = generate_labels(721)

    X_train, X_test, y_train, y_test = train_test_split(allData,labels, test_size=0.2,)
    clf = svm.SVC(kernel='poly',degree=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return clf


def saveModel(model,filename):
    path = "data/models/"
    pickle.dump(model, open(path+filename, 'wb'))


def loadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


'''
openEyePixelsCount = read_csv2('data/csv/dziecko.csv')[:723]
allPartsSpeed,length = speed_body_parts('film3_points')
svmModel = classifierAll(allPartsSpeed,openEyePixelsCount)
saveModel(svmModel,"svm_classifier")

'''





'''
maxSpeedPerSecond = max_speed_for_second('film3_points')
openEyePixelsCount = read_csv2('data/csv/dziecko.csv')[:723]
classifier(maxSpeedPerSecond,openEyePixelsCount)
'''


'''
#plotValues(maxSpeedPerSecond,openEyePixelsCount)
allData = list(zip(maxSpeedPerSecond,openEyePixelsCount))
labels = generate_labels(723)
awakeData,sleepData=split_data(labels,allData)
xSleepData,ySleepData = zip(*sleepData)
xwakeData,yawakeData = zip(*awakeData)
plotTwoSeriesValues(xSleepData,ySleepData,xwakeData,yawakeData)
'''