import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

data = pd.read_csv('testdateimitlabels.csv',low_memory=False)

X = pd.DataFrame(data)
#Delete all boolean type or object type(datum) features
X = X.select_dtypes(exclude=['bool','object'])
#Replace all the NAN with mean
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)
X=pd.DataFrame(X)

########################################################################################################################
#    prediction for label "all"
########################################################################################################################

#get all datas with label "all" is 1
X_all1 = X.loc[X[1] == 1]
#get all datas with label "all" is 0
X_all0 = X.loc[X[1] == 0]

print(X_all1)
X_all1.info()
print(X_all0)
X_all0.info()
#get all data with label "all" is 1 except feature "years","rics" and labels
X_all_1 = X_all1.iloc[:, 5:]
print(X_all_1)
#get all data with label "all" is 0 except feature "years","rics" and labels
X_all_0 = X_all0.iloc[:, 5:]
print(X_all_0)
#set training data
X0_train = X_all_0.loc[0:109196]
print(X0_train)
#set test data
X0_test = X_all_0.loc[109196:]
print(X0_test)
#create a classifier of Isolation Forest and the contamination rate is 0.22
clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_all_1)

# the accuracy for normal data
print("Accuracy of all 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])


# the accuracy for outliers
print("Accuracy of all 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

########################################################################################################################
#     prediction for label "relevant"
########################################################################################################################


#get all datas with label "relevant" is 1
X_rel1 = X.loc[X[2] == 1]
#get all datas with label "all" is 0
X_rel0 = X.loc[X[2] == 0]

print(X_rel1)
X_rel1.info()
print(X_rel0)
X_rel0.info()
#get all data with label "relevant" is 1 except feature "years","rics" and labels
X_rel_1 = X_rel1.iloc[:, 5:]
print(X_rel_1)
#get all data with label "relevant" is 0 except feature "years","rics" and labels
X_rel_0 = X_rel0.iloc[:, 5:]
print(X_rel_0)
#set training data
X0_train = X_rel_0.loc[0:109196]
print(X0_train)
#set test data
X0_test = X_rel_0.loc[109196:]
print(X0_test)
#create a classifier of Isolation Forest and the contamination rate is 0.22
clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_rel_1)

# the accuracy for normal data
print("Accuracy of relevant 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])


# the accuracy for outliers
print("Accuracy of relevant 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

########################################################################################################################
#     prediction for label"relevant5%"
########################################################################################################################

#get all datas with label "relevant5%" is 1
X_5rel1 = X.loc[X[3] == 1]
#get all datas with label "relevant5%" is 0
X_5rel0 = X.loc[X[3] == 0]

print(X_5rel1)
X_5rel1.info()
print(X_5rel0)
X_5rel0.info()
#get all data with label "relevant5%" is 1 except feature "years","rics" and labels
X_5rel_1 = X_5rel1.iloc[:, 5:]
print(X_5rel_1)
#get all data with label "relevant5%" is 0 except feature "years","rics" and labels
X_5rel_0 = X_5rel0.iloc[:, 5:]
print(X_5rel_0)
#set training data
X0_train = X_5rel_0.loc[0:109196]
print(X0_train)
#set test data
X0_test = X_5rel_0.loc[109196:]
print(X0_test)
#create a classifier of Isolation Forest and the contamination rate is 0.22
clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)

y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_5rel_1)

# the accuracy for normal data
print("Accuracy of relevant5% 0 :", list(y_pred_test).count(1)/y_pred_test.shape[0])

# the accuracy for outliers
print("Accuracy of relevant5% 1:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
