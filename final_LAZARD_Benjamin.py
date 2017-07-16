# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:08:38 2017

@author: Benjamin LAZARD
"""
# %% Importing the required modules
# basic python packages for plotting and array management
import numpy as np

# for data import
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

# Classifier retained
from sklearn.neural_network import MLPClassifier

# Ensemble methods and crossvalidation
from sklearn.ensemble import BaggingClassifier, VotingClassifier

# PostProcessing
from sklearn.metrics import confusion_matrix

# Because oh boy some computations take an amazing amount of time !
import time
# %% Important functions to define

# To compute the personalized score


def compute_pred_score(y_true, y_pred):
    y_pred_unq = np.unique(y_pred)
    for i in y_pred_unq:
        if((i != -1) & (i != 1) & (i != 0)):
            raise ValueError('The predictions can contain only -1, 1, or 0!')
    y_comp = y_true * y_pred
    score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score


def makeTimeSignificant(t_seconds):
    # transforms seconds into hours, minutes, and seconds
    m, s = divmod(t_seconds, 60)
    h, m = divmod(m, 60)
    return "%dh%02dm%02ds" % (h, m, s)


def predict_0_labels(XX, clf, threshold=0.7, without=False):
    # Add the 0 labels to a prediction to increase score
    # check whether the classifier is compatible
    can_predict_proba = getattr(clf, "predict_proba", None)
    if callable(can_predict_proba):
        print("0_labels enabled")
        start = time.time()
        prediction_set = clf.predict_proba(XX)  # For each index of XX, the proba that it belongs to class -1 and then 1
        print("total timed used for predicting: %s" % (makeTimeSignificant(time.time() - start)))

        # Adaptating the classifier
        y_pred_with0 = np.zeros(XX.shape[0])  # just to initialize the size
        y_pred_with0[prediction_set[:, 0] > threshold] = -1
        y_pred_with0[prediction_set[:, 1] > threshold] = 1
        # The other values are already set to 0

        if(without):
            y_pred = np.ones(XX.shape[0])
            y_pred[prediction_set[:,0] >= 0.5] = -1
            return y_pred_with0, y_pred
        else:
            return y_pred_with0
    else:
        print("0_labels disabled")
        start = time.time()
        y_pred = clf.predict(XX)
        print("total timed used for predicting: %s"%(makeTimeSignificant(time.time() - start)))
        if(without):
            return y_pred, y_pred
        else:
            return y_pred


def prepare_dataset(XX_train, y_train, XX_test, var_ratio_min=99.9, ratio_sd=100):
    # Scale it
    myScaler = StandardScaler()
    XX_train_scaled = myScaler.fit_transform(XX_train)

    # Select the most significant features
    pca_scaled = PCA(svd_solver='full', whiten=True, n_components=var_ratio_min/100).fit(XX_train_scaled)
    XX_pca_scaled = pca_scaled.transform(XX_train_scaled)
    print("%d features selected out of %d (%d %%) for PCA which explains %d %% of variance" % (pca_scaled.n_components_, XX_train.shape[1], pca_scaled.n_components_/XX_train.shape[1]*100, pca_scaled.explained_variance_ratio_.sum()*100))

    # print("\n explained variance ratio as a 'per thousand' ratio for each of the selected features")
    # print((pca_scaled.explained_variance_ratio_*1000).round())

    # Select a certain amount of observations
    n_sd = XX_train.shape[0]*ratio_sd/100  # effective number of observations retained
    print("%d observations selected out of %d (%d %%) for Shuffling and training" % (n_sd, XX_train.shape[0], ratio_sd))

    #S huffle it
    XX_train_scaled_shuffled, yy_train_scaled_shuffled = shuffle(XX_pca_scaled, y_train, n_samples=n_sd)

    # Adapt the test set accordingly
    XX_test_scaled = myScaler.transform(XX_test)
    XX_test_scaled_pca = pca_scaled.transform(XX_test_scaled)

    return XX_train_scaled_shuffled, yy_train_scaled_shuffled, XX_test_scaled_pca


def save_prediction(X_test, clf, trial_number, threshold=0.7):
    y_pred = predict_0_labels(X_test, clf, threshold=threshold)
    np.savetxt('y_pred_' + str(trial_number) + '.txt', y_pred, fmt='%d')

# %% data import
X_train_fname = 'training_templates.csv'
y_train_fname = 'training_labels.txt'
X_test_fname  = 'testing_templates.csv'
X_train = pd.read_csv(X_train_fname, sep=',', header=None).values
X_test  = pd.read_csv(X_test_fname,  sep=',', header=None).values
y_train = np.loadtxt(y_train_fname, dtype=np.int)

print("We will train our algorithm based on a set of %d pictures, each with %d features." % (X_train.shape[0], X_train.shape[1]))
print("Then we will test it on a set of %d pictures with the same number of features." % (X_test.shape[0]))
print("\nThe training set consists of labels: ")
print(np.unique(y_train))
print("for exemple '-1' = women and '1' = men")
print("There are exactly %d men and %d women" % ((y_train == -1).sum(),(y_train == 1).sum() ))

# %% Classification Best attempt until 22/04/2017
# Customizing the number of features and observations
X_train_adapt, y_train_adapt, X_test_adapt = prepare_dataset(X_train,
                                                             y_train,
                                                             X_test,
                                                             var_ratio_min=99.9,
                                                             ratio_sd=100)
# base estimator
clf_bag = MLPClassifier(batch_size=400,
                        activation='relu',
                        solver='adam',
                        alpha=0.316228)
# intermediate classifiers
clf1 = BaggingClassifier(base_estimator=clf_bag, max_samples=0.3,
                         n_estimators=40, n_jobs=4, verbose=5, oob_score=True)
clf2 = BaggingClassifier(base_estimator=clf_bag, n_estimators=40,
                         n_jobs=4, verbose=5, max_samples=0.2, oob_score=True)
clf3 = BaggingClassifier(base_estimator=clf_bag, n_estimators=40,
                         n_jobs=4, verbose=5, max_samples=0.7, oob_score=True)
clf4 = BaggingClassifier(base_estimator=clf_bag, n_estimators=50,
                         n_jobs=4, verbose=5, max_samples=1.0, oob_score=True)

# Final classifier
clf = VotingClassifier(estimators=[('clf1', clf1),
                                   ('clf2', clf2),
                                   ('clf3', clf3),
                                   ('clf4', clf4)],
                       n_jobs=4,
                       voting='soft')

# Fitting (takes roughly half of an hour)
start = time.time()
clf.fit(X_train_adapt, y_train_adapt)
print("total time used for fitting: %s" % (makeTimeSignificant(time.time() - start)))

# Predicting on the training set
y_pred_train = predict_0_labels(XX=X_train_adapt, clf=clf, threshold=0.82)

# Score on the training set
score = compute_pred_score(y_train_adapt, y_pred_train)
print("Score with bagging + MPLClassifier + Voting %0.3f" % (score))
print("\n\nConfusion matrix")
print(confusion_matrix(y_train_adapt, y_pred_train))

# Saving results for the test set
print("\nNow for the test set")
save_prediction(X_test=X_test_adapt, clf=clf, trial_number=1, threshold=0.82)

# %% Classification Best attempt after 22/04/2017
#Customizing the number of features and observations
X_train_adapt, y_train_adapt, X_test_adapt = prepare_dataset(X_train, y_train, X_test, var_ratio_min=99.9, ratio_sd=100)

#base estimator
clf_bag = MLPClassifier(batch_size='auto', activation='relu', solver='adam', alpha=0.1, tol=0.0007)

#bagging
clf = BaggingClassifier(base_estimator=clf_bag,
                        n_estimators=40,
                        max_samples=0.35,
                        max_features= 0.62,
                        n_jobs=4,
                        verbose=5)

#Fitting on the training set
start = time.time()
clf.fit(X_train_adapt, y_train_adapt)
print("total time used for fitting: %s"%(makeTimeSignificant(time.time() - start)))

#Predicting on the training set
y_pred_train = predict_0_labels(XX=X_train_adapt, clf=clf, threshold=0.73)

#Score on the training set
score = compute_pred_score(y_train_adapt, y_pred_train)
print("Score with bagging + MPLClassifier estimator %0.3f"%(score))
print("\n\nConfusion matrix")
print(confusion_matrix(y_train_adapt, y_pred_train))

#Saving results for the test set
print("\nNow for the test set")
save_prediction(X_test=X_test_adapt, clf=clf, trial_number=2, threshold=0.73)
