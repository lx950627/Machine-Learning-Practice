"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
import sys
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        n,d=X.shape
        c=Counter(y)
        self.probabilities_=dict(c.items())
        for label, count in self.probabilities_.iteritems():
            self.probabilities_[label]=float(count)/n

       # print(self.probabilities_)

        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n,d=X.shape
        label, prob = zip(*self.probabilities_.iteritems())
        y = np.random.choice(list(label),n,p=list(prob))
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error=0
    test_error=0
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_size, test_size=test_size,
                                                            random_state=i+1)
        #print(X_train.shape)
        #print(X_test.shape)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        #print(X_train.shape)
        train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        #print("qqq",metrics.accuracy_score(y_train, y_pred_train, normalize=True))
        y_pred_test = clf.predict(X_test)
        test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)

    train_error=train_error/ntrials
    test_error=test_error/ntrials



        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    #print('Plotting...')
    #for i in range(d) :
      #  plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)  # fit training data using the classifier
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph

    # save the classifier -- requires GraphViz and pydot
    '''
    import StringIO, pydotplus
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    '''


    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    k = 3
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    y_pred = neigh.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- k=%d training error: %.3f' % (k,train_error))

    k = 5
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    y_pred = neigh.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- k=%d training error: %.3f' % (k, train_error))

    k = 7
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    y_pred = neigh.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- k=%d training error: %.3f' % (k, train_error))

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    print("MajorityVoteClassifier")
    clf = MajorityVoteClassifier()
    train_error,test_error = error(clf, X, y, ntrials=100,test_size=0.2)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print("RandomClassifier")
    clf = RandomClassifier()
    train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print("Decision Tree")
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)
    train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print("KNN with k=5")
    k = 5
    neigh = KNeighborsClassifier(n_neighbors=k)
    train_error, test_error = error(neigh, X, y, ntrials=100, test_size=0.2)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    from sklearn.model_selection import cross_val_score
    print('Finding the best k for KNeighbors classifier...')

    val_error=[]
    for k in range(1,50,2):
        neigh = KNeighborsClassifier(n_neighbors=k)
        score=cross_val_score(neigh,X,y,cv=10)
        #print(score)
        val_error.append(1-np.mean(score))
        #print("%d:%f"%(k,1-np.mean(score)))
    #print(val_error)

    plt.plot(range(1,50,2),val_error)
    plt.xticks(np.arange(1, 50, 2))
    plt.title("Validation Error with Different k")
    plt.xlabel("The Number of Neighbors,k")
    plt.ylabel("Validation Error")
    plt.show()

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    from sklearn.model_selection import cross_validate
    print('Investigating depths...')
    train_error = []
    test_error=[]
    for depth in range(1, 21):
        clf = DecisionTreeClassifier(criterion="entropy",max_depth=depth)

        '''
        train_e, test_e=error(clf,X,y)
        train_error.append(train_e)
        test_error.append(test_e)
        '''
        score = cross_validate(clf, X, y, cv=10,return_train_score=True)
        train_error.append(1-np.mean(score['train_score']))
        test_error.append(1-np.mean(score['test_score']))
        
        # print("%d:%f"%(k,1-np.mean(score)))
    #print(train_error)
    #print(test_error)

    fig,ax=plt.subplots()
    ax.plot(np.arange(1, 21), train_error,label="Training Error")
    ax.plot(np.arange(1, 21), test_error,label="Test Error")
    ax.legend(loc=0)
    plt.xticks(np.arange(1, 21))
    plt.title("Training and Test Error with Different Tree Depth Limits")
    plt.xlabel("The Maximum Depth of Decision Tree")
    plt.ylabel("Validation Error Rate")
    plt.show()

    ### ========== TODO : END ========== ###

    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1,random_state=100)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    DTtrainerror= []
    DTtesterror = []

    knn = KNeighborsClassifier(n_neighbors=7)
    kNNtrainerror = []
    kNNtesterror = []
    for i in np.arange(1,10,1):
        fraction=float(i)/10
        print(fraction)
        X_real_train,X_rest,y_real_train, y_rest = train_test_split(X_train, y_train, train_size=fraction)
        print(X_real_train.shape)
        clf.fit(X_real_train, y_real_train)
        y_pred = clf.predict(X_real_train)
        train_error = 1 - metrics.accuracy_score(y_real_train, y_pred, normalize=True)
        print('\t-- DT training error: %.3f' % train_error)
        DTtrainerror.append(train_error)
        y_pred = clf.predict(X_test)
        test_error = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
        print('\t-- DT test error: %.3f' % test_error)
        DTtesterror.append(test_error)

        knn.fit(X_real_train, y_real_train)
        y_pred = knn.predict(X_real_train)
        train_error = 1 - metrics.accuracy_score(y_real_train, y_pred, normalize=True)
        print('\t-- KNN training error: %.3f' % train_error)
        kNNtrainerror.append(train_error)
        y_pred = knn.predict(X_test)
        test_error = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
        print('\t-- KNN test error: %.3f' % test_error)
        kNNtesterror.append(test_error)

    print(1)
    print(X_train.shape)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    train_error = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
    print('\t-- DT training error: %.3f' % train_error)
    DTtrainerror.append(train_error)
    y_pred = clf.predict(X_test)
    test_error = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
    print('\t-- DT test error: %.3f' % test_error)
    DTtesterror.append(test_error)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    train_error = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
    print('\t-- KNN training error: %.3f' % train_error)
    kNNtrainerror.append(train_error)
    y_pred = knn.predict(X_test)
    test_error = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
    print('\t-- KNN test error: %.3f' % test_error)
    kNNtesterror.append(test_error)


    fig, ax = plt.subplots()
    #print(len(DTtrainerror))
    portiton=np.arange(0.1,1.1,0.1)
    print(portiton)
    ax.plot(portiton, DTtrainerror, label="DT Training Error")
    ax.plot(portiton, DTtesterror, label="DT Test Error")
    ax.plot(portiton, kNNtrainerror, label="kNN Training Error")
    ax.plot(portiton, kNNtesterror, label="kNN Test Error")
    ax.legend(loc=0)
    plt.title("Training and Test error against the amount of training data.")
    plt.xlabel("Portiton of Training Dataset")
    plt.ylabel("Error Rate")
    plt.xticks(portiton)
    plt.show()
    ### ========== TODO : END ========== ###

    print('Done')


if __name__ == "__main__":
    main()
