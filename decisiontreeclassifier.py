import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#tucson: 24, baltimore: 30
data_size = 24
#tucson: 8, baltimore: 10
num_of_subsets = 8
subset_size = 3


def main():

    data = pd.read_csv("TucsonRepresentativeData.csv")
    i = 0
    correct = 0

    #######(10,11) - income
    #######(9,11) - employment
    #######(6,7,8,11) - edu
    #######(1,2,3,4,5,11) - racial
    #(0,11) - pop. density
    #######(9,10,11)- employment and income
    #######(1,2,3,4,5,10,11) - income and racial
    #######(6,7,8,10,11) - income and edu
    #######(1,2,3,4,5,6,7,8,11) edu and racial

    fn = ["Pop. Density"]
    cn = ["0.0", "1.0"]

    all_data = data.values[0:, (0,11)]
    Y_train = []
    X_train = []
    for i in all_data:
        Y_train.append(i[-1])
        X_train.append(i[0:-1])

    clf = DecisionTreeClassifier(criterion = "gini",
        random_state = 100,max_depth=3, min_samples_leaf=5)
    clf = clf.fit(X_train, Y_train)
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(clf,feature_names = fn, class_names = cn, filled =True)
    fig.savefig("TRPopDensity.png") 


    #ten/eight-fold cross validation
    # for i in range(num_of_subsets):
    #     X_train, Y_train, X_test, Y_test = splitdata(i, data)
    #     "how we originally trained data without gini index; accuracy for baltimore:60% tuscon:the same -->clf = tree.DecisionTreeClassifier()" 
    #     clf = DecisionTreeClassifier(criterion = "gini",
    #     random_state = 100,max_depth=3, min_samples_leaf=5)

    #     clf = clf.fit(X_train, Y_train)
    #     predictions = clf.predict(X_test)

    #     #get total number of correct predictions
    #     count = 0
    #     while count < subset_size:
    #         if predictions[count] == Y_test[count]:
    #             correct += 1
    #         count += 1         

    #     i += 1
        
    # print(str(correct) + " out of " + str(data_size) + " correct: " + str((correct/data_size) * 100))

# spliting data into sets of 3 
# currently uses employment rate and median income as input
def splitdata(test_set_num, data):

    subsets = [] 
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    #select columns and rows we want to extract, including input and output
    #(10,11) - income
    #(9,11) - employment
    #(6,7,8,11) - edu
    #(1,2,3,4,5,11) - racial
    #(0,11) - pop. density
    #(9,10,11)- employment and income
    #(1,2,3,4,5,10,11) - income and racial
    #(6,7,8,10,11) - income and edu
    #(1,2,3,4,5,6,7,8,11) edu and racial

    count = 0
    while count < data_size:
        #subsets.append(data.values[count:count+3,(10,11)])
        #subsets.append(data.values[count:count+3,(9,11)])
        #subsets.append(data.values[count:count+3,(6,7,8,11)])
        #subsets.append(data.values[count:count+3,(1,2,3,4,5,11)])
        #subsets.append(data.values[count:count+3,(0,11)])
        ###subsets.append(data.values[count:count+3,(9,10,11)])
        subsets.append(data.values[count:count+3,(1,2,3,4,5,10,11)])
        #subsets.append(data.values[count:count+3,(6,7,8,10,11)])
        #subsets.append(data.values[count:count+3,(1,2,3,4,5,6,7,8,11)])
        count += subset_size

    #print(subsets)

    test = subsets[test_set_num]
    #splits testing data into respective X input, y outcome subsets 
    for i in test:
        X_test.append(i[0:len(i)-1])
        Y_test.append(i[len(i)-1])
    
    #remove testing set from training set 
    del subsets[test_set_num]

    #splits the training data in respective X train, Y train subsets 
    for i in subsets:
        for j in i:
            X_train.append(j[0:len(j)-1])
            Y_train.append(j[len(j)-1])

    #print(X_train)
    #print(Y_train)

    return X_train, Y_train, X_test, Y_test

# Calling main function
if __name__=="__main__":
    main()
