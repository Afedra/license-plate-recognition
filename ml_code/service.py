import numpy
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from ml_config import MachineLearningConfig, TemplateMatching
from ml_validation import AccuracyValidation
import matplotlib.pyplot as plt

config = MachineLearningConfig()
validate = AccuracyValidation()
templatematching = TemplateMatching()
image_data, target_data = config.read_training_data(config.training_data[1])
training_directory = config.training_data[0]

###############################################
# decission tree
###############################################

tree_classifier = DecisionTreeClassifier()

tree_classifier.fit(image_data, target_data)

###############################################
# for validation and testing purposes
###############################################

validate.split_validation(tree_classifier, image_data, target_data, True)

validate.cross_validation(tree_classifier, 3, image_data,
    target_data)

###############################################
# end of validation and testing
###############################################

config.save_model(tree_classifier, 'Tree')

###############################################
# decission tree ends
###############################################


###############################################
# dim reduction
###############################################
pca = PCA(2)

new_image_data = pca.fit_transform(image_data)

print new_image_data.shape

plt.scatter(new_image_data[:, 0], new_image_data[:, 1])
plt.show()

###############################################
# dim reduction ends
###############################################

###############################################
# gaussian naive bayes
###############################################

gaussian_naive_bayes = GaussianNB()

gaussian_naive_bayes.fit(image_data, target_data)

###############################################
# for validation and testing purposes
###############################################

validate.split_validation(gaussian_naive_bayes, image_data, target_data, True)

validate.cross_validation(gaussian_naive_bayes, 3, image_data,
    target_data)

###############################################
# end of validation and testing
###############################################

config.save_model(gaussian_naive_bayes, 'GaussianNB')

###############################################
# gaussian naive bayes ends
###############################################

###############################################
# k-neighbours
###############################################

neighbor_model = KNeighborsClassifier()

neighbor_model.fit(image_data, target_data)

###############################################
# for validation and testing purposes
###############################################

validate.split_validation(neighbor_model, image_data, target_data, True)

validate.cross_validation(neighbor_model, 5, image_data,
    target_data)

###############################################
# end of validation and testing
###############################################

config.save_model(neighbor_model, 'KNeighbors3')

###############################################
# k-neighbours ends
###############################################

###############################################
# SVM
###############################################

# kernel can be linear, rbf e.t.c
svc_model = SVC(kernel='linear', probability=True)

svc_model.fit(image_data, target_data)

###############################################
# for validation and testing purposes
###############################################

validate.split_validation(svc_model, image_data, target_data, True)

validate.cross_validation(svc_model, 3, image_data,
    target_data)

###############################################
# end of validation and testing
###############################################

config.save_model(svc_model, 'SVC_model')

###############################################
# SVM ends
###############################################

###############################################
# Random Forest
###############################################

rand_forest_classifier = RandomForestClassifier()

rand_forest_classifier.fit(image_data, target_data)

###############################################
# for validation and testing purposes
###############################################

validate.split_validation(rand_forest_classifier, image_data, target_data, True)

validate.cross_validation(rand_forest_classifier, 3, image_data,
    target_data)

###############################################
# end of validation and testing
###############################################

config.save_model(rand_forest_classifier, 'RandomForest')

###############################################
# Random Forest ends
###############################################

models = dict()
sv_model_1 = SVC(kernel='linear', probability=True)
sv_model_2 = SVC(kernel='rbf', probability=True)
n_model_1 = KNeighborsClassifier(n_neighbors=3)
n_model_2 = KNeighborsClassifier(n_neighbors=4)
models['linearsvm'] = sv_model_1 
models['rbfsvm'] = sv_model_2 
models['3-neighbor'] = n_model_1 
models['4-neighbor'] = n_model_2 

img_train, img_test, target_train, target_test = train_test_split(image_data,
    target_data, test_size=0.4, train_size=0.6)
prediction2Dlist = []

for name, model in models.items():
    print name
    print '-------------------------------'
    model.fit(img_train, target_train)
    prediction = model.predict(img_test)
    prediction2Dlist.append(prediction)
    accuracy = (float(numpy.sum(prediction == target_test)) / len(target_test))
    print str(round(accuracy * 100, 2))+ "% accuracy was recorded"
    print '-------------------------------'

for index in range(len(img_test)):
    if prediction2Dlist[2][index] != target_test[index]:
        print 'Based on LinearSVM'
        print 'Actual Label : '+ target_test[index]+' Predicted Label: '+prediction2Dlist[1][index]
        for name in models:
            print name
            print '-------------------------------'
            prob_predictions = models[name].predict_proba(
                img_test[index].reshape(1, -1))
            validate.top_predictions(prob_predictions)
            print '-------------------------------'
            
        if prediction2Dlist[1][index] in config.ascertain_characters:
            print 'Prediction by template matching'
            print '-------------------------------'
            print templatematching.template_match(prediction2Dlist[1][index],
            img_test[index], training_directory)
            print '-------------------------------'

models = dict()
sv_model_1 = SVC(kernel='linear')
sv_model_2 = SVC(kernel='rbf')
sv_model_3 = SVC(kernel='poly')
n_model_1 = KNeighborsClassifier(n_neighbors=3)
n_model_2 = KNeighborsClassifier(n_neighbors=4)
n_model_3 = KNeighborsClassifier(n_neighbors=5)
gnb = GaussianNB()
dec_tree = DecisionTreeClassifier()
rand_forest = RandomForestClassifier()
models['linearsvm'] = sv_model_1 
models['rbfsvm'] = sv_model_2 
models['poly'] = sv_model_3
models['3-neighbor'] = n_model_1 
models['4-neighbor'] = n_model_2 
models['5-neighbor'] = n_model_3 
models['Decision Tree'] = dec_tree 
models['Gaussian Naive Bayes'] = gnb 
models['Random Forest'] = rand_forest 

image_data, target_data = config.read_training_data(training_directory)  

for name, model in models.items():
    print name
    print '-------------------------------'
    start = time.time()
    scores = cross_val_score(model, image_data, target_data, cv=4)
    timediff = time.time() - start
    print 'Accuracy = '+ str(numpy.mean(scores)*100) + 'Time = '+str(timediff)
    print '-------------------------------'