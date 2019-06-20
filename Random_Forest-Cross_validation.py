

import os
import skimage
from skimage import io
import scipy
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil
from sklearn.ensemble import RandomForestClassifier


def load_images(data_dir,Class):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    persons = [p for p in os.listdir(os.path.join(data_dir,Class)) 
                   if os.path.isdir(os.path.join(data_dir,Class,p))]
    labels = []
    images = []
    for p in persons:
        label_dir = os.path.join(data_dir,Class, p)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if (f.endswith(".jpg")) | (f.endswith(".JPG"))]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(Class)
    print()
    return images, labels

def prepare_data(images,labels,test_images,test_labels):
    X = np.array(images)
    y = np.array(labels)

    Xt = np.array(test_images)
    yt = np.array(test_labels)

    X_train = X
    y_train = y
    X_test = Xt
    y_test = yt

    nsamples, nx, ny = X_train.shape
    X_train_2d = X_train.reshape(nsamples,nx*ny)

    nsamples, nx, ny = X_test.shape
    X_test_2d = X_test.reshape(nsamples,nx*ny)
    return X_train_2d,y_train,X_test_2d,y_test

image_path="/home/amohamed/new_arabic_sign_language"
test_path = "/home/amohamed/test_asl"
classes=['ain','al', 'aleff','bb']
Random_forest_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=7, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

score= []
for i in range (1,32):
    images_train=[]
    labels_train=[]
    images_test=[]
    labels_test=[]
    for Class in classes:
        #move 1 person t be a valid test
        shutil.move(os.path.join("/", "home","amohamed" ,"new_arabic_sign_language", Class,f"{Class}_person_{i}")
                    ,os.path.join("/", "home","amohamed", "test_asl" ,Class))
        #load the rest of the people to be train test   
        images, labels = load_images(image_path,Class)
        for image in images:
            images_train.append(image)
        for label in labels:
            labels_train.append(label)
        #load the test set
        test_images, test_labels = load_images(test_path,Class)
        for image in test_images:
            images_test.append(image)
        for test_label in test_labels:
            labels_test.append(test_label)
        #bring back the person in test class to the origin directory
        shutil.move(os.path.join("/", "home","amohamed" , "test_asl" ,Class,f"{Class}_person_{i}"),
                    os.path.join("/", "home","amohamed"  ,"new_arabic_sign_language", Class))
    #prepare the data for training
    X_train,y_train, X_test,y_test = prepare_data(images_train,labels_train,images_test,labels_test)
    #train the data
    Random_forest_clf.fit(X_train, y_train)
    #get the score of the test set
    y_pred = Random_forest_clf.predict(X_test)
    score_result= accuracy_score(y_test, y_pred)
    print(score_result)
    score.append(score_result)
    
print(score)
result=0
for num in score:
    result=result+num
print(result/len(score))



