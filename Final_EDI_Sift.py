#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Required Libraries
import numpy as np
import pandas as pd
import os
import csv
import cv2
import time

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix , accuracy_score, precision_score, recall_score, f1_score, roc_curve ,roc_auc_score,ConfusionMatrixDisplay
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import servo as sr

ser1 = sr.Servo(3)    # Index Finger
ser1.start(50)
ser2 = sr.Servo(11)   # Middle Finger
ser2.start(50)
ser3 = sr.Servo(12)   # Ring Finger
ser3.start(50)
ser4 = sr.Servo(13)   # Lady's Finger
ser4.start(50)
ser5 = sr.Servo(15)   # Thumb rotate motor
ser5.start(50)
ser6 = sr.Servo(18)   # Arm Motor
ser6.start(50)


def releaseHand():
	ser1.goto(0)
	ser2.goto(0)
	ser3.goto(0)
	ser4.goto(0)
	ser5.goto(90)
	ser6.goto(0)

releaseHand()


def Hold_Bottle():
	ser1.slide(0, 120, 0.02)
	ser2.slide(0, 140, 0.02)
	ser3.slide(0,120,0.02)
	ser4.slide(0,120,0.02)
	ser5.slide(90,0,0.07)
	ser6.slide(0,135, 0.02)


def Hold_Cup():
	ser1.slide(0,100,0.02)
	ser2.slide(0,120,0.02)
	ser3.slide(0,100,0.02)
	ser4.slide(0,100,0.02)
	ser5.slide(90,0,0.07)
    ser6.slide(0, 135, 0.02)


# In[2]:


WaterBottle = 'C:/Users/Arjun/Downloads/ARJUN EDI/water bottle 2'
Cup= 'C:/Users/Arjun/Downloads/ARJUN EDI/coffee_cup'



# In[3]:


i=0
for filename in os.listdir(WaterBottle):
    img = cv2.imread(os.path.join(WaterBottle, filename))
    if img is not None:
        resize=(128,128)
        img=cv2.resize(img,resize)
        #grayscaling the image dataset
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.GaussianBlur(gray, (5, 5), 0)  # gaussian Image

        # creating a Histograms Equalization for folder1(Fall)
        equ = cv2.equalizeHist(img2)

        

       
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) #kernels for prewitt edge detection
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(equ, -1, kernelx)#Horizontal 
        img_prewitty = cv2.filter2D(equ, -1, kernely)#Vertical
        img_prewitt = img_prewittx + img_prewitty

        #Applying Sift  Discriptor Abnormal folder
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_prewitt, None)

        #convert the descriptor array into a dataframe format
        out = pd.DataFrame(des)
        print("descriptor shape ", i, " : ", out.shape)
        i = i + 1

        csv_data = out.to_csv('WaterBottleSIFT.csv', mode='a', header=False, index=False)


# In[4]:


#@title Default title text
i=0
for filename in os.listdir(Cup):
    img = cv2.imread(os.path.join(Cup, filename))
    if img is not None:
        resize=(128,128)
        img=cv2.resize(img,resize)
        #grayscaling the image dataset
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.GaussianBlur(gray, (5, 5), 0)  # gaussian Image

      
        equ = cv2.equalizeHist(img2)


        
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) #kernels for prewitt edge detection
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(equ, -1, kernelx)#Horizontal 
        img_prewitty = cv2.filter2D(equ, -1, kernely)#Vertical
        img_prewitt = img_prewittx + img_prewitty


        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_prewitt, None)

        #convert the descriptor array into a dataframe format
        out = pd.DataFrame(des)
        print("descriptor shape ", i, " : ", out.shape)
        i = i + 1

        csv_data = out.to_csv('CupSIFT.csv', mode='a', header=False, index=False)


# In[5]:


data1 = pd.read_csv('WaterBottleSIFT.csv', dtype='uint8',header=None)
data2 = pd.read_csv('CupSIFT.csv', dtype='uint8',header=None)


# In[6]:


data1


# In[7]:


data2


# In[8]:


#append all the class wise feature descriptor data into one data frame
data=data1.append(data2)

data


# In[9]:


data


# In[10]:


#save appended data into a csv file
csv_data=data.to_csv('finalDataSIFT1.csv', mode='a', header=False,index=False)


# In[11]:


#read the data from the previously saved csv file
data = pd.read_csv('finalDataSIFT1.csv',header=None)
data


# In[12]:


##    K meanse Clustring
#Applying Kmeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(data)


# In[13]:


import pickle

# Assuming you have already applied K-means and obtained the 'kmeans' model

# Specify the file path where you want to save the model
filepath = 'kmeans_modelSIFT.pickle'

# Save the model using pickle
with open(filepath, 'wb') as file:
    pickle.dump(kmeans, file)


# In[14]:


#save the model to disk
import pickle
filename = 'Kmeans_CL_2_ModelSIFT2.sav'
pickle.dump(kmeans, open(filename, 'wb'))


# In[15]:


#calculate histogram of trained kmeans
hist = np.histogram(kmeans.labels_,bins=[0,1,2,3])

print("Histogram of trained kmeans")
print(hist, "\n")


# In[16]:


#performing kmeans prediction on the WaterBottle with the pretrained kmeans model

#initialising i=0; as it is the first class
i=0
data=[]
#k=0

for filename in os.listdir(WaterBottle):
    #path
    path=os.path.join(WaterBottle,filename)
    a=cv2.imread(path)

    #gray image
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    # creating a Histograms Equalization for folder1(Fall)
    equ = cv2.equalizeHist(gray)

    #Applyeda
  
    #Applying Sift  Discriptor Abnormal folder
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) #kernels for prewitt edge detection
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(equ, -1, kernelx)#Horizontal 
    img_prewitty = cv2.filter2D(equ, -1, kernely)#Vertical
    img_prewitt = img_prewittx + img_prewitty    
    
    kp, descriptors = sift.detectAndCompute(gray, None)

    out=pd.DataFrame(descriptors)

    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3])

    #append the dataframe into the array
    data.append(hist[0])


#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class
Output["Class"] = i
csv_data=Output.to_csv('finalWaterBottle12.csv', mode='a',header=False,index=False)


# In[17]:


#performing kmeans prediction on the CoffeeCup with the pretrained kmeans model

#initialising i=0; as it is the first class
i=1
data=[]
#k=0

for filename in os.listdir(Cup):
    #path
    path=os.path.join(Cup,filename)
    a=cv2.imread(path)

    #gray image
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  
    
    kp, descriptors = sift.detectAndCompute(gray, None)

    out=pd.DataFrame(descriptors)

    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3])

    #append the dataframe into the array
    data.append(hist[0])


#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class
Output["Class"] = i
csv_data=Output.to_csv('finalCup12.csv', mode='a',header=False,index=False)


# In[18]:


#Displaying the kmeans predicted data of folder1
print("WaterBottle")
dat1= pd.read_csv('finalWaterBottle12.csv',header=None)
print(dat1)


# In[19]:


#Displaying the kmeans predicted data of folder1
print("Coffee_Cup")
dat2= pd.read_csv('finalCup12.csv',header=None)
print(dat2)


# In[20]:


#appending All kmeans predicted data into 1 dataframe
B = dat1.append(dat2)
B


# In[21]:


df=B


# In[22]:


df


# In[23]:


df.head()


# In[24]:


df.tail()


# In[25]:


#save the predicted data into csv file
csv_data=df.to_csv('FinalFSIFT.csv', mode='a',header=False,index=False)


# In[26]:


#read the data from the previously saved csv file
df = pd.read_csv("FinalFSIFT.csv",header=None)
df


# In[27]:


#Check for NaN under a single DataFrame column
df.isnull().values.any()


# In[28]:


# statistical measures about the data
df.describe()


# In[29]:


X = df.drop(columns=3, axis=1)
Y = df[3]


# In[30]:


X


# In[31]:


Y


# In[32]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                                                 test_size=0.3, random_state = 0)
# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", Y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", Y_test.shape)


# In[33]:


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix , accuracy_score, precision_score, recall_score, f1_score, roc_curve ,roc_auc_score,plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[34]:


#Importing required libraries

import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[35]:


#Loading the dataset
df = pd.read_csv("FinalFSIFT.csv",header=None) 
#df = data.frame
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[36]:


X


# In[37]:


y


# In[38]:


k = 5
kf = KFold(n_splits=k, random_state=None)
model = RandomForestClassifier()
 
acc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


# In[39]:


k = 5
kf = KFold(n_splits=k, random_state=None)
model=KNeighborsClassifier()
 
acc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


# In[40]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                                                 test_size=0.3, random_state = 0)
# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", Y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", Y_test.shape)


# In[41]:


modelRF = RandomForestClassifier()
modelRF.fit(X_train,Y_train)
# accuracy on training data
X_train_prediction = modelRF.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Applying Random Forest")

print('Accuracy on Train data : ', train_data_accuracy)

print('Presion :',precision_score(X_train_prediction,Y_train))
print('Recall :',recall_score(X_train_prediction,Y_train))

print('F1 score : ', f1_score(X_train_prediction,Y_train))

# accuracy on test data
X_test_prediction = modelRF.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

print('Precision :', precision_score(Y_test, X_test_prediction, average='weighted'))

print('Recall :', recall_score(Y_test, X_test_prediction, average='weighted'))

print('F1 score : ', f1_score(Y_test, X_test_prediction, average='weighted'))


# get the confusion matrix
conf_mat = confusion_matrix(Y_test, X_test_prediction)
print("Confusion matrix:\n", conf_mat)





# In[42]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute predicted probabilities for positive class
y_prob = modelRF.predict_proba(X_test)[:, 1]

# Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[43]:


import pickle

# Assuming you have already trained the 'modelRF' Random Forest model

# Specify the file path where you want to save the model
filepath = 'random_forest_model.pickle'

# Save the model using pickle
with open(filepath, 'wb') as file:
    pickle.dump(modelRF, file)


# In[44]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

modelknn = KNeighborsClassifier()
modelknn.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = modelknn.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('\nResults obtained for the knn')
print('\nResults obtained on Training Data')
print('Accuracy on Train data : ', train_data_accuracy)

print('Presion :', precision_score(Y_train, X_train_prediction, average='weighted'))
print('Recall :', recall_score(Y_train, X_train_prediction, average='weighted'))
print('F1 score : ', f1_score(Y_train, X_train_prediction, average='weighted'))

# Accuracy on test data
X_test_prediction = modelknn.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('\nResults obtained on Testing Data')
print('Accuracy on Test data : ', test_data_accuracy)

print('Presion :', precision_score(Y_test, X_test_prediction, average='weighted'))
print('Recall :', recall_score(Y_test, X_test_prediction, average='weighted'))
print('F1 score : ', f1_score(Y_test, X_test_prediction, average='weighted'))
# get the confusion matrix
conf_mat = confusion_matrix(Y_test, X_test_prediction)
print("Confusion matrix:\n", conf_mat)


# In[45]:


import pickle

# Assuming you have already trained the 'modelRF' Random Forest model

# Specify the file path where you want to save the model
filepath = 'K_N_N_model.pickle'

# Save the model using pickle
with open(filepath, 'wb') as file:
    pickle.dump(modelknn, file)


# In[46]:


modelDT = DecisionTreeClassifier(max_depth=5)
modelDT.fit(X_train,Y_train)



# accuracy on training data
X_train_prediction = modelDT.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)

print('Accuracy on Train data : ', train_data_accuracy)

print('Presion :',precision_score(X_train_prediction,Y_train,average='weighted'))

print('Recall :',recall_score(X_train_prediction,Y_train,average='weighted'))

print('F1 score : ', f1_score(X_train_prediction,Y_train,average='weighted'))

# accuracy on test data
X_test_prediction = modelRF.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

print('Presion :',precision_score(X_test_prediction,Y_test,average='weighted'))
print('Recall :',recall_score(X_test_prediction,Y_test,average='weighted'))

print('F1 score : ', f1_score(X_test_prediction,Y_test,average='weighted'))


# In[48]:


modelSVM = SVC()
modelSVM.fit(X_train,Y_train)

# accuracy on training data
X_train_prediction = modelSVM.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)

print('Accuracy on Train data : ', train_data_accuracy)

print('Presion :',precision_score(X_train_prediction,Y_train,average='weighted'))

print('Recall :',recall_score(X_train_prediction,Y_train,average='weighted'))

print('F1 score : ', f1_score(X_train_prediction,Y_train,average='weighted'))

# accuracy on test data
X_test_prediction = modelSVM.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

print('Presion :',precision_score(X_test_prediction,Y_test,average='weighted'))
print('Recall :',recall_score(X_test_prediction,Y_test,average='weighted'))

print('F1 score : ', f1_score(X_test_prediction,Y_test,average='weighted'))

# confusion matrix
cm = confusion_matrix(Y_test, X_test_prediction)
print('Confusion Matrix : ')
print(cm)


# In[49]:


import pickle

# Assuming you have already trained the 'modelRF' Random Forest model

# Specify the file path where you want to save the model
filepath = 'SVM_model.pickle'

# Save the model using pickle
with open(filepath, 'wb') as file:
    pickle.dump(modelSVM, file)


# In[50]:


from sklearn.naive_bayes import GaussianNB

modelNB = GaussianNB()
modelNB.fit(X_train,Y_train)

# accuracy on training data
X_train_prediction = modelNB.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)

print('Accuracy on Train data : ', train_data_accuracy)

print('Presion :',precision_score(X_train_prediction,Y_train,average='weighted'))

print('Recall :',recall_score(X_train_prediction,Y_train,average='weighted'))

print('F1 score : ', f1_score(X_train_prediction,Y_train,average='weighted'))

# accuracy on test data
X_test_prediction = modelNB.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

print('Presion :',precision_score(X_test_prediction,Y_test,average='weighted'))
print('Recall :',recall_score(X_test_prediction,Y_test,average='weighted'))

print('F1 score : ', f1_score(X_test_prediction,Y_test,average='weighted'))

# confusion matrix
cm = confusion_matrix(Y_test, X_test_prediction)
print('Confusion Matrix : ')
print(cm)


# In[51]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[52]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# fit the random forest classifier
modelRF = RandomForestClassifier()
modelRF.fit(X_train,Y_train)

# get the predicted probabilities for the test data
Y_test_proba = modelRF.predict_proba(X_test)



# In[53]:


# Define the bin edges manually
bin_edges = [0, 1, 2, 3]

# Load the image
img = cv2.imread("C:/Users/Arjun/Downloads/realbottle.jpg")
resize=(128,128)
rzimg=cv2.resize(img,resize)
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract SIFT keypoints and descriptors from the grayscale image
kp, descriptors = sift.detectAndCompute(gray, None)

# Convert descriptors to float64
descriptors = np.array(descriptors, dtype=np.float64)

# Predict the cluster labels for the descriptors using KMeans
labels = kmeans.predict(descriptors)

# Compute the histogram of the predicted cluster labels
hist, _ = np.histogram(labels, bins=bin_edges)

# Reshape the histogram to a shape of (1, n_bins)
hist = hist.reshape((1, -1))

# Use the trained model to predict the class of the image using the histogram as input
predicted_class = modelRF.predict(hist)

print('Predicted class:', predicted_class)


# In[54]:


# Define the bin edges manually
bin_edges = [0, 1, 2, 3]

# Load the image
img = cv2.imread("C:/Users/Arjun/Downloads/coffe 4.jpg")
resize=(128,128)
rzimg=cv2.resize(img,resize)
# Convert the image to grayscale
gray = cv2.cvtColor(rzimg, cv2.COLOR_BGR2GRAY)

# Extract SIFT keypoints and descriptors from the grayscale image
kp, descriptors = sift.detectAndCompute(gray, None)

# Convert descriptors to float64
descriptors = np.array(descriptors, dtype=np.float64)

# Predict the cluster labels for the descriptors using KMeans
labels = kmeans.predict(descriptors)

# Compute the histogram of the predicted cluster labels
hist, _ = np.histogram(labels, bins=bin_edges)

# Reshape the histogram to a shape of (1, n_bins)
hist = hist.reshape((1, -1))

# Use the trained model to predict the class of the image using the histogram as input
predicted_class = modelRF.predict(hist)

print('Predicted class:', predicted_class)


# In[55]:


# Display when both the classes are detected

# Load the Random Forest model from the pickle file
rf_model_filepath = 'random_forest_model.pickle'
with open(rf_model_filepath, 'rb') as file:
    modelRF = pickle.load(file)

# Load the K-means model from the pickle file
kmeans_model_filepath = 'kmeans_modelSIFT.pickle'
with open(kmeans_model_filepath, 'rb') as file:
    kmeans = pickle.load(file)

# Define the bin edges for the histogram (With Class as Water bottle and Cup)
bin_edges = [0, 1, 2, 3]

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract SIFT keypoints and descriptors from the grayscale frame
    kp, descriptors = sift.detectAndCompute(gray, None)

    # Convert descriptors to float64
    descriptors = np.array(descriptors, dtype=np.float64)

    # Predict the cluster labels for the descriptors using KMeans
    labels = kmeans.predict(descriptors)

    # Compute the histogram of the predicted cluster labels
    hist, _ = np.histogram(labels, bins=bin_edges)

    # Reshape the histogram to a shape of (1, n_bins)
    hist = hist.reshape((1, -1))

    # Use the trained Random Forest model to predict the class of the frame using the histogram as input
    predicted_class = modelRF.predict(hist)

    # Display the predicted class on the frame
    if predicted_class == 0:  # Only display when WaterBottle is detected
        cv2.putText(frame, 'WaterBottle', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        print('WaterBottle')
        time.sleep(5)
        Hold_Bottle()
        time.sleep(8)
        releaseHand()

    elif predicted_class == 1:  # Only display when Cup is detected
        cv2.putText(frame, 'Coffee Cup', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        print('Coffee Cup')
        time.sleep(5)
        Hold_Cup()
        time.sleep(8)
        releaseHand()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()


# In[ ]:




