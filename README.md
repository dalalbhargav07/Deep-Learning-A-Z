# Deep-Learning-A-Z
Course undertaken at Udemy by Super DataScience Team, taught by Kirill Eremenko &amp; Hadelin de Ponteves

## Tools used in the course
* Tensorflow
* PyTorch
* Keras
* Theano
* Scikit-Learn (For evaluating our modela and data preprocessing) <br />

## Real World Case Studies
### [1. Churn Modelling Problem](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%201%20-%20Supervised%20Deep%20Learning/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/ANN.ipynb)
In this part solved a data analytics challenge for a bank. A dataset with a large sample of the bank's customers was shared. To make this dataset, the bank gathered information such as customer id, credit score, gender, age, tenure, balance, if the customer is active, has a credit card, etc. During a period of 6 months, the bank observed if these customers left or stayed in the bank. <br />

The goal here was to make an Artificial Neural Network (ANN) that can predict, based on geo-demographical and transactional information given above, if any individual customer will leave the bank or stay (customer churn). ANN was built using tools such as:
* Keras
* Tensorflow 
* Scikit-Learn

### [2. Image Recognition](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%201%20-%20Supervised%20Deep%20Learning/Part%202%20-%20Convolutional%20Neural%20Networks%20(CNN)/cnn.py)

In this part, created a Convolutional Neural Network (CNN) that is able to detect various objects in images. The Deep Learning model was implemented to recognize a cat or a dog in a set of pictures. However, this model can be reused to detect anything else - by simply changing the pictures in the input folder. <br />

For example, you will be able to train the same model on a set of brain images, to detect if they contain a tumor or not. But if you want to keep it fitted to cats and dogs, then you will literally be able to a take a picture of your cat or your dog, and your model will predict which pet you have.It was tested as well on some random image downloaded from google.

CNN was build using tools:
* Keras
* Tensorflow
* Scikit-Learn

### [3. Stock Price Prediction](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%201%20-%20Supervised%20Deep%20Learning/Part%203%20-%20Recurrent%20Neural%20Networks%20(RNN)/Recurrent%20Neural%20Network%20(RNN).ipynb)
In this part, created one of the most powerful Deep Learning models.The Deep Learning model was created as closest to “Artificial Intelligence”. Why is that? Because this model will have long-term memory, just like us, humans. <br />

The branch of Deep Learning which facilitates this is Recurrent Neural Networks (RNNs). Classic RNNs have short memory, and were neither popular nor powerful for this exact reason. But a recent major improvement in Recurrent Neural Networks gave rise to the popularity of LSTMs (Long Short Term Memory RNNs) which has completely changed the playing field. <br />

In this part have implemented this ultra-powerful model, and  take on the challenge to use it to predict the real Google stock price. A similar challenge has already been faced by researchers at Stanford University and tried to do at least as good as them.

Tool used to built RNN are:
* Keras
* Tensorflow
* Scikit-Learn

### 4. Fraud Detection
The business challenge here is about detecting fraud in credit card applications. You will be creating a Deep Learning model for a bank and you are given a dataset that contains information on customers applying for an advanced credit card.

#### [4.1 Fraud Detection using Self Organizing Maps (SOM)](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%202%20-%20Unsupervised%20Deep%20Learning/Part%204%20-%20Self_Organizing_Maps/Fraud%20Detection.ipynb)
This is the an Unsupervised Deep Learning Models. The data comprises of information that customers provided when filling the application form. The task is hese is to detect potential fraud within these applications. That means that by the end of this challenge, it will come up with explicit list of customers who potentially cheated on their applications.

Tool used to built SOM are:
* Keras
* Tensorflow
* Scikit-Learn

#### [4.2 Fraud Detection using Hybrid Model](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Mega_Case_Study/Mega%20Case%20Study.ipynb)

Here the hybrid model was built using Artificial Neural Network (ANN) and Self Organization Maps (SOM) deep learning models. The dataset here used for this case study was the credit card application information data. The intutiton behind the case study is to identify the fraud using SOM and then the idea is to develop advance deep learning which will predic the probabilities of each customer cheated.

Tool used to built Hybrid Model are:
* Keras
* Tensorflow
* Scikit-Learn

### 5. Recommendation System

In this project, created a specific recommendation system using Restricted Boltzman Machine (RBM) & Auto Encoder algorithm. The data set used in this project was downloaded from the movie lens site. You can download the dataset from here.

The goal here was to mimic the recommendation systesm of Netflix. Our dataset has similar features as the Netflix dataset: plent of movies. thousands of users, who have rated the movie.

#### [5.1 Movie Recommendation System using RBM](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%202%20-%20Unsupervised%20Deep%20Learning/Part%205%20-%20Boltzmann%20Machines%20(BM)/Binary%20Movie%20Recommendation%20System%20using%20RBM.ipynb)

Tool used to built Recommendation System are:
* PyTorch
* Pandas

#### [5.2 Movie Recommendation System using Auto Encoder](https://github.com/dalalbhargav07/Deep-Learning-A-Z/blob/master/Volume%202%20-%20Unsupervised%20Deep%20Learning/Part%206%20-%20AutoEncoders%20(AE)/Recommendation%20System%20-%20Auto%20Encoder.ipynb)

Tool used to built Recommendation System are:
* PyTorch
* Pandas

### What I have learnt from this course?
* Understand the intuition behind Artificial Neural Networks
* Apply Artificial Neural Networks in practice
* Understand the intuition behind Convolutional Neural Networks
* Apply Convolutional Neural Networks in practice
* Understand the intuition behind Recurrent Neural Networks
* Apply Recurrent Neural Networks in practice
* Understand the intuition behind Self-Organizing Maps
* Apply Self-Organizing Maps in practice
* Understand the intuition behind Boltzmann Machines
* Apply Boltzmann Machines in practice
* Understand the intuition behind AutoEncoders
* Apply AutoEncoders in practice
