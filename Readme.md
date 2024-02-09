# ChangeLog

## Changes
Can find notebook at the following link:<br> https://colab.research.google.com/drive/1-z2FgsH9BkNQTK1XsxBYCmLFVL2eCqEp?usp=drive_link

### Project Goals
<p> The goal of the notebook being followed is to classify newsgroup text into a number of categories representing the newsgroup from which the text was retrieved. A sample of these categories.</p>
<br>

![Categories](Images/categoryImage.png)

<br>
<g> The goal of this project has been changed to the binary classification of a string of text taken from a number of SMS messages into either Spam or not Spam classes. This project will attempt to use Naive Bayes machine learning techniques to make these classifications.

### Data Source
<p> The first change from the source notebook is that the notebook retrieves the data it uses for its predictions from included datasets within the sklearn platform. This project instead sources its data from kaggle using the SMS Spam Collection Dataset from the following address: (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) .
</p>
<p> This dataset contains a string of text representing a SMS and a label either 'ham' to label non-spam messages or 'spam' to label spam messages. It contains 5,574 messages in total</p> 

### Data Preprocessing
<p> As the data source is changed, some changes to the preprocessing are also required for this project.</p>

#### Checking for null values
<p> Some checks are made to ensure that the data is clean and there are no null values in the dataset. This check showed that there are none </p>

```
df[df.isnull().any(axis=1)].count()
```

#### Dropping unneeded columns
<p> The dataset included three commas at the end of each row which created 3 unneeded columns once parsed. This columns were dropped. </p>

```
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
```
### Model Creation

#### Feature Preparation
<p> This project prepares the data for modelling in a different way given the data source and the fact this is a binary classfication algorithm with a single feature variable. The correct variables are placed in X and y arrays and reshape is used to flatten both arrays.</p>

#### Target Variable Encoding
<p> This project uses a label encoder to encode categorical target variable. O for ham and 1 for spam </p>

```
from sklearn.preprocessing import LabelEncoder
labels = ['ham','spam']
lab_encoder = LabelEncoder()
y = lab_encoder.fit_transform(y)
```

### Model Evaluation

#### Classification Report
<p> Here this project does additional metrics to evaluate the model by preparing a classification report </p> 

#### Evaluation Function
<p> This project creates an evaluation function which is designed to test the variance of different models across different subsets of the data. This function takes as parameters, the X and y arrays, the number of iterations that should be run, the training/test split size, the text encoding function, the model types and a boolean that states whether random random_states should be used or not </p>

```
def evaluate_variance(X, y, num_iterations, test_size, textEncoder, modelType, random_state=True):
    accuracy_scores = []

    if not random_state:
        num_iterations = 10
        
    for i in range(num_iterations):

        random_states = [12345, 54321, 32145, 43125, 23145, 0, 10, 20, 30, 40, 50]

        if random_state:
            random_states = np.random.randint(1, 100000, size=num_iterations)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_states[i])
        
        # Create and fit the model
        model = make_pipeline(textEncoder, modelType)
        model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Run ", i, " with random state: ", random_states[i], ": ",accuracy)
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy and variance
    average_accuracy = np.mean(accuracy_scores)
    variance_accuracy = np.var(accuracy_scores)
        
    return average_accuracy, variance_accuracy
```
<p> Model showed itself to have lower variance across different splits of the training data</p>

![alt text](Images/split.png)

#### Feature Engineering
<p> The next difference here is that some basic feature engineering is conducted to try and increase model accuracy. </p>

1. Ignore Case: All words in X are made lowercase. This had no effect on accuracy of algorithm.
2. Removing Punctuation: All punctuation is removed from X. This lowered accuracy from 0.9539 to 0.9451. This implies punctuation is useful in the detection of Spam messages. In particular removing punctuation led to more false positives for spam messages.

![alt text](Images/punc.png)

#### Model Selection
<p> This project also tried to experiment with different encodings and algorithm variations to try and get the highest accuracy possible for predictions.</p>

1. CountVectorizer for text encoding: Switched the text encoding method to a simple CountVectorizer which counts word frequencies like TFid but does not add a weight to word importance. This boosted accuracy from .9539 to .9825 and greatly reduced false positive rates for Spam messages.

![alt text](Images/best.png)

2. CountVectorizer ignoring frequency: Using the binary=True option on CountVectorizer makes all non-zero frequencies = 1. So this only uses a binary consideration of whether a word is within a message or not. With this option, accuracy dropped from .9825 to .9817. This result shows that word frequency, perhaps surprisely, while useful does not have that great an impact on model accuracy.
3. BernoulliDB: As BernoulliNB is designed for binary values, the next test was to switch to this model while keeping CountVectorizer with binary set to True. Even though Bernoulli is designed to work with binary values, it performs worse than Multinomial Naive Bayes with accuracy of .96

<p>Overall the best performance was achieved using a combination of CountVectorizer for text encoding and MultinomialNB </p>

### Deployment
<p> Another change to this project is that a simple webapp was created to demonstrate the deployment of such a model. As such at the end of this Jupyter notebook, pickle was used to dump the finetuned model to a file. This file was then used to build an online webapp using the model to predict if entered text is SPAM. You can find at the following website (http://roadlesswalked.pythonanywhere.com/), please feel free to try it out. </p>

