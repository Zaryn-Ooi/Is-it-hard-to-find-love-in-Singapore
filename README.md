# Is-it-hard-to-find-love-in-Singapore

Runtime disconnected
NLP_youtube_comment.ipynb
NLP_youtube_comment.ipynb_
Is it hard to find love in Singapore ?
Project Objective :
Despite having more opportunities to meet people, the age at which Singaporeans are getting hitched continues to rise.

As of 2020, the median age of first-time grooms in Singapore was 30.4 years old and 28.8 years old for first-time brides. While this had remained relatively constant for men since 2011, the median age of marriage for women had been increasing, indicating a trend amongst Singaporean women in delaying marriage.

According to an interview done by TODAY, singles aged between 20 and 35 said that they were dating actively and going on dates as frequently as once every two weeks. However, these dates usually fell short of their expectations and did not progress to any long-term relationship.

So, the main question of this project is to determine: Is it hard to find love in Singapore ?

Project Overview :
In this project, we will explore multiple machine learning models like Naive Bayes and Support Vector Machine Classifier (SVM), and vectorization strategies to identify the one that performs the best.

Dataset Description :
The dataset consists of 211 data extracted from the comments of a youtube video.

There are 3 columns in this dataset:

Name: the name of the youtube user who commented the video
Comments: the text of the comment
Likes: the number of likes of the comment
Data Source: Is it hard to find love in Singapore? | Mothership Hits The Streets

Video URL: https://www.youtube.com/watch?v=4Bfec3zlujU

Extract the comment of the youtube video as our datasource
[ ]
function scrapeCommentsWithReplies(){
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var result=[['Name','Comment','Time','Likes','Reply Count','Reply Author','Reply','Published','Updated']];
  var vid = SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1AyTz2n2WlArAQ6yBuXJC8__TV1GWJFE6-LJJ9SAo7ZI/edit?usp=sharing").getRange('A1').getValue();
  var nextPageToken=undefined;
  
  while(1){
   
      var data = YouTube.CommentThreads.list('snippet', {videoId: vid, maxResults: 100, pageToken: nextPageToken})
      nextPageToken=data.nextPageToken
      for (var row=0; row<data.items.length; row++) {
            result.push([data.items[row].snippet.topLevelComment.snippet.authorDisplayName,
                 data.items[row].snippet.topLevelComment.snippet.textDisplay,
                 data.items[row].snippet.topLevelComment.snippet.publishedAt,
                 data.items[row].snippet.topLevelComment.snippet.likeCount,
                 data.items[row].snippet.totalReplyCount,'','','','']);
        if(data.items[row].snippet.totalReplyCount>0){
          parent=data.items[row].snippet.topLevelComment.id
          var nextPageTokenRep=undefined
          while(1){
            var data2=YouTube.Comments.list('snippet', {videoId: vid, maxResults: 100, pageToken: nextPageTokenRep,parentId:parent})
            nextPageTokenRep=data2.nextPageToken;
            for (var i =data2.items.length-1;i>=0;i--){
              result.push(['','','','','',
                       data2.items[i].snippet.authorDisplayName,
                       data2.items[i].snippet.textDisplay,
                       data2.items[i].snippet.publishedAt,
                       data2.items[i].snippet.updatedAt]);
            }
            if(nextPageTokenRep =="" || typeof nextPageTokenRep === "undefined"){
              break
            }
          } 
        }
      }   
    if(nextPageToken =="" || typeof nextPageToken === "undefined"){
      break;
    }
}

var newSheet=SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1AyTz2n2WlArAQ6yBuXJC8__TV1GWJFE6-LJJ9SAo7ZI/edit?usp=sharing").insertSheet(SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1AyTz2n2WlArAQ6yBuXJC8__TV1GWJFE6-LJJ9SAo7ZI/edit?usp=sharing").getNumSheets())
newSheet.getRange(1, 1,result.length,9).setValues(result)

}

Import Packages and Load Dataset
[54]
3s
# Import packages
!pip install pywaffle
import pandas as pd
import numpy as np
from pywaffle import Waffle
import matplotlib.pyplot as plt
Requirement already satisfied: pywaffle in /usr/local/lib/python3.7/dist-packages (0.6.4)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from pywaffle) (3.2.2)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pywaffle) (2.8.2)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pywaffle) (3.0.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pywaffle) (0.11.0)
Requirement already satisfied: numpy>=1.11 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pywaffle) (1.21.5)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pywaffle) (1.3.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib->pywaffle) (1.15.0)
[56]
9s
# Load data
from google.colab import files
uploaded = files.upload()

[57]
0s
df = pd.read_csv('youtube_comment.csv')
df.head()

[58]
0s
df = df.drop(["Word","Name"], axis=1)
df.head()

[59]
0s
len(df)
210
Data Cleaning
[60]
0s
# Detect and remove NaN values 
df.isnull().sum()
Comment    0
Likes      0
dtype: int64
[61]
0s
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 210 entries, 0 to 209
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   Comment  210 non-null    object
 1   Likes    210 non-null    int64 
dtypes: int64(1), object(1)
memory usage: 3.4+ KB
The comments contained in the dataset seem to contain “impurities” that may mislead our machine learning models. In this case, we will:

Removing hashtags (#) and mentions(@): The word #cool and cool may be interpreted as different words by a computer.

Removing url

Removing multiple whitespaces: There are comments with several whitespaces. We decided to remove those whitespaces since they don’t add any useful information to the meaning of the text.

[62]
0s
# Convert the comment field from object into string
df['Comment']=df['Comment'].apply(str)
[63]
0s
df['Comment'] = df['Comment'].replace(np.nan, '')
[64]
0s
## optional to run
# from google.colab import drive
# drive.mount('/content/drive')
[65]
0s
# Remove user mentions and hashtag
import re

def remove_mhash(row_text):
    processed_text = re.sub(r'@[\w]+|#[\w]+',"",  row_text)
    return processed_text
[66]
0s
df['Comment'] = df['Comment'].apply(remove_mhash)
[67]
0s
# Remove url
def remove_url(row_text):
    processed_text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+","", row_text)
    processed_text = re.sub(r'<a href=',"", processed_text)
    return processed_text 

df['Comment'] = df['Comment'].apply(remove_url)
[68]
0s
# Check for whitespace strings
blanks = []
for i,lb,rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
            
print(len(blanks), 'blanks: ', blanks)

0 blanks:  []
Sentiment Analysis
[69]
0s
import nltk
nltk.download('vader_lexicon')
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
True
[70]
0s
# Import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
Add new words into the vocabulary
The value range from -4 to 4 where -4.0 is the most negative and +4.0 is the most positive.

For example:

Angmoh = the word 'angmoh' refers to foreigners in Singapore English. In this case, most viewers commented that Singaporean women prefer angmoh over local singaporean. For this reason, we will shift the value of 'angmoh' to a negative value.

Money = most of the word 'money' that was mentioned in the comments refers to 'No money, no love'.

[71]
0s
new_words = {
    'angmoh': -3.0,
    'single': -4.0,
    'money': -4.0,
    'love': 0.0, 
    'materialistic': -2.0,
    'hard': -4.0,
    'cheap': -1.0,
    'tough': -2.0,
    'easy': 4.0
    }

[72]
0s
# Create a SIA object
SIA = SentimentIntensityAnalyzer()
SIA.lexicon.update(new_words)
[73]
0s
# Use SIA to append a label to the dataset
df['scores'] = df['Comment'].apply(lambda image_text:SIA.polarity_scores(image_text))

df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

df['label'] = df['compound'].apply(lambda c: 'positive' if c>= 0.5 else 'negative') 

df.head(10)

[74]
0s
# count of each label
df['label'].value_counts()
negative    158
positive     52
Name: label, dtype: int64
[75]
0s
# Display how negative comments looks like
negative = df.loc[df['label'] == 'negative']
negative.head()

[76]
0s
# Display how positive comment looks like
positive = df.loc[df['label'] == 'positive']
positive.head()

[77]
0s
# Display how neutral comment looks like
neutral = df.loc[df['label'] == 'neutral']
neutral.head()

Visualize the result of our sentiment analysis
[78]
0s
# percentage of each label
df['label'].value_counts(normalize = True)
negative    0.752381
positive    0.247619
Name: label, dtype: float64
[79]
2s
# Visualize the percentage of each label with Waffle Chart
data = {'Agree, hard to find love in Singapore': 75, 'Disagree': 25} 
fig = plt.figure(
    FigureClass = Waffle,
    figsize =(12,4),
    dpi = 77,
    rows = 5,
    colors=["#3776ab","#ffa9b8"],
    icons=['heart-broken','heart'],
    values = data,
    #labels = list(wafflechart.label),
    legend={'loc': 'lower left', 'bbox_to_anchor': (-0.02, -0.4), 'ncol': len(data), 'framealpha': 0, 'fontsize':15},
    labels=["{0} ({1}%)".format(n, v) for n, v in data.items()]
)

fig.text(0.017,1.07,'Is it hard to find love in Singapore ?',fontfamily='DejaVu Sans',fontsize=27,fontweight='bold',color='#323232')
fig.text(0.017,0.95,'The result of sentiment analysis shown that 75% of the comments agrees that it is hard to find love in Singapore, \nwhile 25% disagreeing it.',fontfamily='monospace',color='gray',fontsize=12)
plt.show()

Train test split
[80]
0s
y = df['label']
X = df['Comment']
[81]
0s
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

Build pipelines to vectorize the data, then train and fit a model
Now that we have sets to train and test, we'll develop a selection of pipelines, each with a different model.

[82]
0s
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

# SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', SVC()),
])
Feed the training data through the first pipeline
[83]
0s
# Naïve Bayes
text_clf_nb.fit(X_train, y_train)

# SVC
text_clf_lsvc.fit(X_train, y_train)
Pipeline(steps=[('tfidf', TfidfVectorizer()), ('clf', SVC())])
Run predictions and analyze the results
[84]
0s
from sklearn import metrics
# Naïve Bayes
predictions_nb = text_clf_nb.predict(X_test)
nb_accuracy = metrics.accuracy_score(y_test,predictions_nb)
print("Naïve Bayes Accuracy Score: {}".format(nb_accuracy))

# SVC
predictions_svc = text_clf_lsvc.predict(X_test)
SVC_accuracy = metrics.accuracy_score(y_test,predictions_svc)
print("SVC Accuracy Score: {}".format(SVC_accuracy))
Naïve Bayes Accuracy Score: 0.7142857142857143
SVC Accuracy Score: 0.7142857142857143
[85]
0s
nb_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test,predictions_nb), index=['neg','pos'], columns=['neg','pos'])
nb_confusion_matrix

[86]
0s
print(metrics.classification_report(y_test,predictions_nb))
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
[87]
0s
SVC_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test,predictions_svc), index=['neg','pos'], columns=['neg','pos'])
SVC_confusion_matrix

[88]
0s
print(metrics.classification_report(y_test,predictions_svc))
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
We got 71 % accuracy but did you notice that recall and precision for class positive are always 0. This means that the classifier is always classifying everything into a single class i.e class negative.

Therefore, our model needs to have its parameters tuned.

Cross Validation
Here is when the usefulness of GridSearch comes into the picture. We can search for parameters using GridSearch.

Naive Bayes Hyperparameter Tuning using GridSearchCV
[89]
0s
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

[90]
0s
# get parameters of MultinomialNB
MultinomialNB().get_params().keys()
dict_keys(['alpha', 'class_prior', 'fit_prior'])
[91]
0s
# GridSearchCV on MultinomialNB
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X_trainv = vectorizer.transform(X_train)
X_testv = vectorizer.transform(X_test)

# Create the hyperparameter grid
params_grid = {'alpha': np.arange(0, 1, 0.05), 'class_prior':[None], 'fit_prior':[True, False]}

# Setup the GridSearchCV object: gm_cv
nb_model = MultinomialNB()

gm_cv = GridSearchCV(nb_model, params_grid, cv = 5)

# Fit it to the training data
gm_cv.fit(X_trainv, y_train)
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
/usr/local/lib/python3.7/dist-packages/sklearn/naive_bayes.py:557: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  % _ALPHA_MIN
GridSearchCV(cv=5, estimator=MultinomialNB(),
             param_grid={'alpha': array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]),
                         'class_prior': [None], 'fit_prior': [True, False]})
[92]
0s
# Print best parameter after tuning
print(gm_cv.best_params_)
 
# Print how our model looks after hyper-parameter tuning
print(gm_cv.best_estimator_)
{'alpha': 0.4, 'class_prior': None, 'fit_prior': True}
MultinomialNB(alpha=0.4)
[93]
0s
# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_testv) 
print(metrics.accuracy_score(y_test, y_pred))
0.7142857142857143
[94]
0s
nb_cmtuned = pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index=['neg','pos'], columns=['neg','pos'])
nb_cmtuned

[95]
0s
print(metrics.classification_report(y_test,y_pred))
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
SVM Hyperparameter Tuning using GridSearchCV
[96]
0s
SVC().get_params().keys()
dict_keys(['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])
[97]
0s
kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")
[98]
0s
for i in range(4):
    # Separate data into test and training sets
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_trainv, y_train)# Make prediction
    svc_pred = svclassifier.predict(X_testv)# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(metrics.classification_report(y_test,svc_pred))
Evaluation: Polynomial kernel
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

Evaluation: RBF kernel
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

Evaluation: Sigmoid kernel
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

Evaluation: Linear kernel
              precision    recall  f1-score   support

    negative       0.71      1.00      0.83        30
    positive       0.00      0.00      0.00        12

    accuracy                           0.71        42
   macro avg       0.36      0.50      0.42        42
weighted avg       0.51      0.71      0.60        42

/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
[99]
4s
# Create the hyperparameter grid
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

# Setup the GridSearchCV object: svc_cv
SVC_model = SVC()

grid = GridSearchCV(SVC_model,param_grid,refit=True,verbose=2)

# Fit it to the training data
grid.fit(X_trainv, y_train)
Fitting 5 folds for each of 48 candidates, totalling 240 fits
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s
[CV] END .....................C=0.1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=0.1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=0.1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=0.1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=0.1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ...................C=0.1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=0.1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=0.1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=0.1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=0.1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ..................C=0.1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=0.1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=0.1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=0.1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=0.1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .................C=0.1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=0.1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=0.1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=0.1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=0.1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s
[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s
[CV] END .......................C=1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=1, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END .....................C=1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=1, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ....................C=1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=1, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ...................C=1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=1, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s
[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s
[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s
[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s
[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s
[CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ..................C=10, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=10, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=10, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=10, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=10, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.0s
[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.0s
[CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.0s
[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s
[CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.0s
[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
[CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.0s
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
[CV] END .................C=100, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=100, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=100, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=100, gamma=0.001, kernel=sigmoid; total time=   0.0s
[CV] END .................C=100, gamma=0.001, kernel=sigmoid; total time=   0.0s
GridSearchCV(estimator=SVC(),
             param_grid={'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
                         'kernel': ['rbf', 'poly', 'sigmoid']},
             verbose=2)
[100]
0s
# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
{'C': 10, 'gamma': 1, 'kernel': 'sigmoid'}
SVC(C=10, gamma=1, kernel='sigmoid')
[101]
0s
grid_predictions = grid.predict(X_testv)
print(metrics.accuracy_score(y_test, grid_predictions))
0.8333333333333334
[102]
0s
SVC_cmtuned = pd.DataFrame(metrics.confusion_matrix(y_test,grid_predictions), index=['neg','pos'], columns=['neg','pos'])
SVC_cmtuned

[103]
0s
print(metrics.classification_report(y_test,grid_predictions))
              precision    recall  f1-score   support

    negative       0.81      1.00      0.90        30
    positive       1.00      0.42      0.59        12

    accuracy                           0.83        42
   macro avg       0.91      0.71      0.74        42
weighted avg       0.86      0.83      0.81        42

Conclusion
Key Takeaways:
Throughout this project, we observed that the majority of Singaporeans think that it is hard to find love in Singapore !

Prediction Model:
Based on the cross validation result, Support Vector Machine Classifier (SVM) is our current best model of this project. The model has achieved 83% of accuracy score. The model shows and F1-score of 0.90 on class negative and 0.59 on class positive on the test set.


check
0s
completed at 11:53 AM
refit=True, hint
Could not connect to the reCAPTCHA service. Please check your internet connection and reload to get a reCAPTCHA challenge.

Runtime disconnected
