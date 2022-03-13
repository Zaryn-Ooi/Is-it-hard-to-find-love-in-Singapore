# Is it hard to find love in Singapore ? 

![image](https://user-images.githubusercontent.com/86367657/158003302-d3dfc04d-9ee9-427e-96a2-0d4faebc29e4.png)

## Project Objective :
Despite having more opportunities to meet people, the age at which Singaporeans are getting hitched continues to rise.

As of 2020, the median age of first-time grooms in Singapore was 30.4 years old and 28.8 years old for first-time brides. While this had remained relatively constant for men since 2011, the median age of marriage for women had been increasing, indicating a trend amongst Singaporean women in delaying marriage.

According to an interview done by TODAY, singles aged between 20 and 35 said that they were dating actively and going on dates as frequently as once every two weeks. However, these dates usually fell short of their expectations and did not progress to any long-term relationship.

So, the main question of this project is to determine: Is it hard to find love in Singapore ?

## Project Overview :
#### The Training Process :
In the training process, our VADER Sentiment Analysis model learns to associate a particular input (i.e. a comment) to the corresponding output (tag) based on the test samples used for training. The feature extractor transfers the text input into a feature vector. Pairs of feature vectors and tags (e.g. positive or negative) are fed into the machine learning algorithm to generate a model.

#### The Prediction Process :
In the prediction process, we explored vectorization strategies and multiple classification models like Naïve Bayes and Support Vector Machine Classifier (SVM) to identify which model performs the best. In this case, the feature extractor is used to transform unseen text inputs into feature vectors. These feature vectors are then fed into the classification model, which generates predicted tags (eg. positive or negative).

## Dataset Description :
The dataset consists of 210 data extracted from the comments of a youtube video.

There are 3 columns in this dataset:

- Name: the name of the youtube user who commented the video
- Comments: the text of the comment
- Likes: the number of likes of the comment


Data Source: Is it hard to find love in Singapore? | Mothership Hits The Streets

Video URL: https://www.youtube.com/watch?v=4Bfec3zlujU
