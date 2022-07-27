# Sentiment-Hate-System

## Semi-Supervised Learning
### Pseudo-labeling
To train a machine learning model using supervised learning, the data must be labeled. Does this rule out the use of unlabeled data for supervised tasks like regression and classification? Certainly not! In addition to using the extra data for analysis, we can use it to train our model with semi-supervised learning, which combines labeled and unlabeled data.

The main idea is to keep things simple. After training the model on labeled data, generate pseudo-labels by using the trained model to predict labels on unlabeled data. Additionally, by combining the labeled data and the recently pseudo-labeled data, create a new dataset that will be used to train the data.

![pseudo-labeling](https://user-images.githubusercontent.com/63624329/181389042-704b8cf1-af3c-42cd-b125-8477c1721525.png)
