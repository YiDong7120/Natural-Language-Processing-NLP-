import pandas as pd
import matplotlib.pyplot as plt
from textPreprocessing import text_preprocessing
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle

## Load and Display Data ###
df=pd.read_csv("IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
print(df.head())

# Checking for missing values
df.isnull().sum()

# checking the distribution of Target Varibale
df['sentiment'].value_counts()

### Text Preprosessing ###
df = text_preprocessing(df)

"""
### Visualize Word Cloud ###
# Visualisation of Positive Sentiments
postive_list=list(df[df['sentiment'] == 1]["review"]) 
positive_text =  " ".join(postive_list)
wordcloud_positive = WordCloud(width = 512,height = 512, background_color="white").generate(positive_text)
plt.figure( figsize=(10, 8))
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Visualisation of Negative Sentiments
negative_list=list(df[df['sentiment'] == 0]["review"])
negative_text =  " ".join(negative_list)
wordcloud_negative = WordCloud(width = 512,height = 512, background_color="white").generate(negative_text)
plt.figure( figsize=(10, 8))
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
"""

### Save Processed Data ###
df.to_csv("Processed Data.csv")

### TF-ITF ###
review = df["review"]
sentiment = df['sentiment']

tfd=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
vector=tfd.fit_transform(review)
vector=vector.toarray()

### Divide data into train and test data ###
X_train, X_test, Y_train, Y_test = train_test_split(vector, sentiment, test_size=0.2, random_state=2)
print('Training dataset shape:', X_train.shape, Y_train.shape)
print('Testing dataset shape:', X_test.shape, Y_test.shape)

### Model Training ###
model_lr=LogisticRegression()
model_lr.fit(X_train,Y_train)

### Performance Measurement ###
predictions_lr=model_lr.predict(X_test)
print("Classification Report")
print(classification_report(Y_test,predictions_lr))
matrix=confusion_matrix(Y_test,predictions_lr)
print('Confusion Matrix : \n', matrix)

### Saving Model Using Pickle ###
pickle.dump(model_lr, open('model_lr.pkl','wb'))
pickle.dump(tfd, open('tf_idf_transformer.pkl','wb'))