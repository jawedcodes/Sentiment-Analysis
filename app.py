import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import streamlit as st
import re
import string
import nltk


lemmatizer=nltk.WordNetLemmatizer()

# Ensure necessary NLTK resources are available
# for resource in ['stopwords', 'punkt', 'wordnet']:
#     try:
#         nltk.data.find(f'corpora/{resource}')
#     except LookupError:
#         nltk.download(resource)
# Download resources (only first time)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))




# print(sklearn.__version__)





model=pickle.load(open('lr.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 5. Tokenize
    tokens = nltk.word_tokenize(text)
    # 6. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # 7. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


st.header('Sentiment Analysis Web App')
text=st.text_area('write your review',
                  max_chars=1000,height='content')

text=preprocess_text(text)
vector=tfidf.transform([text])

if st.button('Analys'):
    predict=model.predict(vector)
    if predict[0]==0:
        st.success('🤷‍♂️Irrelevent')
    elif predict[0]==1:
        st.success('😒Negative')
    elif predict[0]==2:
        st.success('😊Neutral')
    else:
        st.success('✌️Positive')





