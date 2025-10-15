import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import streamlit as st



model=pickle.load(open('lr.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))



st.header('Sentiment Analysis Web App')
text=st.text_area('write your review',
                  max_chars=1000,height='content')

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







