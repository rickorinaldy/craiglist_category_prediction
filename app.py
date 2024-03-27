import re
import nltk
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download('wordnet')


model = tf.keras.models.load_model('model/nn_model.h5')

sections = ["community", "for-sale", "housing", "services"]

cities = [
    'bangalore',
    'chicago',
    'delhi',
    'dubai.en',
    'frankfurt.en',
    'geneva.en',
    'hyderabad',
    'kolkata.en',
    'london',
    'manchester',
    'mumbai',
    'newyork',
    'paris.en',
    'seattle',
    'singapore',
    'zurich.en'
]

with open('encoder/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('encoder/section_label_encoder.pkl', 'rb') as f:
    section_label_encoder = pickle.load(f)

with open('encoder/city_label_encoder.pkl', 'rb') as f:
    city_label_encoder = pickle.load(f)

with open('encoder/label_encoder_category.pkl', 'rb') as f:
    category_label_encoder = pickle.load(f)

with open('encoder/section_onehot_encoder.pkl', 'rb') as f:
    section_onehot_encoder = pickle.load(f)

with open('encoder/city_onehot_encoder.pkl', 'rb') as f:
    city_onehot_encoder = pickle.load(f)

with open("encoder/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def text_preprocessing(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = (text.lower()).split()
    text = [w for w in text if not w in set(stopwords.words('english'))]

    lem = WordNetLemmatizer()
    text_list = [lem.lemmatize(w) for w in text if len(w) > 1]

    return " ".join(text_list)

def one_hot_encoding(label_encoder, onehot_encoder, feature):
    integer_encoded = label_encoder.transform(np.array([feature]))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    return onehot_encoded

def preprocessing(input_list: list) -> np.array:
    section = np.array([input_list[0]])
    city = np.array([input_list[1]])
    heading = np.array([text_preprocessing(input_list[2])])

    section = one_hot_encoding(section_label_encoder, section_onehot_encoder, section).toarray()
    city = one_hot_encoding(city_label_encoder, city_onehot_encoder, city).toarray()
    heading = vectorizer.transform(heading).toarray()

    X = np.concatenate((section, city, heading), axis=1)

    return X

def main():
    st.title('Craiglist - Category Prediction App')

    city = st.selectbox('City', options=cities)
    section = st.selectbox('Section', options=sections)
    heading = st.text_input('Post Heading')

    if st.button('Predict'):
        preprocessed_data = preprocessing([section, city, heading])
        prediction = model.predict(preprocessed_data)
        prediction = category_label_encoder.inverse_transform(np.argmax(prediction, axis=1))
        st.subheader(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    main()
