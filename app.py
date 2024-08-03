import keras
import tensorflow as tf
import streamlit as st
import pycrfsuite
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import re



# Convert features to the format required by pycrfsuite
def convert_features(X):
    return [{k: str(v) for k, v in x.items()} for x in X]

# Load the CRF model
tagger = pycrfsuite.Tagger()
tagger.open('sindhiposmodel.crfsuite')

# Prepare the sentence for tagging
def prepare_sentence_for_tagging(sentence):
    tokens = sentence.split()
    prepared_sentence = []
    
    for i, token in enumerate(tokens):
        token_dict = {'word': token}
        
        if i == 0:
            token_dict['BOS'] = 'True'
        else:
            token_dict['-1:word'] = tokens[i-1]
        
        if i == len(tokens) - 1:
            token_dict['EOS'] = 'True'
        else:
            token_dict['+1:word'] = tokens[i+1]
        
        prepared_sentence.append(token_dict)
    
    return prepared_sentence



def load_models_and_tokenizers():
    encoder_model = load_model('encoder_model.h5')
    decoder_model = load_model('decoder_model.h5')

    with open('input_tokenizer.pickle', 'rb') as handle:
        input_tokenizer = pickle.load(handle)

    with open('target_tokenizer.pickle', 'rb') as handle:
        target_tokenizer = pickle.load(handle)

    with open('config.pickle', 'rb') as handle:
        config = pickle.load(handle)

    return encoder_model, decoder_model, input_tokenizer, target_tokenizer, config


def lemmatize(input_text, encoder_model, decoder_model, input_tokenizer, target_tokenizer, config):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=config['max_input_len'], padding='post')

    # Encode the input
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character
    target_seq[0, 0] = 2  # start token

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word.get(sampled_token_index, '')
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if (sampled_char == '' or len(decoded_sentence) > config['max_target_len']):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def load_stopwords():
    return set(pd.read_csv('Stopwords.csv')['Stopwords'].tolist())

stopwords = load_stopwords()

# Function to process the text
def remove_stopwords(text):
    # words = re.findall(r'\S+', text)
    words = text.split(" ")
    processed_words = [word for word in words if word not in stopwords]
    return ' '.join(processed_words)


encoder_model, decoder_model, input_tokenizer, target_tokenizer, config = load_models_and_tokenizers()
# Test the lemmatizer
# lemma = lemmatize(word, encoder_model, decoder_model, input_tokenizer, target_tokenizer, config)
#   print(f"Word: {word}, Lemma: {lemma}")

# Streamlit app
st.title("Sindhi Language Preprocessing")

sentence = st.text_input("Enter a sentence for Preprocessing:", "")

if st.button("Preprocess Sentence"):
    if sentence:
        prepared = prepare_sentence_for_tagging(sentence)
        tags = tagger.tag(convert_features(prepared))
        lemmas = [lemmatize(word, encoder_model, decoder_model, input_tokenizer, target_tokenizer, config) for word in sentence.split()]
        st.write("Input Sentence:")
        st.write(sentence)
        
        # Create a DataFrame for better presentation
        df = pd.DataFrame({
            'Token': sentence.split(),
            'Tag': tags,
            'Lemma':lemmas
        })

        st.write("Preprocessed Output:")
        st.dataframe(df, use_container_width=True)
        
        # Optional: Add some CSS styling to make it look better
        st.markdown("""
            <style>
                .dataframe thead th {
                    font-weight: bold;
                    color: #4CAF50;
                }
                .dataframe tbody td {
                    font-size: 16px;
                }
                .dataframe {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
            </style>
        """, unsafe_allow_html=True)
    
        processed_text = remove_stopwords(sentence)

        # Display results
        st.subheader("Original Text")
        st.write(sentence)
        st.subheader("Preprocessed Text")
        st.write(processed_text)
        # Highlight stopwords in original text
        highlighted_text = sentence
        for word in stopwords:
            if word in sentence:
                f"<span style='color:#F5EDED;background-color: #6482AD;font-size: 24px'>{word}</span>"
            else:
                f"<span style='color:#F5EDED;font-size: 24px'>{word}</span>"
        #         highlighted_text = highlighted_text.replace(word, f"<span style='color:#F5EDED;background-color: 6482AD;font-size: 24px'>{word}</span>")
        
        # st.markdown(f"<p style='font-size: 24px;'>{highlighted_text}</p>", unsafe_allow_html=True)

        st.subheader("Processed Text")
        st.write(processed_text)

        # Show statistics
        original_word_count = len(sentence.split())
        processed_word_count = len(processed_text.split())
        removed_words = original_word_count - processed_word_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Word Count", original_word_count)
        col2.metric("Processed Word Count", processed_word_count)
        col3.metric("Removed Stopwords", removed_words)
    else:
        st.write("Please enter a sentence.")
