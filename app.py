import keras
import tensorflow as tf
import streamlit as st
import pycrfsuite
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle



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
    else:
        st.write("Please enter a sentence.")
