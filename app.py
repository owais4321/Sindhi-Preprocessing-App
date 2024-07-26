import streamlit as st
import pycrfsuite
import pandas as pd

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

# Streamlit app
st.title("Sindhi POS Tagging")

sentence = st.text_input("Enter a sentence for tagging:", "")

if st.button("Tag Sentence"):
    if sentence:
        prepared = prepare_sentence_for_tagging(sentence)
        tags = tagger.tag(convert_features(prepared))
        
        st.write("Input Sentence:")
        st.write(sentence)
        
        # Create a DataFrame for better presentation
        df = pd.DataFrame({
            'Token': sentence.split(),
            'Tag': tags
        })

        st.write("Tagged Output:")
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
