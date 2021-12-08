# Importando todas as bibliotecas que serão utilizadas 

import pandas as pd 
import re
import numpy as np

from sklearn.model_selection import train_test_split

from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import WhitespaceTokenizer
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends

# Regex's 
regex_html_01_codes = re.compile(r'<code>(.|\n)*?</code>')
regex_html_02_tags = re.compile(r'<.*?>')
regex_html_03_punctuation = re.compile(r'[^\w\s]')
regex_html_04_numbers = re.compile(r'^\d+')
regex_html_05_linebrokentag = re.compile(r'\n')
regex_html_06_whitespace = re.compile(r' +')

# Random State
SEED = 59
np.random.seed(SEED)

# Modelos

#WhitespaceTokenizer = WhitespaceTokenizer()

# Importando os dataframes que serão utilizados 

df_pt = pd.read_csv('G:\Meu Drive\My Repositories\datasets_for_data_science\databases\datasets\\stackoverflow_portuguese.csv')
df_en = pd.read_csv('G:\Meu Drive\My Repositories\datasets_for_data_science\databases\datasets\\stackoverflow_english.csv')

# Alterando os dados do nosso dataframe

df_pt['Idioma'] = 'port'
df_en['Idioma'] = 'eng'

# Tratando dataframe values

def def_values_tratment(data, column):

    new_senteces = []   

    for c in data[column]:
        sentence_string = c
        sentence_string = regex_html_01_codes.sub(' ', sentence_string)
        sentence_string = regex_html_02_tags.sub(' ', sentence_string)
        sentence_string = regex_html_03_punctuation.sub(' ', sentence_string)
        sentence_string = regex_html_04_numbers.sub(' ', sentence_string)
        sentence_string = regex_html_05_linebrokentag.sub('', sentence_string)
        sentence_string = regex_html_06_whitespace.sub(' ', sentence_string)
        sentence_string = sentence_string.lower()
        new_senteces.append(sentence_string)

    data[column + '_Tratamento'] = new_senteces    

def_values_tratment(df_pt, 'Questão')
def_values_tratment(df_en, 'Questão')

# Criando o modelo MLE para Português e Inglês

def def_modelo_mle_pt_split(x):
    global test_pt, train_pt

    train_pt, test_pt = train_test_split(x, test_size = 0.2)

    train_all_senteces = ' '.join(train_pt)
    train_all_words = WhitespaceTokenizer().tokenize(train_all_senteces)
    train_bigram, vocabulary_bigramns = padded_everygram_pipeline(2, train_all_words)

    modelo = Laplace(2)
    modelo.fit(train_bigram, vocabulary_bigramns)
    
    return modelo

def def_modelo_mle_en_split(x):
    global test_en, train_en

    train_en, test_en = train_test_split(x, test_size = 0.2)

    train_all_senteces = ' '.join(train_en)
    train_all_words = WhitespaceTokenizer().tokenize(train_all_senteces)
    
    train_bigram, vocabulary_bigramns = padded_everygram_pipeline(2, train_all_words)

    modelo = Laplace(2)
    modelo.fit(train_bigram, vocabulary_bigramns)

    return modelo

model_laplace_pt = def_modelo_mle_pt_split(df_pt.Questão_Tratamento)
model_laplace_en = def_modelo_mle_en_split(df_en.Questão_Tratamento)

# Medindo a perplexidade de cada palavra de um determinado data

def perplexity_counter(model, data):
    global words, fake_char, words_bigrams

    perplexities = 0
    words = WhitespaceTokenizer().tokenize(data)
    fake_char = [list(pad_both_ends(c, n = 2)) for c in words]
    words_bigrams = [list(bigrams(k)) for k in fake_char]
    for word in words_bigrams:
        perplexities = model.perplexity(word) + perplexities
    
    return perplexities

# Função que determina o idioma a partir da perplexidade

def language_classifier(data):

    results = []

    if type(data) != str:
        for row in data:
            return_pt = perplexity_counter(model_laplace_pt, row)
            return_en = perplexity_counter(model_laplace_en, row)

            if return_pt < return_en:
                results.append('Português')
            
            else:
                results.append('English')
        
    else:
        return_pt = perplexity_counter(model_laplace_pt, data)
        return_en = perplexity_counter(model_laplace_en, data)

        if return_pt < return_en:
            results.append('Português')
            
        else:
            results.append('English')   
        
        data = [data]
    
    df_results = pd.DataFrame(columns = ['Text', 'Predict'], data = list(zip(data, results)))

    return print(df_results)