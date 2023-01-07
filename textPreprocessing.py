import nltk
from nltk.stem.porter import PorterStemmer
import re

# Remove HTML Tag 
from bs4 import BeautifulSoup
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# Removing URL
def remove_url(text):
    pattern=re.compile(r'https ? ://\s+|www\.\s+')
    return pattern.sub(r'',text)

# Expanding Contractions 
from contractions import CONTRACTION_MAP # make sure copy contraction.py in the same folder as this file
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    try:
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except:
        return text
    return expanded_text

# Remove Special Characters and Digit
def remove_special_characters_digit(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

# Remove Stopword
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Word Stemmer: Porter Stemmer
ps=PorterStemmer()
def stem(text):
    tokenization = nltk.word_tokenize(text)
    text_list = [ps.stem(w) for w in tokenization]
    return " ".join(text_list)

def text_preprocessing(df):
    df["review"] = df["review"].apply(remove_special_characters_digit)
    df["review"] = df['review'].apply(strip_html_tags)
    df['review'] = df['review'].apply(remove_url)
    df["review"] = df["review"].apply(expand_contractions)
    df["review"] = df["review"].apply(remove_stopwords)
    df["review"] = df["review"].apply(stem)
    return df