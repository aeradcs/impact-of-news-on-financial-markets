from nltk.corpus import stopwords as nltk_stopwords
import re
import pandas as pd


def preprocessing(text):
    stops = nltk_stopwords.words('english')

    text = text.lower()

    # remove emails
    text = re.sub('\S*@\S*\s?', ' ', text)

    # remove numbers and dates
    text = re.sub('\$?[0-9]+[\.]?[0-9]*s?%?\$?\s?', ' ', text)

    # remove hastags
    text = re.sub('#\S*\s?', ' ', text)

    # remove https
    text = re.sub('https://\S*\s?', ' ', text)

    # remove http
    text = re.sub('http://\S*\s?', ' ', text)

    for x in [",", ":", "!", "?", ";", "[", "]",
              "(", ")", "\"", "\'", ".", "\"",
              "#", "@", "&", "`", "'", "’", "-",
              "+", "=", "_", "<", ">", "\\",
              "|", "}", "{", "/", "—", "$", "“", "”"]:
        text = text.replace(x, "")
    text = text.split()
    cleaned_text = []
    for word in text:
        if not (word in stops):
            cleaned_text.append(word)
    text = cleaned_text
    return text


