import math

import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.hdpmodel import HdpModel
from gensim.corpora import Dictionary
import re
import plotly.express as px
import gensim.matutils as matutils
from sklearn.decomposition import SparsePCA


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

def get_topics_for_texts(texts, likelihood_df, dictionary):
    topic_num_sum_log_p = pd.DataFrame({'topic_num': [], 'sum_log_p': []})
    topic_nums_list = []
    sum_log_p = 0
    for text in texts:
        for topic in range(0, 149):
            for word in text:
                id = dictionary.token2id[word]
                p = likelihood_df[id][topic]
                sum_log_p = sum_log_p + math.log2(p)
            topic_num_sum_log_p = topic_num_sum_log_p.append({'topic_num': topic, 'sum_log_p': sum_log_p}, ignore_index=True)
            sum_log_p = 0
        max = topic_num_sum_log_p['sum_log_p'].max()
        topic_num = topic_num_sum_log_p['sum_log_p'].loc[lambda v: v == max].index[0]
        topic_nums_list.append(topic_num)
        topic_num_sum_log_p = pd.DataFrame({'topic_num': [], 'sum_log_p': []})
    return topic_nums_list


def HDP(filename):
    df_orig = pd.read_csv(filename, sep='\n', header=None)
    df_orig = df_orig.rename(columns={0: 'text'})

    df = df_orig.copy()
    # df = df[0:10]
    df['text'] = df['text'].apply(lambda text: preprocessing(text))
    print("df")
    print(df)

    dictionary = Dictionary(df['text'])
    corpus = [dictionary.doc2bow(text) for text in df['text']]

    model = HdpModel(corpus, dictionary)
    likelihood_df = pd.DataFrame(model.get_topics())
    print("likelihood_df")
    print(likelihood_df)
    likelihood_df.to_csv('df/likelihood_df.csv', index=False)

    # result_df = pd.DataFrame({'text': df['text'],
    #                           'cluster HDP': pd.Series(get_topics_for_texts(df['text'], likelihood_df, dictionary))})
    # print("result_df")
    # print(result_df)
    # result_df.to_csv('df/texts_clusters_df.csv', index=False)

    sparse_matrix = matutils.corpus2csc(corpus)
    print("sparse_matrix")
    print(sparse_matrix)

    dense = sparse_matrix.toarray().T
    print("dense")
    print(dense)

    dense_df = pd.DataFrame(dense)
    print("dense_df")
    print(dense_df)
    dense_df.to_csv('df/dense_df.csv')

    pca = SparsePCA(n_components=3, random_state=0)
    reduced = pca.fit_transform(dense)
    print("reduced")
    print(reduced)

    reduced_df = pd.DataFrame(reduced)
    print("reduced_df")
    print(reduced_df)
    reduced_df.to_csv('df/reduced_df.csv', index=False)

    fig = px.scatter_3d(reduced, x=0, y=1, z=2)
    fig.show()





if __name__ == '__main__':
    # HDP('BloombergScraping.txt')

    df = pd.read_csv('df/reduced_df.csv')
    fig = px.scatter_3d(df, x='0', y='1', z='2')
    fig.show()