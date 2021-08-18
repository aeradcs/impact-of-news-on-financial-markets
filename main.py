import math
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.hdpmodel import HdpModel
from gensim.corpora import Dictionary
import re
import plotly.express as px
import gensim.matutils as matutils
from sklearn.decomposition import SparsePCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics



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
    print(df_orig.head())

    df = df_orig.copy()
    df = df[0:20]
    df['text'] = df['text'].apply(lambda text: preprocessing(text))
    print("df")
    print(df.head())

    dictionary = Dictionary(df['text'])
    corpus = [dictionary.doc2bow(text) for text in df['text']]
    print(corpus[0])

    model = HdpModel(corpus, dictionary)
    likelihood_df = pd.DataFrame(model.get_topics())
    print("likelihood_df")
    print(likelihood_df)
    # likelihood_df.to_csv('df/likelihood.csv', index=False)

    result_df = pd.DataFrame({'text': df['text'],
                              'cluster HDP': pd.Series(get_topics_for_texts(df['text'], likelihood_df, dictionary)),
                              'target': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0]})
    print("result_df")
    print(result_df)
    # result_df.to_csv('df/texts_clusters.csv', index=False)

    sparse_matrix = matutils.corpus2csc(corpus)
    print("sparse_matrix")
    print(sparse_matrix)

    pca = SparsePCA(n_components=3)
    reduced = pca.fit_transform(sparse_matrix.toarray().T)
    print("reduced")
    print(reduced)

    # dense = sparse_matrix.toarray().T
    # print("dense")
    # print(dense)
    #
    # dense_df = pd.DataFrame(dense)
    # print("dense_df")
    # print(dense_df)
    # # dense_df.to_csv('df/dense.csv')

    reduced_df = pd.DataFrame(reduced)
    print("reduced_df")
    print(reduced_df)
    # reduced_df.to_csv('df/reduced.csv', index=False)

    # df to build colored plot
    reduced_vector_cluster_target = pd.concat([reduced_df, result_df['cluster HDP'], result_df['target']], axis=1)
    print("reduced_vector_cluster_target")
    print(reduced_vector_cluster_target.head())
    # reduced_vector_cluster_target.to_csv('df/reduced_vectors_clusters.csv', index=False)

    # # colored plot
    # fig = px.scatter_3d(reduced_vector_cluster_target, x=0, y=1, z=2, color='cluster HDP')
    # fig.update_traces(marker=dict(size=5))
    # # fig.write_html('colored_vis.html')
    # fig.show()
    #
    # # usual plot
    # fig_usual = px.scatter_3d(reduced_df, x=0, y=1, z=2)
    # fig_usual.update_traces(marker=dict(size=5))
    # # fig_usual.write_html('vis.html')
    # fig_usual.show()

    p = count_p(result_df)
    print("p")
    print(p)

    svc_predict_cluster(sparse_matrix.toarray().T, result_df)



def count_p(result_df):
    count_clusters = pd.DataFrame(result_df.groupby(['cluster HDP']).size().sort_values(ascending=False)).reset_index().\
        rename(columns={0: 'count all'})
    count_target_1 = pd.DataFrame(result_df.loc[result_df['target'] == 1].groupby(['cluster HDP']).size().sort_values(ascending=False)).reset_index().\
        rename(columns={0: 'count target 1'})
    # print("count")
    # print(count_clusters)
    # print("target")
    # print(count_target_1)
    final = count_clusters.merge(count_target_1, how='outer', on='cluster HDP').fillna(0)
    print("final")
    print(final)

    return pd.concat([final['cluster HDP'],
                           final.apply(lambda row: row['count target 1'] / row['count all'], axis=1)], axis=1).rename(columns={0: 'p'})


def svc_predict_cluster(matrix, result_df):
    X = matrix[0:15]
    X_test = matrix[15:20]
    y = result_df['cluster HDP'].values[0:15]
    y_test = result_df['cluster HDP'].values[15:20]

    svc = SVC()
    svc.fit(X, y)

    predicted = svc.predict(X_test)
    print(f"predicted {predicted} ?= test {y_test}")


    # average_precision_svc = average_precision_score(y_test, predicted)
    # print(f'average precision score: {average_precision_svc}')


if __name__ == '__main__':
    HDP('BloombergScraping.txt')


    # df = pd.read_csv('df/reduced_vectors_clusters.csv')
    # print(df)
    # fig = px.scatter_3d(df, x='0', y='1', z='2', color='cluster HDP')
    # fig.update_traces(marker=dict(size=5))
    # # fig.write_html('colored_vis.html')
    # fig.show()
    #
    # df_ = pd.read_csv('df/reduced.csv')
    # print(df_)
    # fig_ = px.scatter_3d(df_, x='0', y='1', z='2')
    # fig_.update_traces(marker=dict(size=5))
    # # fig_.write_html('vis.html')
    # fig_.show()
