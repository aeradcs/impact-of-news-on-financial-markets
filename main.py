import math
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.hdpmodel import HdpModel
from gensim.corpora import Dictionary
import re
import pprint
import plotly.express as px
import gensim.matutils as matutils
from sklearn.decomposition import SparsePCA
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier



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


def classify(filename):
    df_orig = pd.read_csv(filename, sep='\n', header=None)
    df_orig = df_orig.rename(columns={0: 'text'})
    print(df_orig.head())

    df = df_orig.copy()
    df = df[0:100]
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
    # topics = model.print_topics(num_topics=150, num_words=20)
    # key_words = []
    # coefs_in_topic = []
    # for topic in topics:
    #     print(topic[1])
    #     str = topic[1].replace("*", " ")
    #     str = str.replace("+ ", "")
    #     # print(str)
    #     coefs_in_topic.append(re.findall(r'[0][\.][0-9]{3}', topic[1]))
    #     key_words.append(re.findall(r'[a-z]+', str))
    #     # print([float(s) for s in str.split() if s.isdigit()])
    # print(coefs_in_topic)
    # print(key_words)
    # to_delete = []
    # for words, i in zip(key_words, range(0,150)):
    #     for words_, j in zip(key_words, range(0,150)):
    #         # если количество одинаковых ключевых слов (ключевых слов изначально во всех темах 10) в двух темах больше 6, то одну из тем можно убрать
    #         if i != j and len(list(set(words).intersection(words_))) > 2:
    #             print(len(list(set(words).intersection(words_))))
    #             print(i, j)

    # likelihood_df.to_csv('df/likelihood.csv', index=False)

    result_df = pd.DataFrame({'text': df['text'],
                              'cluster HDP': pd.Series(get_topics_for_texts(df['text'], likelihood_df, dictionary)),
                              'target': [1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1]})
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

    p = count_p(result_df)
    print("p")
    print(p)

    clusters = predict_cluster(sparse_matrix.toarray().T, result_df)
    print(f"for cluster {clusters[0]}")
    print(f"ANSWER IS {predict_impact(p, clusters[0])}")

    # df to build colored plot
    reduced_vector_cluster_target = pd.concat([reduced_df, result_df['cluster HDP'], result_df['target']], axis=1)
    # print("reduced_vector_cluster_target")
    # print(reduced_vector_cluster_target.head())
    # reduced_vector_cluster_target.to_csv('df/reduced_vectors_clusters.csv', index=False)

    # colored plot
    fig = px.scatter_3d(reduced_vector_cluster_target, x=0, y=1, z=2, color='cluster HDP')
    fig.update_traces(marker=dict(size=5))
    # fig.write_html('colored_vis.html')
    fig.show()

    # usual plot
    fig_usual = px.scatter_3d(reduced_df, x=0, y=1, z=2)
    fig_usual.update_traces(marker=dict(size=5))
    # fig_usual.write_html('vis.html')
    fig_usual.show()





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


def predict_cluster(matrix, result_df):
    X = matrix[0:80]
    X_test = matrix[80:100]
    y = result_df['cluster HDP'].values[0:80]
    y_true = result_df['cluster HDP'].values[80:100]

    # classifier = SVC()
    classifier = RandomForestClassifier()
    classifier.fit(X, y)

    y_predicted = classifier.predict(X_test)
    print(f"\npredicted {y_predicted} \ntrue {y_true}")
    print(f"matches amount = {len(list(set(y_predicted).intersection(y_true)))}\n\n")

    # print(metrics.confusion_matrix(y_true, y_predicted))
    # print(metrics.classification_report(y_true, y_predicted, digits=3))
    # print("_______________________________________________________________________________________")

    return y_predicted
    # y = np.array(y)
    # classes = np.unique(y)
    # y_true = label_binarize(y_true, classes=classes)
    # # n_classes = y_true.shape[1]
    # y_predicted = label_binarize(y_predicted, classes=np.unique(y))
    # print("y_true", y_true)
    # print("y_pred", y_predicted)
    # print("classes", classes)

    # For each class
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in classes:
    #     precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
    #                                                         y_predicted[:, i])
    #     average_precision[i] = average_precision_score(y_true[:, i], y_predicted[:, i])

    # # A "micro-average": quantifying score on all classes jointly
    # precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
    #                                                                 y_predicted.ravel())
    # average_precision["micro"] = average_precision_score(y_true, y_predicted,
    #                                                      average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))


def predict_impact(p, cluster):
    if p.loc[p['cluster HDP'] == cluster, 'p'].iloc[0] > 0.5:
        return 1
    else:
        return 0



if __name__ == '__main__':
    classify('BloombergScraping.txt')


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
