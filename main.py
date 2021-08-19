import math
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.hdpmodel import HdpModel
from gensim.corpora import Dictionary
import re
import plotly.express as px
import gensim.matutils as matutils
from sklearn.decomposition import SparsePCA
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
    df_orig = pd.read_csv(filename)
    print(df_orig.head())
    df = df_orig.copy()
    df = df[0:100]
    df['news'] = df['news'].apply(lambda text: preprocessing(text))
    print(df.head())

    dictionary = Dictionary(df['news'])
    corpus = [dictionary.doc2bow(text) for text in df['news']]
    model = HdpModel(corpus, dictionary)
    likelihood_df = pd.DataFrame(model.get_topics())
    print("likelihood_df\n", likelihood_df.head())

    df = pd.concat([df, pd.DataFrame({'cluster HDP': pd.Series(get_topics_for_texts(df['news'], likelihood_df, dictionary))})], axis=1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("df\n", df.head())

    sparse_matrix = matutils.corpus2csc(corpus)
    pca = SparsePCA(n_components=3)
    reduced = pd.DataFrame(pca.fit_transform(sparse_matrix.toarray().T))
    print("reduced\n", reduced[:5])

    p = count_p(df)
    print("p\n", p.head(20))

    clusters = predict_cluster(sparse_matrix.toarray().T, df)
    for cluster in clusters:
        print(f"for cluster {cluster} answer is {predict_impact(p, cluster)}")

    # df to build colored plot
    reduced_vector_cluster = pd.concat([reduced, df['cluster HDP']], axis=1)

    # colored plot
    fig = px.scatter_3d(reduced_vector_cluster, x=0, y=1, z=2, color='cluster HDP')
    fig.update_traces(marker=dict(size=5))
    # fig.write_html('colored_vis.html')
    fig.show()

    # usual plot
    fig_usual = px.scatter_3d(reduced, x=0, y=1, z=2)
    fig_usual.update_traces(marker=dict(size=5))
    # fig_usual.write_html('vis.html')
    fig_usual.show()


def count_p(df):
    count_clusters = pd.DataFrame(df.groupby(['cluster HDP']).size().sort_values(ascending=False)).reset_index().\
        rename(columns={0: 'count all'})
    count_target_1 = pd.DataFrame(df.loc[df['mark'] == 1].groupby(['cluster HDP']).size().sort_values(ascending=False)).reset_index().\
        rename(columns={0: 'count target 1'})
    counted = count_clusters.merge(count_target_1, how='outer', on='cluster HDP').fillna(0)
    print("counted\n", counted.head(20))
    return pd.concat([counted['cluster HDP'],
                           counted.apply(lambda row: row['count target 1'] / row['count all'], axis=1)], axis=1).rename(columns={0: 'p'})


def predict_cluster(matrix, df):
    X = matrix[0:80]
    X_test = matrix[80:100]
    y = df['cluster HDP'].values[0:80]
    y_true = df['cluster HDP'].values[80:100]

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
    classify('new_date_target_df.csv')


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
