import numpy as np
import pandas as pd
from gensim.models.hdpmodel import HdpModel
from gensim.corpora import Dictionary
import plotly.express as px
import gensim.matutils as matutils
from sklearn.decomposition import SparsePCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import utils
import math
import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def transform_corpus_to_corpus_dict(corpus):
    corpus_list_of_dicts = []
    corpus_dict = {}
    for text in corpus:
        for pair in text:
            corpus_dict[pair[0]] = pair[1]
        corpus_list_of_dicts.append(corpus_dict)
        corpus_dict = {}
    return corpus_list_of_dicts


def classify(filename, news_source_name):
    # preprocessing
    df_orig = pd.read_csv(filename)
    print("df_orig\n", df_orig)
    df = df_orig.copy()
    df['news'] = df['news'].apply(lambda text: utils.preprocessing(text))
    print("df after preprocessing\n", df)

    # get likelihood by hdp
    dictionary = Dictionary(df['news'])
    corpus = [dictionary.doc2bow(text) for text in df['news']]
    corpus_dict = transform_corpus_to_corpus_dict(corpus)
    model = HdpModel(corpus, dictionary, T=10, random_state=1)
    likelihood_df = pd.DataFrame(model.get_topics())
    print("likelihood_df\n", likelihood_df.head(50))
    likelihood_df.to_csv(f'df/{news_source_name}_likelihood_df.csv', index=False)
    # print("likelihood_df sum of all columns\n", likelihood_df.sum(axis=1))

    # get cluster for every new
    df = pd.concat([df, pd.DataFrame({'cluster HDP': pd.Series(get_topics_for_texts(df['news'], likelihood_df,
                                                                                    dictionary, corpus_dict))})],
                   axis=1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("df after get_topics_for_texts\n", df)
    pd.DataFrame(pd.concat([df['mark'], df['date'], df_orig['news'], df['cluster HDP']], axis=1)) \
        .to_csv(f'df/{news_source_name}_mark_new_cluster_df.csv', index=False)

    # reduce dims to build plot
    sparse_matrix = matutils.corpus2csc(corpus)
    dense = sparse_matrix.toarray().T
    print("dense\n", dense)
    dense_df = pd.DataFrame(dense)
    dense_df.to_csv(f'df/{news_source_name}_dense_df.csv', index=False)
    # pca = SparsePCA(n_components=3)
    # reduced = pd.DataFrame(pca.fit_transform(dense))
    # print("reduced\n", reduced)
    # reduced.to_csv(f'df/{news_source_name}_reduced.csv', index=False)

    # p = amount of news with target 1 in cluster / amount of news in cluster
    p_list = count_p(df)
    print("p_list\n", p_list)

    # predict cluster using random forest
    clusters, news = predict_cluster_random_forest(dense, df, df_orig)

    # predict impact - get 0/1 for new by cluster and p of cluster
    predicted_true_impacts_df = pd.DataFrame({'predicted': [], 'true': []})
    for cluster, new in zip(clusters, news):
        predicted_impact = predict_impact(p_list, cluster)
        predicted_true_impacts_df = predicted_true_impacts_df.append({'predicted': predicted_impact,
                                                                      'true': df_orig.loc[
                                                                          df_orig['news'] == new, 'mark'].iloc[0]},
                                                                     ignore_index=True)
        # if predicted_impact == 1:
        #     print(f"for cluster {cluster} answer is {predict_impact(p_list, cluster)} new {new}")
    compare_predicting_impact = pd.DataFrame(
        np.where(predicted_true_impacts_df['predicted'] == predicted_true_impacts_df['true'],
                 True, False))
    true_count = compare_predicting_impact.pivot_table(columns=[0], aggfunc='size')[True]
    accuracy_predicting_impact = true_count / len(compare_predicting_impact[0])
    print("-----------accuracy_predicting_impact--------------\n",
          true_count, len(compare_predicting_impact[0]), accuracy_predicting_impact)

    # # df to build colored plot
    # reduced_vector_cluster = pd.concat([reduced, df['cluster HDP']], axis=1)
    #
    # # colored plot
    # fig = px.scatter_3d(reduced_vector_cluster, x=0, y=1, z=2, color='cluster HDP')
    # fig.update_traces(marker=dict(size=5))
    # # fig.write_html('colored_vis.html')
    # fig.show()
    #
    # # usual plot
    # fig_usual = px.scatter_3d(reduced, x=0, y=1, z=2)
    # fig_usual.update_traces(marker=dict(size=5))
    # # fig_usual.write_html('vis.html')
    # fig_usual.show()


def get_topics_for_texts(texts, likelihood_df, dictionary, corpus_dict):
    topic_num_sum_log_p = pd.DataFrame({'topic_num': [], 'sum_log_p': []})
    topic_nums_list = []
    sum_log_p = 0
    sum_p = 0

    #
    topic_num_sum_p = pd.DataFrame({'topic_num': [], 'sum_p': []})
    #

    for text, text_number in zip(texts, range(0, texts.size)):
        for topic in range(0, 10):
            for word in text:
                id = dictionary.token2id[word]
                p = likelihood_df[id][topic]
                count = corpus_dict[text_number][id]

                #
                sum_p = sum_p + p / count
                #

                sum_log_p = sum_log_p + math.log2(p / count)
            topic_num_sum_log_p = topic_num_sum_log_p.append({'topic_num': topic, 'sum_log_p': sum_log_p},
                                                             ignore_index=True)
            sum_log_p = 0

            #
            topic_num_sum_p = topic_num_sum_p.append({'topic_num': topic, 'sum_p': sum_p}, ignore_index=True)
            sum_p = 0
            #

        max = topic_num_sum_log_p['sum_log_p'].max()
        topic_num = topic_num_sum_log_p['sum_log_p'].loc[lambda v: v == max].index[0]
        topic_nums_list.append(topic_num)
        topic_num_sum_log_p = pd.DataFrame({'topic_num': [], 'sum_log_p': []})

        #
        # print("-------------------------------------")
        # print(text)
        # print(topic_num_sum_p)
        # print(topic_num_sum_p.sum())
        topic_num_sum_p = pd.DataFrame({'topic_num': [], 'sum_p': []})
        #

    return topic_nums_list


def count_p(df):
    count_clusters = pd.DataFrame(df.groupby(['cluster HDP']).size()).reset_index(). \
        rename(columns={0: 'count all'})
    count_target_1 = pd.DataFrame(df.loc[df['mark'] == 1].groupby(['cluster HDP']).size()).reset_index(). \
        rename(columns={0: 'count target 1'})
    counted = count_clusters.merge(count_target_1, how='outer', on='cluster HDP').fillna(0)
    return pd.concat([counted['cluster HDP'],
                      counted.apply(lambda row: row['count target 1'] / row['count all'], axis=1)], axis=1).rename(
        columns={0: 'p'})


def predict_cluster_random_forest(matrix, df, df_orig):
    X = pd.concat([pd.DataFrame(matrix), df_orig[['news']]], axis=1)
    y = df['cluster HDP']

    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = X_train.drop(['news'], axis=1)
    X_test_news = X_test['news']
    X_test = X_test.drop(['news'], axis=1)

    # classifier = SVC()
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    y_predicted = classifier.predict(X_test)
    print(f"\npredicted cluster {y_predicted} \ntrue {y_true}")
    print(f"matches amount = {len([x for x, y in zip(y_predicted, y_true) if x == y])} from {len(y_predicted)}")
    print(metrics.confusion_matrix(y_true, y_predicted))
    print(metrics.classification_report(y_true, y_predicted, digits=3))

    return y_predicted, X_test_news


def predict_impact(p_list, cluster):
    if p_list.loc[p_list['cluster HDP'] == cluster, 'p'].iloc[0] > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # classify('marked_dfs/business_standart_marked_df.csv', 'business_standart')
    # classify('marked_dfs/bloomberg_marked_df.csv', 'bloomberg')
    classify('marked_dfs/test.csv', 'test')
