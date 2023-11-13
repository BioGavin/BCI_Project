import os.path

import pandas as pd
import re
import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt


def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        # word = seg_word.word  #如果想要分析英语文本，注释这行代码，启动下行代码
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return (" ").join(word_list)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


if __name__ == '__main__':
    data_file = "data.xlsx"  # content type
    output_folder = "result"
    pdf_filenama = "plot.pdf"
    dic_file = "dict.txt"
    stop_file = "stopwords.txt"

    data = pd.read_excel(data_file)
    data["content_cutted"] = data.content.apply(chinese_word_cut)

    n_features = 1000  # 提取1000个特征词语
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(data.content_cutted)

    n_topics = 8  # 超参

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50,
                                    # doc_topic_prior=0.1,
                                    # topic_word_prior=0.01,
                                    random_state=0)
    lda.fit(tf)

    n_top_words = 25
    tf_feature_names = tf_vectorizer.get_feature_names()
    topic_word = print_top_words(lda, tf_feature_names, n_top_words)

    topics = lda.transform(tf)

    topic = []
    for t in topics:
        topic.append("Topic #" + str(list(t).index(np.max(t))))
    data['概率最大的主题序号'] = topic
    data['每个主题对应概率'] = list(topics)
    data.to_excel("data_topic.xlsx", index=False)

    # pyLDAvis.enable_notebook()
    pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    # pyLDAvis.display(pic)
    pyLDAvis.save_html(pic, os.path.join(output_folder, 'lda_pass' + str(n_topics) + '.html'))
    # pyLDAvis.display(pic)
    # 去工作路径下找保存好的html文件
    # 和视频里讲的不一样，目前这个代码不需要手动中断运行，可以快速出结果

    plexs = []
    scores = []
    n_max_topics = 16
    for i in range(1, n_max_topics):
        print(i)
        lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                        learning_method='batch',
                                        learning_offset=50, random_state=0)
        lda.fit(tf)
        plexs.append(lda.perplexity(tf))
        scores.append(lda.score(tf))

    n_t = 15  # 区间最右侧的值。注意：不能大于n_max_topics
    x = list(range(1, n_t + 1))
    plt.plot(x, plexs[0:n_t])
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.savefig(os.path.join(output_folder, pdf_filenama), format='pdf', dpi=300)
    plt.show()
