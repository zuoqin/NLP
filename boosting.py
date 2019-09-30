import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

bbc_text_df = pd.read_csv('./bbc-text.csv')
bbc_text_df.head()


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
def simple_plot():
    font_name = "SimSun"
    plt.rcParams['font.family']=font_name
    plt.rcParams['axes.unicode_minus']=False # in case minus sign is shown as box
    


    fontP = font_manager.FontProperties()
    fontP.set_family('SimSun')
    fontP.set_size(14)
    


    plt.figure(figsize=(12,5))
    sns.countplot(x=bbc_text_df.category, color='green')
    plt.title('BBC text class distribution', fontsize=16)
    
    plt.legend(loc=0, prop=fontP)
    plt.title('债券收益率', fontproperties=fontP)
    
    plt.ylabel('Class Counts', fontsize=16)
    plt.xlabel('Class Label', fontproperties=fontP)
    plt.xticks(rotation='vertical');

    plt.show()

from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s
   

from wordcloud import WordCloud

def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords=None,
                min_font_size = 10).generate(text) 
             
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud_instance) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()
    
def plot_word_cloud_for_category(bbc_text_df, category):
    text_df = bbc_text_df.loc[bbc_text_df['category'] == str(category)]
    texts = ''
    for index, item in text_df.iterrows():
        texts = texts + ' ' + clean_text(item['text'])
    
    plot_word_cloud(texts)

texts = ''
for index, item in bbc_text_df.iterrows():
    texts = texts + ' ' + clean_text(item['text'])
    
#plot_word_cloud(texts)

#plot_word_cloud_for_category(bbc_text_df,'politics')

df_x = bbc_text_df['text']
df_y = bbc_text_df['category']

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm

import multiprocessing
import numpy as np

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(clean_text(row).split(), [index]) for index, row in enumerate(df_x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(clean_text(row).split())
                                     for index, row in enumerate(df_x)]))
                                     
                                     
  
from sklearn.feature_extraction.text import TfidfVectorizer

class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)
        
        
def test():                                  
    #doc2vec_trf = Doc2VecTransformer()
    #doc2vec_features = doc2vec_trf.fit(df_x).transform(df_x)
    #print(doc2vec_features)

    #pl_log_reg = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
    #                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))])
    #scores = cross_val_score(pl_log_reg, df_x, df_y, cv=5,scoring='accuracy')
    #print('Accuracy for Logistic Regression: ', scores.mean())
    tfidf_transformer = Text2TfIdfTransformer()
    tfidf_vectors = tfidf_transformer.fit(df_x).transform(df_x)
    #print(tfidf_vectors.shape)

    
def test_tfidf_logistic_normal():
    pl_log_reg_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))])
    scores = cross_val_score(pl_log_reg_tf_idf, df_x, df_y, cv=5,scoring='accuracy')
    print('Accuracy for Tf-Idf & Logistic Regression: ', scores.mean())


def test_tfidf_xgboost():
    pl_xgb_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                         ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
    scores = cross_val_score(pl_xgb_tf_idf, df_x, df_y, cv=5)
    print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())
    
    
test_tfidf_logistic_normal()
