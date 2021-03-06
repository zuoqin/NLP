import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

def create_data():
    Corpus = pd.read_csv(r"./corpus.csv",encoding='latin-1')
    #print(Corpus.head())

    # Step - a : Remove blank rows if any.
    Corpus['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['text'] = [entry.lower() for entry in Corpus['text']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)

    Corpus.to_pickle("./corpus.pkl")

def create_chinese_data():
    Corpus = pd.read_excel('./wangwenjia.xlsx')
    #Corpus = Corpus[(Corpus['category'] == 'RTDEVM_PIC') | (Corpus['category'] =='SR_FRAME_DRV') |
    #    (Corpus['category'] =='BR_UM') | (Corpus['category'] =='RT_NSE') | (Corpus['category'] =='RTADAPT_L2VPN') | (Corpus['category'] =='PRODUCT_INCLUDE')]
    Corpus = Corpus[(Corpus['category'] == 'FEI_DOP_588X') | (Corpus['category'] =='BR_UM') |
        (Corpus['category'] =='BOARD_DRIVER') | (Corpus['category'] =='公共模块') | (Corpus['category'] =='BR_AAA') | (Corpus['category'] =='VSM')]
    #Corpus = Corpus[Corpus.text.apply(lambda x: x.isalpha())]
    Corpus = Corpus[~Corpus.text.isnull()]
    #print(Corpus.head())
    Corpus.to_pickle("./corpus_chinese_clean.pkl")


def process1():
    Corpus = pd.read_pickle("./corpus.pkl")
    print(Corpus.label.unique())

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


def process_chinese():
    Corpus = pd.read_pickle("./corpus_chinese_clean.pkl")

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'],Corpus['category'],test_size=0.3)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    
    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


#create_chinese_data()
#process_chinese()
print('6666')
