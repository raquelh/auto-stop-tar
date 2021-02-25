# coding=utf-8
import sys

sys.path.append("/home/raquel/Documentos/TCCII/auto-stop-tar/autostop/")
import pyltr
import scipy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score
from sklearn.svm import SVC

from rank_bm25 import BM25Okapi
from nltk.stem.porter import *
porter_stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize

from tar_framework.utils import *


def preprocess_text(text):
    """
    1. Remove punctuation.
    2. Tokenize.
    3. Remove stopwords.
    4. Stem word.
    """
    #sys.exit()
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', ' ', text)
    #print(text)
    # tokenize
    tokens = word_tokenize(text)
    # lowercase & filter stopwords
    #filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    #filtered = [token for token in tokens]
    # # stem
    stemmed = [porter_stemmer.stem(token) for token in filtered]
    return stemmed


def bm25_okapi_rank(complete_dids, complete_texts, query):
    #sys.exit()
    tokenized_texts = [preprocess_text(doc) for doc in complete_texts]
    tokenized_query = preprocess_text(query)

    bm25 = BM25Okapi(tokenized_texts)
    scores = bm25.get_scores(tokenized_query)

    did_scores = sorted(zip(complete_dids, scores), key=lambda x: x[1], reverse=True)
    ranked_dids, ranked_scores = zip(*did_scores)
    print(ranked_dids)
    return list(ranked_dids), list(ranked_scores)


class Ranker(object):
    """
    Manager the ranking module of the TAR framework.
    """
    def __init__(self, model_type='svm', min_df=2, C=1.0, random_state=0):
        self.model_type = model_type
        self.random_state = random_state
        self.min_df = min_df
        self.C = C
        self.did2feature = {}
        #print(type(self.did2feature)) tupo dicionario 
        self.name2features = {}

        if self.model_type == 'lr':
            self.model = LogisticRegression(solver='lbfgs', random_state=self.random_state, C=self.C, max_iter=10000)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, gamma='scale', random_state=self.random_state)
        elif self.model_type == 'lambdamart':
            self.model = None
        else:
            raise NotImplementedError

    def set_did_2_feature(self, dids, texts, corpus_texts):
        #texts e corpus_texts são listas
        #print(type(texts)) 
        #original
        #tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=int(self.min_df))
        #usado no harmless
        #print(corpus_texts)
        #sys.exit()
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, stop_words=None, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False,decode_error="ignore", max_features=4000)
        tfidf_vectorizer.fit(corpus_texts)
        features = tfidf_vectorizer.transform(texts)
        print('tamanho',features.shape[1])
        #sys.exit()
        for did, feature in zip(dids, features):
            self.did2feature[did] = feature
        logging.info('Ranker.set_feature_dict is done.')
        return

    def get_feature_by_did(self, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        return features

    def set_features_by_name(self, name, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        self.name2features[name] = features
        return

    def get_features_by_name(self, name):
        return self.name2features[name]

    def train(self, features, labels):
        if self.model_type == 'lambdamart':
            # retrain the model at each TAR iteration. Otherwise, the training speed will be slowed drastically.
            model = pyltr.models.LambdaMART(
                metric=pyltr.metrics.NDCG(k=10),
                n_estimators=100,
                learning_rate=0.02,
                max_features=0.5,
                query_subsample=0.5,
                max_leaf_nodes=10,
                min_samples_leaf=64,
                verbose=0,
                random_state=self.random_state)
            
        else:
            model = self.model
        
        model.fit(features, labels)
        #probs = model.predict(features)
        
        
        #print('score',model.score(features, labels))
        #print('acuracia', accuracy_score(labels, probs, normalize=False))
        #print('recall', recall_score(labels, probs))
        # logging.info('Ranker.train is done.')
        return

    def predict(self, features):
        probs = self.model.predict_proba(features)
        #print(probs)
        rel_class_inx = list(self.model.classes_).index(REL)

        scores = probs[:, rel_class_inx]
        #print(list(self.model.classes_))
        #print(features), len(labels))
        #print( self.model.score(features, labels))
        return scores

