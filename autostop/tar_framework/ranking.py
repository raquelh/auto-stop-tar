# coding=utf-8

import sys 

sys.path.append("/home/raquel/Documentos/TCCII/novo/auto-stop-tar-master/")
import pyltr
import scipy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

from rank_bm25 import BM25Okapi
from nltk.stem.porter import *
porter_stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize

from autostop.tar_framework.utils import *


def preprocess_text(text):
    """
    1. Remove punctuation.
    2. Tokenize.
    3. Remove stopwords.
    4. Stem word.
    """
    # lowercase
    text = text.lower()
    # remove punctuation
    #text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', ' ', text)
    # tokenize
    tokens = word_tokenize(text)
    # lowercase & filter stopwords
    #filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    # # stem
    stemmed = [porter_stemmer.stem(token) for token in filtered]
    
    return stemmed


def bm25_okapi_rank(complete_dids, complete_texts, query):
    tokenized_texts = [preprocess_text(doc) for doc in complete_texts]
    tokenized_query = preprocess_text(query)

    bm25 = BM25Okapi(tokenized_texts)
    scores = bm25.get_scores(tokenized_query)

    did_scores = sorted(zip(complete_dids, scores), key=lambda x: x[1], reverse=True)
    ranked_dids, ranked_scores = zip(*did_scores)

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
        self.name2features = {}

        if self.model_type == 'lr':
            self.model = LogisticRegression(solver='lbfgs', random_state=self.random_state, C=self.C, max_iter=10000)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, gamma='scale', random_state=self.random_state)
        elif self.model_type == 'lambdamart':
            self.model = None
        else:
            raise NotImplementedError

    def set_did_2_feature(self, dids, texts, corpus_texts, metrics):
        
        
        
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, stop_words=None, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False,decode_error="ignore", max_features=4000)
        tfidf_vectorizer.fit(corpus_texts)

        features = tfidf_vectorizer.transform(texts)

        for did, feature in zip(dids, features):
            self.did2feature[did] = feature

        self.did3feature = metrics    
        logging.info('Ranker.set_feature_dict is done.')
        return

    def get_feature_by_did(self, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        return features


    def get_feature_metric_by_did(self, dids):
        #features_mtc = np.array([])
        count = 0
        for did in dids:
            if did == 'pseudo_did':
                i = 115
            else:
                i = int(did)

            lista = [int(val) for val in self.did3feature[i]]
            if count == 0:
                features_mtc = np.array([lista])
            else:
                features_mtc = np.append(features_mtc, [lista], axis=0)
            count +=1
            #np.vstack([features_mtc, lista])
            #features_mtc = np.append(features_mtc, [lista], axis=1)
        
        #print(features_mtc.shape[0])
        return features_mtc     

    def set_features_by_name(self, name, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        self.name2features[name] = features
        return

    def get_features_by_name(self, name):
        return self.name2features[name]

    def get_metrics_by_name(self, dids):
        count = 0
        for did in dids:
            i = int(did)
            lista = [int(val) for val in self.did3feature[i]]
            if count == 0:
                features_mtc = np.array([lista])
            else:
                features_mtc = np.append(features_mtc, [lista], axis=0)
            count +=1
        return features_mtc    

    def train(self, features, metrics, labels):
        metrics = scipy.sparse.csr_matrix(metrics)
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

        features_mix = scipy.sparse.hstack([features,metrics]).toarray()    
        model.fit(features_mix, labels)
        # logging.info('Ranker.train is done.')
        return

    def predict(self, features, metrics):
        metrics = scipy.sparse.csr_matrix(metrics)
        features_mix = scipy.sparse.hstack([features,metrics]).toarray()
        probs = self.model.predict_proba(features_mix)
        rel_class_inx = list(self.model.classes_).index(REL)
        scores = probs[:, rel_class_inx]
        return scores

