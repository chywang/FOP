from gensim.models import KeyedVectors
import numpy as np
import math
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import warnings


def load_pairs(path):
    pairs=set()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        hypo = str[0]
        hyper = str[1]
        pairs.add((hypo,hyper))
    file.close()
    return pairs


def cluster_embeddings(pairs, en_model):
    global n_clusters, n_embeddings
    new_pairs=list()
    temp=np.zeros(shape=(len(pairs),n_embeddings))
    i=0
    for hypo,hyper in pairs:
        temp[i]=en_model[hyper]-en_model[hypo]
        i=i+1
    estimator = KMeans(n_clusters)
    estimator.fit(temp)
    centroids = estimator.cluster_centers_
    for hypo,hyper in pairs:
        vector=en_model[hyper]-en_model[hypo]
        weights = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            weights[i]=np.power(math.e, euclidean_distances(vector,centroids[i]))
        weights=preprocessing.normalize(weights, norm='l1').T
        new_pairs.append((hypo,hyper,weights))
    return centroids, new_pairs


def learn_single_projection(en_model, new_pairs, cluster_index):
    global n_embeddings
    #w: hyper embeddings, v: hypo embeddings
    B=np.zeros(shape=(n_embeddings,n_embeddings))
    for hypo,hyper,weights in new_pairs:
        #print(hypo+'\t'+hyper)
        temp_weights=np.full((n_embeddings, n_embeddings), weights[cluster_index])
        a=en_model[hyper].reshape(n_embeddings, 1)
        b=en_model[hypo].reshape(1,n_embeddings)
        B=B+np.multiply(temp_weights,np.matmul(a,b))
    U, Sigma, Vt = np.linalg.svd(B)
    M=np.eye(n_embeddings,n_embeddings)
    M[n_embeddings-1,n_embeddings-1]=np.linalg.det(U)*np.linalg.det(Vt.T)
    R=np.matmul(np.matmul(U,M),Vt)
    return R


def learn_projections(en_model, new_pairs):
    global n_clusters
    projection_matrices=list()
    for i in range(n_clusters):
        R=learn_single_projection(en_model,new_pairs,cluster_index=i)
        projection_matrices.append(R)
    return projection_matrices


def generate_features(en_model, pairs, positive_projections, positive_centroids, negative_projections, negative_centroids):
    global n_clusters
    features=list()
    for hypo, hyper in pairs:
        vector = en_model[hyper] - en_model[hypo]
        positive_weights = np.zeros(shape=n_clusters)
        negative_weights = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            positive_weights[i] = np.power(math.e, euclidean_distances(vector, positive_centroids[i]))
            negative_weights[i] = np.power(math.e, euclidean_distances(vector, negative_centroids[i]))
        positive_weights = preprocessing.normalize(positive_weights, norm='l1').T
        negative_weights = preprocessing.normalize(negative_weights, norm='l1').T
        pos_concat=np.zeros(shape=(n_clusters,n_embeddings))
        neg_concat=np.zeros(shape=(n_clusters,n_embeddings))
        for i in range(n_clusters):
            pos_f_i=positive_weights[i]*(np.matmul(positive_projections[i], en_model[hypo])-en_model[hyper])
            neg_f_i=negative_weights[i]*(np.matmul(negative_projections[i], en_model[hypo])-en_model[hyper])
            pos_concat[i]=pos_f_i
            neg_concat[i]=neg_f_i
        pos_concat=pos_concat.reshape(n_embeddings*n_clusters, order='C')
        neg_concat=neg_concat.reshape(n_embeddings*n_clusters, order='C')
        all_features= np.concatenate((pos_concat, neg_concat), axis=None)
        features.append(all_features)
    return features


def train_classifier_and_report(positive_features, negative_features):
    # for simplicity, in this function, we use a part of the data for training and the rest for testing
    # please refer to the paper for detailed evaluation methods
    thres=0.1

    dim=2*n_clusters*n_embeddings
    pos_len = len(pos_features)
    neg_len = len(neg_features)
    pos_train=list()
    pos_test=list()
    neg_train = list()
    neg_test = list()
    for i in range(0,pos_len):
        if random.random()>thres:
            pos_train.append(pos_features[i])
        else:
            pos_test.append(pos_features[i])
    for i in range(0, neg_len):
        if random.random() > thres:
            neg_train.append(neg_features[i])
        else:
            neg_test.append(neg_features[i])

    train_data = np.zeros(shape=(len(pos_train)+len(neg_train), dim))
    train_labels = np.zeros(shape=(len(pos_train)+len(neg_train), 1))
    for i in range(0,len(pos_train)):
        train_data[i]=pos_train[i]
        train_labels[i]=1
    for i in range(0,len(neg_train)):
        train_data[i+len(pos_train)]=neg_train[i]
        train_labels[i+len(pos_train)]=0

    test_data = np.zeros(shape=(len(pos_test)+len(neg_test), dim))
    test_labels = np.zeros(shape=(len(pos_test)+len(neg_test), 1))
    for i in range(0,len(pos_test)):
        test_data[i]=pos_test[i]
        test_labels[i]=1
    for i in range(0,len(neg_test)):
        test_data[i+len(pos_test)]=neg_test[i]
        test_labels[i+len(pos_test)]=0
    #Train the model
    cls = MLPClassifier(solver='adam', alpha=1e-5)
    cls.fit(train_data, train_labels)
    result=cls.score(test_data, test_labels)
    print(result)


#global parameter settings
global n_clusters
n_clusters=4
global n_embeddings
n_embeddings=300

#loading...
warnings.filterwarnings("ignore")
print('load fast text model...')
en_model = KeyedVectors.load_word2vec_format('en.vec')
print('model load successfully')
positive_pairs=load_pairs('positive.txt')
negative_pairs=load_pairs('negative.txt')
print('data load successfully')

#projection learning
pos_centroids, pos_new_pairs=cluster_embeddings(positive_pairs, en_model)
print('cluster pos embeddings successfully')
neg_centroids, neg_new_pairs=cluster_embeddings(negative_pairs, en_model)
print('cluster neg embeddings successfully')
pos_projections=learn_projections(en_model, pos_new_pairs)
print('learn pos projections successfully')
neg_projections=learn_projections(en_model, neg_new_pairs)
print('learn neg projections successfully')

#feature generation
pos_features=generate_features(en_model, positive_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('positive features generation successfully')
neg_features=generate_features(en_model, negative_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('negative features generation successfully')

#classifier training
train_classifier_and_report(pos_features, neg_features)