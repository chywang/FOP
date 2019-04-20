from gensim.models import KeyedVectors
import numpy as np
import math
import random
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
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


def cluster_embeddings(pairs, en_model, original_pairs, original_model, matrix):
    global n_clusters, n_embeddings
    new_pairs=list()
    new_original_pairs=list()
    temp=np.zeros(shape=(len(pairs)+len(original_pairs),n_embeddings))
    i=0
    for hypo,hyper in pairs:
        temp[i]=en_model[hyper]-en_model[hypo]
        i=i+1
    for hypo,hyper in original_pairs:
        temp[i] = np.dot(matrix, original_model[hyper]) - np.dot(matrix, original_model[hypo])
        i = i + 1
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
    for hypo,hyper in original_pairs:
        vector = np.dot(matrix, original_model[hyper]) - np.dot(matrix, original_model[hypo])
        weights = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            weights[i]=np.power(math.e, euclidean_distances(vector,centroids[i]))
        weights=preprocessing.normalize(weights, norm='l1').T
        new_original_pairs.append((hypo,hyper,weights))
    return centroids, new_pairs, new_original_pairs


def learn_single_projection(en_model, new_pairs, original_model, new_original_pairs, cluster_index):
    global n_embeddings
    #w: hyper embeddings, v: hypo embeddings

    B=np.zeros(shape=(n_embeddings,n_embeddings))
    #for target language
    for hypo,hyper,weights in new_pairs:
        #print(hypo+'\t'+hyper)
        temp_weights=np.full((n_embeddings, n_embeddings), weights[cluster_index])
        a=en_model[hyper].reshape(n_embeddings, 1)
        b=en_model[hypo].reshape(1,n_embeddings)
        B=B+np.multiply(temp_weights,np.matmul(a,b))

    #average target
    total_features = np.zeros(shape=(1,n_embeddings))
    for hypo,hyper,_ in new_pairs:
        #print(hypo+'\t'+hyper)
        total_features=total_features+(en_model[hyper]-en_model[hypo]).reshape(1, n_embeddings)
    average_features=total_features/len(new_pairs)

    #for source language
    for hypo, hyper, weights in new_original_pairs:
        # print(hypo+'\t'+hyper)
        #transfer learning weight
        t_weight=cosine_similarity(np.dot(matrix,original_model[hyper])-np.dot(matrix,original_model[hypo]),average_features)
        temp_weights = np.full((n_embeddings, n_embeddings), t_weight*weights[cluster_index])
        a = np.dot(matrix,original_model[hyper]).reshape(n_embeddings, 1)
        b = np.dot(matrix,original_model[hypo]).reshape(1, n_embeddings)
        B = B + np.multiply(temp_weights, np.matmul(a, b))

    U, Sigma, Vt = np.linalg.svd(B)
    M=np.eye(n_embeddings,n_embeddings)
    M[n_embeddings-1,n_embeddings-1]=np.linalg.det(U)*np.linalg.det(Vt.T)
    R=np.matmul(np.matmul(U,M),Vt)
    return R


def learn_projections(en_model, new_pairs, original_model, new_original_pairs):
    global n_clusters
    projection_matrices=list()
    for i in range(n_clusters):
        R=learn_single_projection(en_model,new_pairs,original_model, new_original_pairs,cluster_index=i)
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
            #for one projection matrix, compute features, a*(Mx-y)
            pos_f_i=positive_weights[i]*(np.matmul(positive_projections[i], en_model[hypo])-en_model[hyper])
            neg_f_i=negative_weights[i]*(np.matmul(negative_projections[i], en_model[hypo])-en_model[hyper])
            pos_concat[i]=pos_f_i
            neg_concat[i]=neg_f_i
        pos_concat=pos_concat.reshape(n_embeddings*n_clusters, order='C')
        neg_concat=neg_concat.reshape(n_embeddings*n_clusters, order='C')
        all_features= np.concatenate((pos_concat, neg_concat), axis=None)
        features.append(all_features)
    return features


def generate_original_features(original_model, original_pairs, positive_projections, positive_centroids, negative_projections, negative_centroids, matrix):
    global n_clusters
    features=list()
    for hypo, hyper in original_pairs:
        vector = np.dot(matrix, original_model[hyper]) - np.dot(matrix, original_model[hypo])
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
            #for one projection matrix, compute features, a*(Mx-y)
            pos_f_i=positive_weights[i]*(np.matmul(positive_projections[i], np.dot(matrix, original_model[hypo]))-np.dot(matrix, original_model[hyper]))
            neg_f_i=negative_weights[i]*(np.matmul(negative_projections[i], np.dot(matrix, original_model[hypo]))-np.dot(matrix, original_model[hyper]))
            pos_concat[i]=pos_f_i
            neg_concat[i]=neg_f_i
        pos_concat=pos_concat.reshape(n_embeddings*n_clusters, order='C')
        neg_concat=neg_concat.reshape(n_embeddings*n_clusters, order='C')
        all_features= np.concatenate((pos_concat, neg_concat), axis=None)
        features.append(all_features)
    return features


def train_classifier_and_report(pos_features, neg_features, original_positive_freatures, original_negative_freatures):
    global n_clusters, n_embeddings
    thres=0.8
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
    for i in range(0,len(original_positive_freatures)):
        pos_train.append(original_positive_freatures[i])
    for i in range(0,len(original_negative_freatures)):
        neg_train.append(original_negative_freatures[i])

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


#setting the target language
lang='it'

#setting global parameters
global n_clusters
n_clusters=4
global n_embeddings
n_embeddings=300

#loading models
matrix=torch.load('best_mapping_'+lang+'.pth')
warnings.filterwarnings("ignore")
print('load fast text model...')
lang_model = KeyedVectors.load_word2vec_format(lang+'.vec')
print('model load successfully')
print('load eng fast text model...')
original_model = KeyedVectors.load_word2vec_format('en.vec')
print('model eng load successfully')

positive_pairs=load_pairs(lang+'_positive.txt')
negative_pairs=load_pairs(lang+'_negative.txt')
print('data load successfully')
original_positive_pairs=load_pairs('en_positive.txt')
original_negative_pairs=load_pairs('en_negative.txt')
print('data load successfully')


#projection learning
pos_centroids, pos_new_pairs, pos_new_original_pairs=cluster_embeddings(positive_pairs, lang_model, original_positive_pairs, original_model, matrix)
print('cluster pos embeddings successfully')
neg_centroids, neg_new_pairs, neg_new_original_pairs=cluster_embeddings(negative_pairs, lang_model, original_negative_pairs, original_model, matrix)
print('cluster neg embeddings successfully')
pos_projections=learn_projections(lang_model, pos_new_pairs, original_model, pos_new_original_pairs)
print('learn pos projections successfully')
neg_projections=learn_projections(lang_model, neg_new_pairs, original_model, neg_new_original_pairs)
print('learn neg projections successfully')


#feature generation
pos_features=generate_features(lang_model, positive_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('positive features generation successfully')
neg_features=generate_features(lang_model, negative_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('negative features generation successfully')

original_pos_features=generate_original_features(original_model, original_positive_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids, matrix)
print('positive features generation successfully')
original_neg_features=generate_original_features(original_model, original_negative_pairs, pos_projections, pos_centroids, neg_projections, neg_centroids, matrix)
print('negative features generation successfully')

#classifier training
train_classifier_and_report(pos_features, neg_features, original_pos_features, original_neg_features)