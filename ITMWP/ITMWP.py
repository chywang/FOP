from gensim.models import KeyedVectors
import numpy as np
import math
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


def generate_features_for_one_pair(en_model, pair, positive_projections, positive_centroids, negative_projections, negative_centroids):
    global n_clusters
    hypo, hyper=pair
    vector = en_model[hyper] - en_model[hypo]
    positive_weights = np.zeros(shape=n_clusters)
    negative_weights = np.zeros(shape=n_clusters)
    for i in range(n_clusters):
        positive_weights[i] = np.power(math.e, euclidean_distances(vector, positive_centroids[i]))
        negative_weights[i] = np.power(math.e, euclidean_distances(vector, negative_centroids[i]))
    positive_weights = preprocessing.normalize(positive_weights, norm='l1').T
    negative_weights = preprocessing.normalize(negative_weights, norm='l1').T
    pos_concat = np.zeros(shape=(n_clusters, n_embeddings))
    neg_concat = np.zeros(shape=(n_clusters, n_embeddings))
    for i in range(n_clusters):
        # for one projection matrix, compute features, a*(Mx-y)
        pos_f_i = positive_weights[i] * (np.matmul(positive_projections[i], en_model[hypo]) - en_model[hyper])
        neg_f_i = negative_weights[i] * (np.matmul(negative_projections[i], en_model[hypo]) - en_model[hyper])
        pos_concat[i] = pos_f_i
        neg_concat[i] = neg_f_i
    pos_concat = pos_concat.reshape(n_embeddings * n_clusters, order='C')
    neg_concat = neg_concat.reshape(n_embeddings * n_clusters, order='C')
    return pos_concat, neg_concat


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


def train_classifier(pos_features, neg_features, original_positive_freatures, original_negative_freatures):
    global n_clusters, n_embeddings
    dim=2*n_clusters*n_embeddings
    pos_len = len(pos_features)
    neg_len = len(neg_features)
    pos_train=list()
    neg_train = list()
    for i in range(0,pos_len):
        pos_train.append(pos_features[i])
    for i in range(0, neg_len):
        neg_train.append(neg_features[i])
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
    #Train the model
    cls = MLPClassifier(solver='adam', alpha=1e-5)
    cls.fit(train_data, train_labels)
    return cls


def test_classifier(pos_features, neg_features, cls):
    global n_clusters, n_embeddings
    dim = 2 * n_clusters * n_embeddings
    test_data = np.zeros(shape=(len(pos_features) + len(neg_features), dim))
    test_labels = np.zeros(shape=(len(pos_features) + len(neg_features), 1))
    for i in range(0, len(pos_features)):
        test_data[i] = pos_features[i]
        test_labels[i] = 1
    for i in range(0, len(neg_features)):
        test_data[i + len(pos_features)] = neg_features[i]
        test_labels[i + len(pos_features)] = 0
    result=cls.score(test_data, test_labels)
    print(result)


def training_for_one_iteration(iter):
    print(iter)
    if iter==0:
        positive_pairs_train = load_pairs(lang + '_positive_train.txt')
        negative_pairs_train = load_pairs(lang + '_negative_train.txt')
    else:
        positive_pairs_train = load_pairs(lang + '_positive_train_' + str(iter) + '.txt')
        negative_pairs_train = load_pairs(lang + '_negative_train_' + str(iter) + '.txt')

    print('data load successfully')
    original_positive_pairs = load_pairs('en_positive.txt')
    original_negative_pairs = load_pairs('en_negative.txt')
    print('data load successfully')

    # projection learning
    pos_centroids, pos_new_pairs, pos_new_original_pairs = cluster_embeddings(positive_pairs_train, lang_model,
                                                                              original_positive_pairs, original_model,
                                                                              matrix)
    print('cluster pos embeddings successfully')
    neg_centroids, neg_new_pairs, neg_new_original_pairs = cluster_embeddings(negative_pairs_train, lang_model,
                                                                              original_negative_pairs, original_model,
                                                                              matrix)
    print('cluster neg embeddings successfully')
    pos_projections = learn_projections(lang_model, pos_new_pairs, original_model, pos_new_original_pairs)
    print('learn pos projections successfully')
    neg_projections = learn_projections(lang_model, neg_new_pairs, original_model, neg_new_original_pairs)
    print('learn neg projections successfully')
    # feature generation
    pos_features = generate_features(lang_model, positive_pairs_train, pos_projections, pos_centroids, neg_projections,
                                     neg_centroids)
    print('positive features generation successfully')
    neg_features = generate_features(lang_model, negative_pairs_train, pos_projections, pos_centroids, neg_projections,
                                     neg_centroids)
    print('negative features generation successfully')
    original_pos_features = generate_original_features(original_model, original_positive_pairs, pos_projections,
                                                       pos_centroids, neg_projections, neg_centroids, matrix)
    print('positive features generation successfully')
    original_neg_features = generate_original_features(original_model, original_negative_pairs, pos_projections,
                                                       pos_centroids, neg_projections, neg_centroids, matrix)
    print('negative features generation successfully')
    # classifier training
    cls = train_classifier(pos_features, neg_features, original_pos_features, original_neg_features)
    return cls, pos_projections, pos_centroids, neg_projections, neg_centroids


def training_data_aug_for_one_iteration(iter, cls, pos_projections, pos_centroids, neg_projections, neg_centroids, tau):
    global n_clusters, n_embeddings, new_training_pairs

    new_positive_pairs=list()
    new_negative_pairs=list()

    positive_pairs_test = load_pairs(lang + '_positive_test.txt')
    negative_pairs_test = load_pairs(lang + '_negative_test.txt')
    unlabeled_set=positive_pairs_test.union(negative_pairs_test)
    for pair in unlabeled_set:
        if pair in new_training_pairs:
            continue
        pos_features, neg_features=generate_features_for_one_pair(lang_model, pair, pos_projections, pos_centroids, neg_projections,
                                     neg_centroids)
        max_value=np.linalg.norm(pos_features, ord=2)
        if np.linalg.norm(neg_features, ord=2)>max_value:
            max_value=np.linalg.norm(neg_features, ord=2)
        diff=np.abs(np.linalg.norm(pos_features, ord=2)-np.linalg.norm(neg_features, ord=2))
        conf=2.5*diff/max_value
        if conf<tau:
            continue
        #do testing
        test_features = np.concatenate((pos_features, neg_features), axis=None)
        predict_result=cls.predict(test_features)
        if predict_result==1:
            new_positive_pairs.append(pair)
        else:
            new_negative_pairs.append(pair)
        new_training_pairs.add(pair)
    return new_positive_pairs, new_negative_pairs


def update_training_sets(iter, new_positive_pairs, new_negative_pairs):
    print('writing files')
    if iter==0:
        positive_pairs_train_old = load_pairs(lang + '_positive_train.txt')
        negative_pairs_train_old = load_pairs(lang + '_negative_train.txt')
    else:
        positive_pairs_train_old = load_pairs(lang + '_positive_train_' + str(iter) + '.txt')
        negative_pairs_train_old = load_pairs(lang + '_negative_train_' + str(iter) + '.txt')
    new_iter=iter+1
    out_positive=open(lang + '_positive_train_' + str(new_iter) + '.txt','w+')
    for hypo,hyper in positive_pairs_train_old:
        out_positive.write(hypo+'\t'+hyper+'\n')
    for hypo,hyper in new_positive_pairs:
        out_positive.write(hypo+'\t'+hyper+'\n')
    out_positive.close()

    out_negative = open(lang + '_negative_train_' + str(new_iter) + '.txt', 'w+')
    for hypo, hyper in negative_pairs_train_old:
        out_negative.write(hypo + '\t' + hyper + '\n')
    for hypo, hyper in new_negative_pairs:
        out_negative.write(hypo + '\t' + hyper + '\n')
    out_negative.close()


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

#setting number of iterations
n_iter=5
#setting hyper-parameters
tau=0.6

prediction_cls = MLPClassifier()
global new_training_pairs
new_training_pairs=set()
for i in range(0,n_iter):
    cls, pos_projections, pos_centroids, neg_projections, neg_centroids=training_for_one_iteration(i)
    prediction_cls=cls
    new_positive_pairs, new_negative_pairs=training_data_aug_for_one_iteration(i, cls, pos_projections, pos_centroids, neg_projections, neg_centroids, tau)
    print(len(new_positive_pairs)+len(new_negative_pairs))
    if len(new_positive_pairs)+len(new_negative_pairs)==0:
        break
    update_training_sets(i, new_positive_pairs, new_negative_pairs)
print('iteration ends')

#for testing data
positive_pairs_test=load_pairs(lang+'_positive_test.txt')
negative_pairs_test=load_pairs(lang+'_negative_test.txt')
pos_features=generate_features(lang_model, positive_pairs_test, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('positive features generation successfully')
neg_features=generate_features(lang_model, negative_pairs_test, pos_projections, pos_centroids, neg_projections, neg_centroids)
print('negative features generation successfully')
test_classifier(pos_features, neg_features, prediction_cls)
