# Fuzzy Orthogonal Projection Models for Monolingual and Cross-lingual Hypernymy Prediction 

### By Chengyu Wang (https://chywang.github.io)

**Introduction:** This software is the implementation of three fuzzy orthogonal projection models for both monolingual and cross-lingual hypernymy prediction. A Multi-Wahba Projection (MWP) model is designed to distinguish monolingual hypernymy vs. non-hypernymy relations based on word embeddings. The Transfer MWP (TMWP) model and the Iterative Transfer MWP (ITMWP) model are implemented for cross-lingual hypernymy prediction, which transfer the semantic knowledge from the source language to target languages based on neural word translation.

**Paper:** Wang et al. A Family of Fuzzy Orthogonal Projection Models for Monolingual and Cross-lingual Hypernymy Prediction. WWW 2019


**Models**

+ MWP: The monolingual Multi-Wahaba Projection model, in the MWP package

Codes

1. MWP.py: The script for training the MWP model and reporting the performance over the testing set.

Inputs

1. en.vec: The embeddings of all words. The start of the first line is the number of words and the dimensionality of word embeddings. After that, each line contains the words and its embeddings. All the values in a line are separated by a blank (' '). In practice, the embeddings can be learned by all deep neural language models.

> NOTE: Due to the large size of neural language models, we only upload the embedding vectors of words in the training and testing sets. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. positive_train.txt and negative_train.txt: Positive and negative training sets. The format of the file is "word1 \t word2" pairs.

3. positive_test.txt and negative_test.txt: Positive and negative testing sets. The format of the file is "word1 \t word2" pairs.

Parameters

1. n_clusters: the number of clusters

2. n_embeddings: the dimension of word embeddings

+ TMWP: The cross-lingual Transfer Multi-Wahaba Projection model, in the TMWP package

>NOTE: Due to the large sizes of training sets and neural languae models, we only upload part of the source language training set and use Italian as the target languages. More datasets can be downloaded from https://chywang.github.io.

Codes

1. TMWP.py: The script for training the TMWP model and reporting the performance over the testing set.

Inputs

1. en.vec: Same as in MWP.

2. it.vec: The embeddings of all Italian words in the training and testing sets.

3. best_mapping_it.pth: The mapping matrix from the source language embedding space to the target language embedding space.

>NOTE: Refer to the MUSE project for details: https://github.com/facebookresearch/MUSE.

4. en_positive.txt and en_negative.txt: Positive and negative training sets of the source language.

5. it_positive_train.txt, it_negative_train.txt, it_positive_test.txt and it_negative_test.txt: The training and testing sets of the target language (i.e., Italian in this project).

Parameters: Same as in MWP.

+ ITMWP: The cross-lingual Iterative Transfer Multi-Wahaba Projection model, in the ITMWP package

Inputs: Same as in TMWP.

Additional Parameters:

1. n_iter: The number of iterations.

2. tau: The threshold to control the training data augmentation process.

**Dependencies**

The main Python packages that we use in this project include but are not limited to:

1. torch: 0.4.1
2. gensim: 2.0.0
3. numpy: 1.15.4
4. tensorflow: 1.12.0
5. scikit-learn: 0.18.1

The codes can run properly under the packages of other versions as well.

**More Notes on the Algorithms**

This software is the implementation of the WWW 2019 paper. The codes have been slightly modified to make them easier for NLP researcher to re-use. Due to size limitation, we only provide part of the data related to our paper. However, it should be noted that all the data and resources are publicly available. Please refer to the links and references for details.


**Citation**

If you find this software useful for your research, please cite the following paper.

> @inproceedings{www2019,<br/>
&emsp;&emsp; author = {Chengyu Wang and Yan Fan and Xiaofeng He and Aoying Zhou},<br/>
&emsp;&emsp; title = {A Family of Fuzzy Orthogonal Projection Models for Monolingual and Cross-lingual Hypernymy Prediction},<br/>
&emsp;&emsp; booktitle = {Proceedings of the 2019 World Wide Web Conference},<br/>
&emsp;&emsp; year = {2019}<br/>
}

More research works can be found here: https://chywang.github.io.



