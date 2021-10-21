# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math
import numpy as np

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    dev_labels = []
    freq_table_pos = {}
    freq_table_neg = {}
    n_positive = np.count_nonzero(train_labels==1)
    n_negative = np.count_nonzero(train_labels==0)
    n_words_positive = 0
    n_words_negative = 0
    
    for idx,text in enumerate(train_set):
        if train_labels[idx] == 1:
            n_words_positive += len(text)
            for word in text:
                if word in freq_table_pos:
                 freq_table_pos[word] += 1
                else:
                 freq_table_pos[word] = 1
        else:
            n_words_negative += len(text)
            for word in text:
                if word in freq_table_neg:
                 freq_table_neg[word] += 1
                else:
                 freq_table_neg[word] = 1

    for idx, text in enumerate(dev_set):
        log_p_pos = math.log(n_words_positive) 
        log_p_neg = math.log(n_words_negative)
        # calc positive prob
        for word in text:
            if word in freq_table_pos:
                log_p_pos += math.log((freq_table_pos[word]+smoothing_parameter)/(n_words_positive + (len(freq_table_pos.keys())+1)*smoothing_parameter))
            else:
                log_p_pos += math.log(smoothing_parameter/(n_words_positive + (len(freq_table_pos.keys())+1)*smoothing_parameter))

            if word in freq_table_neg:
                log_p_neg += math.log((freq_table_neg[word]+smoothing_parameter)/(n_words_negative + (len(freq_table_neg.keys())+1)*smoothing_parameter))
            else:
                log_p_neg += math.log(smoothing_parameter/(n_words_negative + (len(freq_table_neg.keys())+1)*smoothing_parameter))
        dev_labels.append(int(log_p_pos>log_p_neg))

    return dev_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.1, bigram_smoothing_parameter=0.1, bigram_lambda=0.5, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    dev_labels = []
    freq_uni_pos = {}
    freq_uni_neg = {}
    freq_bi_pos = {}
    freq_bi_neg = {}
    n_words_positive = 0
    n_words_negative = 0
    n_words_positive_bi = 0
    n_words_negative_bi = 0
    log_uni_pos = math.log(pos_prior) 
    log_uni_neg = math.log(1-pos_prior)
    log_bi_pos = math.log(pos_prior)
    log_bi_neg = math.log(1-pos_prior)
    uni_smooth = unigram_smoothing_parameter
    bi_smooth  = bigram_smoothing_parameter
    
    # constructing frequency table
    for idx,text in enumerate(train_set):
        # label is positive
        if train_labels[idx] == 1:
            n_words_positive += len(text)
            n_words_positive_bi += (len(text)-1)
            # unigram
            for word in text:
                if word in freq_uni_pos:
                 freq_uni_pos[word] += 1
                else:
                 freq_uni_pos[word] = 1
            # bigram
            for i in range(len(text)-1):
                if (text[i], text[i+1]) in freq_bi_pos:
                    freq_bi_pos[(text[i], text[i+1])] += 1  
                else:
                    freq_bi_pos[(text[i], text[i+1])] = 1  
        # label is negative            
        else:
            n_words_negative += len(text)
            n_words_negative_bi += (len(text)-1)
            # unigram
            for word in text:
                if word in freq_uni_neg:
                 freq_uni_neg[word] += 1
                else:
                 freq_uni_neg[word] = 1
            # bigram
            for i in range(len(text)-1):
                if (text[i], text[i+1]) in freq_bi_neg:
                    freq_bi_neg[(text[i], text[i+1])] += 1  
                else:
                    freq_bi_neg[(text[i], text[i+1])] = 1     
    
    # constructing dev_labels
    for text in dev_set:
        # unigram
        for word in text:
            if word in freq_uni_pos:
                log_uni_pos += math.log((freq_uni_pos[word]+uni_smooth)/(n_words_positive + (len(freq_uni_pos.keys())+1)*uni_smooth))
            else:
                log_uni_pos += math.log(uni_smooth/(n_words_positive + (len(freq_uni_pos.keys())+1)*uni_smooth))

            if word in freq_uni_neg:
                log_uni_neg += math.log((freq_uni_neg[word]+uni_smooth)/(n_words_negative + (len(freq_uni_neg.keys())+1)*uni_smooth))
            else:
                log_uni_neg += math.log(uni_smooth/(n_words_negative + (len(freq_uni_neg.keys())+1)*uni_smooth))
        # bigram
        for i in range(len(text)-1):
            if (text[i],text[i+1]) in freq_bi_pos:
                log_bi_pos += math.log((freq_bi_pos[(text[i],text[i+1])]+bi_smooth)/(n_words_positive_bi + (len(freq_bi_pos.keys())+1)*bi_smooth))
            else:
                log_bi_pos += math.log(bi_smooth/(n_words_positive_bi + (len(freq_bi_pos.keys())+1)*bi_smooth))

            if (text[i],text[i+1]) in freq_bi_neg:
                log_bi_neg += math.log((freq_bi_neg[(text[i],text[i+1])]+bi_smooth)/(n_words_negative_bi + (len(freq_bi_neg.keys())+1)*bi_smooth))
            else:
                log_bi_neg += math.log(bi_smooth/(n_words_negative_bi + (len(freq_bi_neg.keys())+1)*bi_smooth))
        # combine unigram and bigram
        l_pos = (1-bigram_lambda)*log_uni_pos + bigram_lambda*log_bi_pos
        l_neg = (1-bigram_lambda)*log_uni_neg + bigram_lambda*log_bi_neg
        dev_labels.append(int(l_pos>l_neg))
    
    return dev_labels