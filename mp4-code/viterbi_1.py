"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import sys
import numpy as np 

def trace_back(trellis, tag_list):
    path = ['END']
    idx_word = len(trellis[0])-1
    prev_tag = int(trellis[tag_list.index('END')][idx_word][1])
    counter = 0 
    while prev_tag != -1 and counter <= 200:
        path.append(tag_list[prev_tag])
        idx_word -= 1
        counter += 1 
        prev_tag = int(trellis[prev_tag][idx_word][1])
        
    path.reverse()
    return path 

def smooth_twTable(tag_word_table, smoothing_param):
    for tag in tag_word_table.keys():
        for word in tag_word_table[tag]:
            tag_word_table[tag][word] += smoothing_param
        tag_word_table[tag]['UNKNOWN'] = smoothing_param

def smooth_transMtx(trans_mtx, tag_list, smoothing_param):
    trans_mtx['END'] = dict()
    for curTag in trans_mtx.keys():
        for tag in tag_list:
            if tag not in trans_mtx[curTag]:
                trans_mtx[curTag][tag] = smoothing_param
            else:
                trans_mtx[curTag][tag] += smoothing_param

def logProb_transMtx(trans_mtx):
    # compute the log probabilities
    for curTag in trans_mtx.keys():
        sum_trans = sum(trans_mtx[curTag].values())
        if sum_trans == 0:
            print(curTag)
        for nextTag in trans_mtx[curTag].keys():
            trans_mtx[curTag][nextTag] = np.log(trans_mtx[curTag][nextTag]/sum_trans)

def logProb_emission(tag_word_table):
    # compute the log probabilities
    for tag in tag_word_table:
        sum_freq = sum(tag_word_table[tag].values())
        for word in tag_word_table[tag].keys():
            tag_word_table[tag][word] = np.log(tag_word_table[tag][word]/sum_freq)

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_word_table = dict()
    trans_mtx = dict()
    smoothing_param_emis = 0.00001 
    smoothing_param_tran = 0.00001
    tag_list = []
    res = []

    # training - generating transition matrix
    for sen in train:
        prev_tag = ''
        for i, (word, tag) in enumerate(sen):
            # add to tag list 
            if tag not in tag_list:
                tag_list.append(tag)
            # tag-word frequency table
            if tag not in tag_word_table:
                tag_word_table[tag] = dict()
                tag_word_table[tag][word] = 1 
            else:
                if word not in tag_word_table[tag]:
                    tag_word_table[tag][word] = 1
                else:
                    tag_word_table[tag][word] += 1
            # transition matrix
            if i == 0:
                prev_tag = tag
                continue
            if prev_tag not in trans_mtx:
                trans_mtx[prev_tag] = dict()
                trans_mtx[prev_tag][tag] = 1
            else:
                if tag not in trans_mtx[prev_tag]:
                    trans_mtx[prev_tag][tag] = 1
                else:
                    trans_mtx[prev_tag][tag] += 1
            prev_tag = tag

    # Laplace-smoothing/computing log probabilities for the tag-word table
    smooth_twTable(tag_word_table, smoothing_param_emis)
    logProb_emission(tag_word_table)

    #print('trans_mtx', trans_mtx)
    # Laplace-smoothing/computing log probabilities for the transition matrix 
    smooth_transMtx(trans_mtx, tag_list, smoothing_param_tran)
    logProb_transMtx(trans_mtx)

    # constructing the trellis data structure
    for sen in test:
        # dimensions: num of tags x num of words x probability/previous tag
        trellis = np.zeros((len(tag_list), len(sen), 2))
        # initialized trellis for START
        for idx,tag in enumerate(tag_list):
            trellis[idx][0][1] = -1
            if tag != 'START':
                trellis[idx][0][0] = ~sys.maxsize

        for idx_w,word in enumerate(sen):
            if word == 'START':
                continue
            for idx_t,tag in enumerate(tag_list):
                probs = []
                for idx_pt,prev_tag in enumerate(tag_list):
                    if word not in tag_word_table[tag]:
                        emis_prob = tag_word_table[tag]['UNKNOWN']
                    else:
                        emis_prob = tag_word_table[tag][word]
                    
                    tran_prob = trans_mtx[prev_tag][tag]
                    probs.append(trellis[idx_pt][idx_w-1][0] + emis_prob + tran_prob)
                
                trellis[idx_t][idx_w][0] = max(probs)
                trellis[idx_t][idx_w][1] = probs.index(max(probs))
        # trace back from END to START
        res_tags = trace_back(trellis, tag_list)
        assert len(res_tags) == len(sen)
        res_pairs = []
        for i,word in enumerate(sen):
            res_pairs.append((word, res_tags[i]))
        res.append(res_pairs)

    return res
