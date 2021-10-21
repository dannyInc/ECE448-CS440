"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
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
    """
    sum_hapax = sum(appear_once.values())
    for tag in appear_once.keys():
        appear_once[tag] /= sum_hapax
    """
    for tag in tag_word_table.keys():
        for word in tag_word_table[tag]:
            tag_word_table[tag][word] += smoothing_param
        tag_word_table[tag]['UNKNOWN'] = smoothing_param
        # tag_word_table[tag]['UNKNOWN'] = smoothing_param*appear_once[tag]
        
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

def get_hapax(tag_word_table):
    # word:tag, used to record unique hapax words
    # if a word appears once again, lookup this dict, find its previous tag, and delete the word from hapax set
    unique_hapax = dict() 
    # tag:[word], resulting hapax_set
    unique_hapax_list = []
    hapax_set = dict() 
    hapax_count = dict()
    for tag in tag_word_table.keys():
        hapax_list = []
        for word in tag_word_table[tag].keys():
            if tag_word_table[tag][word] == 1:
                if word not in unique_hapax:
                    unique_hapax[word] = tag
                    hapax_list.append(word)
                else:
                    # remove word from hapax_set if appeared once before
                    if word in hapax_set[unique_hapax[word]]:
                        hapax_set[unique_hapax[word]].remove(word)
        hapax_set[tag] = hapax_list
    
    for tag in hapax_set.keys():
        hapax_count[tag] = len(hapax_set[tag])
    
    return hapax_set, hapax_count

def get_suffix(hapax, cutoff):
    # target_tags = ['NOUN', 'ADJ', 'VERB']
    suffixes = dict()

    for tag in hapax.keys():
        suf2 = dict()
        for word in hapax[tag]:
            if word[-2:] not in suf2:
                suf2[word[-2:]] = 1
            else:
                suf2[word[-2:]] += 1
            """
            if word[-3:] not in suf3:
                suf3[word[-3:]] = 1
            else:
                suf3[word[-3:]] += 1
            if word[-4:] not in suf4:
                suf4[word[-4:]] = 1
            else:
                suf4[word[-4:]] += 1
            """

        suf_list = [suf for suf in suf2 if suf2[suf]>=cutoff]
        # suf_list += [suf for suf in suf3 if suf3[suf]>=cutoff]
        # suf_list += [suf for suf in suf4 if suf4[suf]>=cutoff]
        suffixes[tag] = suf_list

    return suffixes

def get_prefix(hapax, cutoff):
    #target_tags = ['NOUN', 'ADJ', 'VERB']
    prefixes = dict()
    
    for tag in hapax.keys():
        pre2, pre3, pre4 = dict(), dict(), dict()
        for word in hapax[tag]:
            if word[:2] not in pre2:
                pre2[word[:2]] = 1
            else:
                pre2[word[:2]] += 1
            if word[:3] not in pre3:
                pre3[word[:3]] = 1
            else:
                pre3[word[:3]] += 1
            if word[:4] not in pre4:
                pre4[word[:4]] = 1
            else:
                pre4[word[:4]] += 1
    
        pre_list = [pre for pre in pre2 if pre2[pre]>=cutoff]
        pre_list += [pre for pre in pre3 if pre3[pre]>=cutoff]
        pre_list += [pre for pre in pre4 if pre4[pre]>=cutoff]
        prefixes[tag] = pre_list

    return prefixes

def get_hapax_nops(hapax_set, suffixes, prefixes):
    hapax_nops = dict()
    hapax_nops_count = dict()

    for tag in hapax_set.keys():
        nops_list = []
        for word in hapax_set[tag]:
            has_ps = False
            for s in range(2, 5):
                if word[-s:] in suffixes[tag] or word[:s] in prefixes[tag]:
                    has_ps = True
            if not has_ps:
                nops_list.append(word)        
        hapax_nops[tag] = nops_list

    for tag in hapax_nops.keys():
        hapax_nops_count[tag] = len(hapax_nops[tag])
    
    return hapax_nops, hapax_nops_count

def get_ps_freq(hapax_set, prefixes, suffixes):
    prefix_freq = dict()
    suffix_freq = dict()
    
    for tag in prefixes:
        for pre in prefixes[tag]:
            if pre not in prefix_freq:
                prefix_freq[pre] = dict()
            prefix_freq[pre][tag] = 0

    for tag in suffixes:
        for suf in suffixes[tag]:
            if suf not in suffix_freq:
                suffix_freq[suf] = dict()
            suffix_freq[suf][tag] = 0

    for tag in hapax_set.keys():
        for word in hapax_set[tag]:
            for s in range(2, 5):
                if word[:s] in prefixes[tag]:
                    prefix_freq[word[:s]][tag] += 1
                if word[-s:] in suffixes[tag]:
                    suffix_freq[word[-s:]][tag] += 1 
    
    return prefix_freq, suffix_freq

def get_hapax_nops_prob(hapax_nops_count):
    hapax_nops_prob = hapax_nops_count.copy()
    for tag in hapax_nops_prob.keys():
        hapax_nops_prob[tag] += 0.0000001 

    sum_hapax_nops = sum(hapax_nops_prob.values())
    
    for tag in hapax_nops_prob.keys():
        hapax_nops_prob[tag] /= sum_hapax_nops
    
    return hapax_nops_prob


def get_ps_prob(pre_freq, suf_freq):
    pre_prob = pre_freq.copy()
    suf_prob = suf_freq.copy()

    for pre in pre_prob:
        sum_pre = sum(pre_prob[pre].values())
        for tag in pre_prob[pre]:
            pre_prob[pre][tag] /= sum_pre
    
    for suf in suf_prob:
        sum_suf = sum(suf_prob[suf].values())
        for tag in suf_prob[suf]:
            suf_prob[suf][tag] /= sum_suf

    return pre_prob, suf_prob

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_word_table = dict()
    trans_mtx = dict()
    smoothing_param_emis = 0.0001 
    smoothing_param_tran = 0.00001
    # cutoff freq for finding prefix and suffix
    cutoff_prefix = 200
    cutoff_suffix = 200
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
    
    # extract hapax_set, hapax_count from training data
    hapax_set, hapax_count = get_hapax(tag_word_table)
    
    # get prefixes and suffixes
    suffixes = get_suffix(hapax_set, cutoff_suffix)
    prefixes = get_prefix(hapax_set, cutoff_prefix)
    
    # print('suffix', suffixes)
    # print('prefix', prefixes)
    
    # words in hapax set without prefixes or suffixes
    hapax_nops, hapax_nops_count = get_hapax_nops(hapax_set, suffixes, prefixes)
    hapax_nops_prob = get_hapax_nops_prob(hapax_nops_count)
    
    # print('hapax_set:', hapax_nops)
    # print('hapax_nops:', hapax_nops_prob)
    pre_freq, suf_freq = get_ps_freq(hapax_set, prefixes, suffixes)
    pre_prob, suf_prob = get_ps_prob(pre_freq, suf_freq)

    # Laplace-smoothing/computing log probabilities for the tag-word table
    smooth_twTable(tag_word_table, smoothing_param_emis)
    logProb_emission(tag_word_table)

    # print('trans_mtx', trans_mtx)
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
                        #using prefixes/suffixes
                        scale_ps = hapax_nops_prob[tag]
                        """
                        if word[:2] in pre_prob:
                            HAS_PS = True
                            if tag in pre_prob[word[:2]]:
                                scale_ps = max(scale_ps, pre_prob[word[:2]][tag])
                        """
                        if len(suffixes[tag]) != 0:    
                            if word[-2:] in suffixes[tag]:
                                scale_ps = max(scale_ps, suf_prob[word[-2:]][tag])
                        
                        emis_prob = tag_word_table[tag]['UNKNOWN'] + np.log(scale_ps)
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