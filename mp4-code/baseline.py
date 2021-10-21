"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    freq_table = dict()
    tag_table = dict()
    res = []
    for sen in train:
        for word, tag in sen:
            if word == 'START' or word == 'END':
                continue
            if word not in freq_table:
                freq_table[word] = dict()
                freq_table[word][tag] = 1
            else:
                if tag not in freq_table[word]:
                    freq_table[word][tag] = 1
                else:
                    freq_table[word][tag] += 1 
            if tag not in tag_table:
                tag_table[tag] = 1
            else:
                tag_table[tag] += 1

    for sen in test:
        res_sen = []
        for word in sen:
            if word == 'START' or word == 'END':
                res_sen.append((word, word))
            else:
                if word not in freq_table:
                    tag = max(tag_table, key=tag_table.get)
                    res_sen.append((word, tag))
                else:
                    tag = max(freq_table[word], key=freq_table[word].get)
                    res_sen.append((word, tag))
        res.append(res_sen)

    return res