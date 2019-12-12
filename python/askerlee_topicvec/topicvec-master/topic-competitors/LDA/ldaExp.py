import sys
import pdb
import os
import getopt
import time
import gensim
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from corpusLoader import *

# �����Ͽ���ӳ�䵽���غ�����֮�����
corpus2loader = { '20news': load_20news, 'reuters': load_reuters }

def usage():
    print """Usage: ldaExp.py corpus_name"""

corpusName = sys.argv[1]
# ���غ���
loader = corpus2loader[corpusName]

# 20news���ĵ��������������Щ���������������һЩ
if corpusName == "20news":
    topicNum = 100
else:
    topicNum = 50

# �������϶��ѷֳ�train��test���ϡ�����ֱ���
setNames = [ 'train', 'test' ]
basenames = []
subcorpora = []
corpus = []
word2id = {}
id2word = {}
maxWID = 0

for setName in setNames:
    print "Process set '%s':" %setName
    
    # �������ϵ�train��test�Ӽ��������Ծ���Ϊ��λ���� orig_docs_words����ȷ��� orig_docs_cat
    setDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
            cats_docNames, category_names = loader(setName)
    # �ļ���ǰ׺
    basename = "%s-%s-%d" %( corpusName, setName, setDocNum )
    basenames.append(basename)
    
    # ��ǰѭ��������������Ӽ�����һ��list��list��ÿ�����listԪ�ض�Ӧһ���ĵ�
    # ÿ���ڲ�listΪһ�� (word_id, frequency) ��pair
    # ���ָ�ʽ��gensim�ı�׼�����ʽ
    subcorpus = []
    
    # ����ԭʼ�ı����Թ��˲鿴
    orig_filename = "%s.orig.txt" %basename
    ORIG = open( orig_filename, "w" )

    # ÿ�� wordsInSentences ��Ӧһ���ĵ�
    # ÿ�� wordsInSentences ����������ɣ�ÿ��������һ��list of words
    for wordsInSentences in orig_docs_words:
        # ͳ�Ƶ�ǰ�ĵ���ÿ���ʵ�Ƶ��
        doc_wid2freq = {}
        # ѭ��ȡ��ǰ�ĵ���һ������
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()
                ORIG.write( "%s " %w )
                
                # ���w����word2idӳ����У�ӳ���wid
                if w in word2id:
                    wid = word2id[w]
                # ���򣬰�w����ӳ�����ӳ�����wid
                else:
                    wid = maxWID
                    word2id[w] = maxWID
                    id2word[maxWID] = w
                    maxWID += 1
                
                # ͳ�� wid ��Ƶ��
                if wid in doc_wid2freq:
                    doc_wid2freq[wid] += 1
                else:
                    doc_wid2freq[wid] = 1
                    
        ORIG.write("\n")
        # ���ĵ��г��ֵ�wid��id��С����
        sorted_wids = sorted( doc_wid2freq.keys() )
        doc_pairs = []
        # �� (wid, frequency) �Ķ�׷�ӵ���ǰ�ĵ���list��
        for wid in sorted_wids:
            doc_pairs.append( (wid, doc_wid2freq[wid]) )
        
        # ��ǰ�ĵ���list�Ѿ���ȫ���ɣ���������subcorpus���������Ӽ���list��    
        subcorpus.append(doc_pairs)

    ORIG.close()
    print "%d original docs saved in '%s'" %( setDocNum, orig_filename )

    # �����������Ӽ�list��֮ǰ��list�ϲ����õ�һ������train��test���ϵ������ĵ��ļ���
    corpus += subcorpus
    # �����train��test���Ϸֿ��ţ�֮���Ѳ�ͬ���ϵ�ÿ���ĵ��ġ�doc-topic����������ɲ�ͬ�ļ�
    subcorpora.append( (subcorpus, orig_docs_cat) )
    
print "Training LDA..."
startTime = time.time()
# LDAѵ����ʱ���ǰ�train��test��һ��ѵ����(���ϸ�İ취Ӧ����ֻ��train������ѵ��)
lda = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=topicNum, passes=20 )
endTime = time.time()
print "Finished in %.1f seconds" %( endTime - startTime )

for i in xrange(2):
    lda_filename = "%s.svm-lda.txt" %basenames[i]
    LDA = open( lda_filename, "w" )
    print "Saving topic proportions into '%s'..." %lda_filename
    
    # �ó�һ�������Ӽ� (train����test)
    subcorpus, labels = subcorpora[i]

    # �����Ӽ���ÿ���ĵ�
    for d, doc_pairs in enumerate(subcorpus):
        label = labels[d]
        # �ѵ�ǰ�ĵ���Ϊ���룬��ѵ���õ�LDAģ����doc-topic������
        topic_props = lda.get_document_topics( doc_pairs, minimum_probability=0.001 )
        LDA.write( "%d" %label )
        # ��K�����������K��������svmlight��ʽ
        for k, prop in topic_props:
            LDA.write(" %d:%.3f" %(k, prop) )
        LDA.write("\n")
    LDA.close()
    print "%d docs saved" %len(subcorpus)
