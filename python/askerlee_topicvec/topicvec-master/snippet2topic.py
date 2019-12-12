import numpy as np
import getopt
import sys
import pdb
import os
from topicvecDir import topicvecDir
from utils import *

customStopwords = "based via using approach learning multi algorithm algorithms"

config = dict(  snip_filenames = None,
                short_name = None,
                unigramFilename = "top1grams-wiki.txt",
                word_vec_file = "25000-180000-500-BLK-8.0.vec",
                K = 20,
                N0 = 500,
                max_l = 5,
                init_l = 1,
                max_grad_norm = 0,
                # cap the sum of Em when updating topic embeddings
                # to avoid too big gradients
                grad_scale_Em_base = 2500,
                topW = 30,
                topTopicMassFracPrintThres = 0.1,
                alpha0 = 0.1,
                alpha1 = 0.1,
                iniDelta = 0.1,
                MAX_EM_ITERS = 100,
                topicDiff_tolerance = 2e-3,
                printTopics_iterNum = 10,
                zero_topic0 = True,
                useDrdtApprox = False,
                customStopwords = customStopwords,
                remove_stop = True,
                normalize_vecs = False,
                # shift all embeddings in a document, so that their average is 0
                rebase_vecs = True,
                rebase_norm_thres = 0.2,
                evalKmeans = False,
                verbose = 1,
                seed = 0
            )

def usage():
    print """topicvecDir.py [ -v vec_file -a alpha ... ] snip_file
Options:
  -k:  Number of topic embeddings to extract. Default: 20
  -v:  Existing embedding file of all words.
  -r:  Existing residual file of core words.
  -a:  Hyperparameter alpha. Default: 0.1.
  -i:  Number of iterations of the EM procedure. Default: 100
  -u:  Unigram file, to obtain unigram probs.
  -l:  Magnitude of topic embeddings.
  -A:  Append to the old log file.
  -s:  Seed the random number generator to x. Used to repeat experiments
  -n:  Nickname (short name) for the snip_file
"""

def getOptions():
    global config

    try:
        opts, args = getopt.getopt(sys.argv[1:],"k:v:i:u:l:s:n:Ah")
        if len(args) != 1:
            raise getopt.GetoptError("")
        config['snip_filename'] = args[0]
            
        for opt, arg in opts:
            if opt == '-k':
                config['K'] = int(arg)
            if opt == '-v':
                config['vec_file'] = arg
            if opt == '-a':
                config['alpha1'] = float(opt)
            if opt == '-i':
                config['MAX_EM_ITERS'] = int(arg)
            if opt == '-u':
                config['unigramFilename'] = arg
            if opt == '-l':
                config['max_l'] = int(arg)
            if opt == '-s':
                config['seed'] = int(arg)
            if opt == '-A':
                config['appendLogfile'] = True
            if opt == '-n':
                config['short_name'] = arg
            if opt == '-r':
                config['useDrdtApprox'] = True
            if opt == '-h':
                usage()
                sys.exit(0)

        basename = os.path.basename(args[0])
        if config['short_name']:
            config['logfilename'] = config['short_name']
        elif len(args) > 1:
            config['logfilename'] = "(%d)%s" %( len(args), basename )
        else:
            config['logfilename'] = basename

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    return config

def main():

    config = getOptions()
    snip_filename = config['snip_filename']
    snips_words = []
    snips_name = []
    
    with open(snip_filename) as DOC:
        snip_lines = []
        snipcount = 0
        snips_wc = 0
        for line in DOC:
            line = line.strip()
            if line:
                snip_lines.append(line)
            else:
                sniptext = " ".join(snip_lines)
                wordsInSentences, wc = extractSentenceWords(sniptext, remove_punc="iso-8859-1")
                snips_wc += wc
                snipcount += 1
                snips_words.append(wordsInSentences)
                snips_name.append( "%s-row%d" %(snip_filename, snipcount) )
                
    snipfile_avgwc = snips_wc * 1.0 / snipcount
    print "%d words extracted from %d snippets in '%s'. Avg %.1f words each row" %( snips_wc, 
                snipcount, snip_filename, snipfile_avgwc )
    
    topicvec = topicvecDir(**config)
    topicvec.setDocs( snips_words, snips_name )
    
    best_last_Ts, Em, docs_Em, Pi = topicvec.inference()

    basename = os.path.basename(config['logfilename'])
    basetrunk = os.path.splitext(basename)[0]

    best_it, best_T, best_loglike = best_last_Ts[0]
    save_matrix_as_text( basetrunk + "-em%d-best.topic.vec" %best_it, "topic", best_T  )

    if best_last_Ts[1]:
        last_it, last_T, last_loglike = best_last_Ts[1]
        save_matrix_as_text( basetrunk + "-em%d-last.topic.vec" %last_it, "topic", last_T  )

if __name__ == '__main__':
    main()
