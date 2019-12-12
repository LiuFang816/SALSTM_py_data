from optparse import OptionParser
import json, utils, easylstm, os, pickle, time

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="data/PTB_SD_3_3_0/train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="data/PTB_SD_3_3_0/dev.conll")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="easyfirst.model")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--k", type="int", dest="window", default=1)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--cnn-seed", type="int", dest="seed", default=0)


    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print 'Initializing Hierarchical Tree LSTM parser:'
        parser = easylstm.EasyFirstLSTM(words, pos, rels, w2i, stored_opt) 

        parser.Load(options.model)
        tespath = os.path.join(options.output, 'test_pred.conll')
        
        ts = time.time()
        test_res = list(parser.Predict(options.conll_test))
        te = time.time()
        print 'Finished predicting test.', te-ts, 'seconds.'
        utils.write_conll(tespath, test_res)

        os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt')
    else:
        print 'Preparing vocab'
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing Hierarchical Tree LSTM parser:'
        parser = easylstm.EasyFirstLSTM(words, pos, rels, w2i, options) 

        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train)
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch+1) + '.conll')
            utils.write_conll(devpath, parser.Predict(options.conll_dev))
            parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch+1)))
            os.system('perl src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')

