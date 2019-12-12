from sklearn import svm, metrics
from sklearn.datasets import load_svmlight_file
import sys

# ����precision, recall, f1, accuracy
def getScores( true_classes, pred_classes, average):
    precision = metrics.precision_score( true_classes, pred_classes, average=average )
    recall = metrics.recall_score( true_classes, pred_classes, average=average )
    f1 = metrics.f1_score( true_classes, pred_classes, average=average )
    accuracy = metrics.accuracy_score( true_classes, pred_classes )
    return precision, recall, f1, accuracy

# ������: python classEval.py ������ �ļ�����(lda, bow�ȵ�)     
corpus = sys.argv[1]
filetype = sys.argv[2]
# (��ѡ)��ָ��ֻʹ��һ��������
# selected feature dimensions can be specified in the last argument as:
# 1-400 (starting from 1)
if len(sys.argv) > 3:
    dims = sys.argv[3].split("-")
    dims[0] = int(dims[0]) - 1
    dims[1] = int(dims[1])
else:
    dims = None

# ����ģ�壬�õ�training��test�ļ���        
if corpus == '20news':
    train_file = "20news-train-11314.svm-%s.txt" %filetype
    test_file = "20news-test-7532.svm-%s.txt" %filetype
else:
    train_file = "reuters-train-5770.svm-%s.txt" %filetype
    test_file = "reuters-test-2255.svm-%s.txt" %filetype

# ����training��test�ļ�������
train_features_sparse, true_train_classes = load_svmlight_file(train_file)
test_features_sparse, true_test_classes = load_svmlight_file(test_file)

# ȱʡ����Ϊϡ�����ת��Ϊ��ͨnumpy array
train_features = train_features_sparse.toarray()
test_features = test_features_sparse.toarray()

print "Train: %dx%d. Test: %dx%d" %( tuple( train_features.shape + test_features.shape ) )

if dims:
    train_features = train_features[ :, dims[0]:dims[1] ]
    test_features = test_features[ :, dims[0]:dims[1] ]
    print "Choose only features %d-%d" %( dims[0]+1, dims[1] )
else:
    train_features = train_features[ :, : ]
    test_features = test_features[ :, : ]

# ����SVM��L1����        
model = svm.LinearSVC(penalty='l1', dual=False)

# ��training�ļ���ѵ��
print "Training...",
model.fit( train_features, true_train_classes )
print "Done."

# ��test�ļ�����Ԥ��
pred_train_classes = model.predict( train_features )
pred_test_classes = model.predict( test_features )

# �㱨���
print metrics.classification_report(true_train_classes, pred_train_classes, digits=3)
print metrics.classification_report(true_test_classes, pred_test_classes, digits=3)

for average in ['micro', 'macro']:
    train_precision, train_recall, train_f1, train_acc = getScores( true_train_classes, pred_train_classes, average )
    print "Train Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %( average, 
                        train_precision, train_recall, train_f1, train_acc )
    
    test_precision, test_recall, test_f1, test_acc = getScores( true_test_classes, pred_test_classes, average )
    print "Test Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %(  average, 
                        test_precision, test_recall, test_f1, test_acc )
