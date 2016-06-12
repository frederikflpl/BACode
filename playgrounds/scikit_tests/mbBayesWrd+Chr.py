from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics, svm
from sklearn.grid_search import GridSearchCV
from sklearn.semi_supervised import label_propagation
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack

import playgrounds.scikit_tests.ReviewPreparator as revprep

# accuracy 0.739247311828
# accuracy 0.806451612903 with feature selection
# accuracy 0.798387096774 with feature selection and word+char features

if __name__ == '__main__':
    data = revprep.prepareReviews()
    traindata = data[0]
    traintargets = data[1]
    testdata = data[2]
    testtargets = data[3]
    cvec1 = CountVectorizer(ngram_range=(1,1),analyzer="word")
    cvec2 = CountVectorizer(ngram_range=(1,1),analyzer="char")
    a_new_counts = cvec1.fit_transform(traindata)
    b_new_counts = cvec2.fit_transform(traindata)
    c_new_counts = hstack((a_new_counts, b_new_counts),format="csr")
    clf = MultinomialNB(alpha=0.3).fit(c_new_counts,traintargets)
    a_new_tcounts = cvec1.transform(testdata)
    b_new_tcounts = cvec2.transform(testdata)
    c_new_tcounts = hstack((a_new_tcounts, b_new_tcounts),format="csr")
    predicted = clf.predict(c_new_tcounts)
    print('+: '+str(predicted.tolist().count('+')))
    print('-: '+str(predicted.tolist().count('-')))
    print('+: '+str(testtargets.count('+')))
    print('-: '+str(testtargets.count('-')))
    print(np.mean(predicted == testtargets))
    print(metrics.classification_report(testtargets, predicted,
                                        target_names=['+', '-']))
    print(metrics.confusion_matrix(testtargets, predicted))
    a_new_ccounts = cvec1.transform(['God is love','I love this product.','I hate this product.'])
    b_new_ccounts = cvec2.transform(['God is love','I love this product.','I hate this product.'])
    c_new_ccounts = hstack((a_new_ccounts, b_new_ccounts),format="csr")
    print(clf.predict(c_new_ccounts))

    # parameters = {'alpha': (0.3,0.5,1),
    #               'fit_prior': (True, False),
    #               }
    # gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(c_new_counts, traintargets)
    # print(gs_clf.predict(c_new_ccounts))
    # for param_name in sorted(gs_clf.best_params_.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    # print(gs_clf.best_score_)
    # predicted = gs_clf.predict(c_new_tcounts)
    # print('+: ' + str(predicted.tolist().count('+')))
    # print('-: ' + str(predicted.tolist().count('-')))
    # print('+: ' + str(testtargets.count('+')))
    # print('-: ' + str(testtargets.count('-')))
    # print(np.mean(predicted == testtargets))
    # print(metrics.classification_report(testtargets, predicted,
    #                                     target_names=['+', '-']))
    # print(metrics.confusion_matrix(testtargets, predicted))