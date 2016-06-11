from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics, svm
from sklearn.grid_search import GridSearchCV
from sklearn.semi_supervised import label_propagation
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB

import playgrounds.scikit_tests.ReviewPreparator as revprep

# accuracy 0.706989247312
# accuracy 0.798387096774 with best features

if __name__ == '__main__':
    data = revprep.prepareReviews()
    traindata = data[0]
    traintargets = data[1]
    testdata = data[2]
    testtargets = data[3]
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42,
                                                  n_jobs=-1)),
                         ])
    _ = text_clf.fit(traindata, traintargets)
    predicted = text_clf.predict(testdata)
    print('+: '+str(predicted.tolist().count('+')))
    print('-: '+str(predicted.tolist().count('-')))
    print('+: '+str(testtargets.count('+')))
    print('-: '+str(testtargets.count('-')))
    print(np.mean(predicted == testtargets))
    print(metrics.classification_report(testtargets, predicted,
                                        target_names=['+', '-']))
    print(metrics.confusion_matrix(testtargets, predicted))
    print(text_clf.predict(['I love this product.']))
    print(text_clf.predict(['I hate this product.']))
    # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    #               'tfidf__use_idf': (True, False),
    #               'clf__alpha': (1e-2, 1e-3),
    #               }
    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(traindata, traintargets)
    # print(gs_clf.predict(['God is love']))
    # print(gs_clf.predict(['I love it.']))
    # print(gs_clf.predict(['I hate it.']))
    # for param_name in sorted(gs_clf.best_params_.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    # print(gs_clf.best_score_)
    # predicted = gs_clf.predict(testdata)
    # print('+: ' + str(predicted.tolist().count('+')))
    # print('-: ' + str(predicted.tolist().count('-')))
    # print('+: ' + str(testtargets.count('+')))
    # print('-: ' + str(testtargets.count('-')))
    # print(np.mean(predicted == testtargets))
    # print(metrics.classification_report(testtargets, predicted,
    #                                     target_names=['+', '-']))
    # print(metrics.confusion_matrix(testtargets, predicted))