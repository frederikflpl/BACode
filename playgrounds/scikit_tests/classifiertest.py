from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import playgrounds.scikit_tests.ReviewPreparator as revprep


def getData():
    data = revprep.prepareReviews()
    global traindata, traintargets, testdata, testtargets
    traindata = data[0]
    traintargets = data[1]
    testdata = data[2]
    testtargets = data[3]


def fit(classifier):
    classifier.fit(traindata, traintargets)


def fit_and_evaluate(classifier):
    fit(classifier)
    predicted = classifier.predict(testdata)
    print('+: ' + str(predicted.tolist().count('+')))
    print('-: ' + str(predicted.tolist().count('-')))
    print('+: ' + str(testtargets.count('+')))
    print('-: ' + str(testtargets.count('-')))
    print(np.mean(predicted == testtargets))
    print(metrics.classification_report(testtargets, predicted,
                                        target_names=['+', '-']))
    print(metrics.confusion_matrix(testtargets, predicted))
    print('I love this product. : ',classifier.predict(['I love this product.']))
    print('I hate this product. : ',classifier.predict(['I hate this product.']))


def optimizeParameters(classifier, parameters):
    gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(traindata, traintargets)
    print(gs_clf.predict(['God is love']))
    print(gs_clf.predict(['I love it.']))
    print(gs_clf.predict(['I hate it.']))
    for param_name in sorted(gs_clf.best_params_.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print(gs_clf.best_score_)
    predicted = gs_clf.predict(testdata)
    print('+: ' + str(predicted.tolist().count('+')))
    print('-: ' + str(predicted.tolist().count('-')))
    print('+: ' + str(testtargets.count('+')))
    print('-: ' + str(testtargets.count('-')))
    print(np.mean(predicted == testtargets))
    print(metrics.classification_report(testtargets, predicted,
                                        target_names=['+', '-']))
    print(metrics.confusion_matrix(testtargets, predicted))


if __name__ == '__main__':
    getData()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42,
                                                  n_jobs=-1)),
                         ])
    fit_and_evaluate(text_clf)
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
    optimizeParameters(text_clf,parameters)