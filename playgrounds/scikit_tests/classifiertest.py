from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics, svm
from sklearn.grid_search import GridSearchCV
import playgrounds.scikit_tests.ReviewPreparator as revprep


def getData(neutral=False):
    data = revprep.prepareReviews(neutral)
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
                                        target_names=['+', '-', '~']))
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
    classifiers = [
        (
            "SGDClassifier",
            (
                Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                          ('tfidf', TfidfTransformer(use_idf=True)),
                          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42,
                                                n_jobs=-1)),
                          ]),
                {'vect__ngram_range': [(1, 1), (1, 2)],
                 'tfidf__use_idf': (True, False),
                 'clf__alpha': (1e-2, 1e-3),
                 }
            )

        ),
        (
            "BernoulliNB",
            (
                Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), binary=True)),
                          ('clf', BernoulliNB(alpha=1, fit_prior=True)),
                          ]),
                {'vect__ngram_range': [(1, 1), (1, 2)],
                 'clf__alpha': (1, 1.5, 2),
                 'clf__fit_prior': (True, False),
                 }
            )

        ),
        (
            "LinearSVC",
            (
                Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                          ('tfidf', TfidfTransformer(use_idf=True)),
                          ('clf', svm.LinearSVC(C=1, loss='hinge', max_iter=10, tol=1e-5)),
                          ]),
                {'vect__ngram_range': [(1, 1), (1, 2)],
                 'clf__C': (1e-1, 1, 1.5),
                 'clf__loss': ("hinge", "squared_hinge"),
                 'clf__tol': (1e-5, 1e-4, 1e-3),
                 'clf__max_iter': (10, 100, 1000),
                 }
            )

        ),
        (
            "MultinomialNB with word and character features",
            (
                Pipeline([('vect', FeatureUnion([("word", CountVectorizer(ngram_range=(1, 2), analyzer="word")),
                                                 ("char", CountVectorizer(ngram_range=(1, 2), analyzer="char", ))])),
                          ('clf', MultinomialNB(alpha=1, fit_prior=True)),
                          ]),
                {'vect__word__ngram_range': [(1, 1), (1, 2)],
                 'vect__char__ngram_range': [(1, 1), (1, 2)],
                 'clf__alpha': (1, 1.5, 2),
                 'clf__fit_prior': (True, False),
                 }
            )

        ),
        (
            "MultinomialNB",
            (
                Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                          ('clf', MultinomialNB(alpha=1, fit_prior=True)),
                          ]),
                {'vect__ngram_range': [(1, 1), (1, 2)],
                 'clf__alpha': (1, 1.5, 2),
                 'clf__fit_prior': (True, False),
                }
            )
        )
    ]
    getData()
    for classifier in classifiers:
        print("\n\n",classifier[0])
        text_clf = classifier[1][0]
        parameters = classifier[1][1]
        fit_and_evaluate(text_clf)
        #optimizeParameters(text_clf,parameters)