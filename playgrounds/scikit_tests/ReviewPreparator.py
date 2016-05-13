from nltk.corpus import product_reviews_1
import AspectFinder

trainfeats = []
traintargets = []
testfeats = []
testtargets = []

def prepareReviews():
    aspects = AspectFinder.find_aspects()

    minus = [f for f in aspects if f[1][0] == "-"]
    plus = [f for f in aspects if f[1][0] == "+"]

    sentences = [' '.join(s) for s in product_reviews_1.sents()]

    minusfeatsann = [(s, '-') for s in sentences for f in minus if s.find(f[0]) != -1]
    plusfeatsann = [(s, '+') for s in sentences for f in plus if s.find(f[0]) != -1]

    if len(minusfeatsann)>len(plusfeatsann):
        minusfeatsann = minusfeatsann[:len(plusfeatsann)]
    else:
        plusfeatsann = plusfeatsann[:len(minusfeatsann)]

    minusfeats = [f[0] for f in minusfeatsann]
    plusfeats = [f[0] for f in plusfeatsann]

    minustargets = [f[1] for f in minusfeatsann]
    plustargets = [f[1] for f in plusfeatsann]

    minuscutoff = int(len(minusfeats) * 3 / 4)
    pluscutoff = int(len(plusfeats) * 3 / 4)

    trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
    traintargets = minustargets[:minuscutoff] + plustargets[:pluscutoff]
    testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
    testtargets = minustargets[minuscutoff:] + plustargets[pluscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
    return [trainfeats, traintargets, testfeats, testtargets]
def prepareReviewData():
    return trainfeats

def prepareReviewTargets():
    return traintargets

def prepareReviewTestData():
    return ['Hello']

def prepareReviewTestTargets():
    return ['Greeting']