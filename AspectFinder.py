from nltk.corpus import product_reviews_1
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer

class AspectFinder:

    intext = [' '.join(s) for s in product_reviews_1.sents()]
    intextSents = []
    intextWords = []
    intextWordsTagged = []
    intextNouns = []
    intextNounsLemmatized = []


    def get_aspects(self):
        return product_reviews_1.features()


    def set_input_text(self, intext):
        self.intext = intext


    def tokenize_sentences(self):
        self.intextSents = product_reviews_1.sents()


    def tokenize_words(self):
            self.intextWords = product_reviews_1.sents()


    def postag_words(self):
        for sent in self.intextWords:
            self.intextWordsTagged.append(pos_tag(sent))


    def extract_nouns(self):
        for sent in self.intextWordsTagged:
            sentnouns = []
            for word in sent:
                if word[1]=='NN' or word[1]=='NNP' or word[1]=='NNPS' or word[1]=='NNS':
                    sentnouns.append(word[0])
            self.intextNouns.append(sentnouns)


    def lemmatize_nouns(self):
        stemmer = SnowballStemmer("english")
        for sent in self.intextNouns:
            sentlemmas = []
            for word in sent:
                sentlemmas.append(stemmer.stem(word))
            self.intextNounsLemmatized.append(sentlemmas)


    def compute_nouns(self):
        self.tokenize_sentences()
        self.tokenize_words()
        self.postag_words()
        self.extract_nouns()
        self.lemmatize_nouns()