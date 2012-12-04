__author__ = 'samwing'
import imaplib
import email
import re
import time
import os

import lxml.html

from collections import defaultdict

from functools import reduce
from operator import add

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer, PorterStemmer

from nlp.feature_extractor import get_LIWC
from MSOL import msol

from WNExtension.domains import WN

def combine_dicts(*dictionaries):
    return dict(reduce(add, map(lambda dictionary: dictionary.items(), dictionaries)))

def prepend_key(dictionary, string):
    return {'{}_{}'.format(string, key): value for (key, value) in dictionary.iteritems()}

def build_features(body, subject, stemmer=None, wn=None, features=[]):#'bow_disc']): #'bow_disc', 'liwc'
    all_bow = dict()
    body_bow = dict()
    subj_bow = dict()
    stemmed_body = stemmer.stem(body)
    stemmed_subj = stemmer.stem(subject)
    tokens_body = wordpunct_tokenize(stemmed_body)
    tokens_subj = wordpunct_tokenize(stemmed_subj)
    all_bow = {'bow_{}'.format(token): True for token in tokens_body + tokens_subj}
    if 'bow_desc' in features:
        body_bow = {'body_{}'.format(token): True for token in tokens_body}
        subj_bow = {'subject_{}'.format(token): True for token in tokens_subj}
    polarity = defaultdict(int)
    subj_liwc = dict()
    body_liwc = dict()
    if 'liwc' in features:
        get_LIWC(subj_liwc, stemmed_subj)
        get_LIWC(body_liwc, stemmed_body)
    if 'polarity' in features:
       for token in tokens_subj + tokens_body:
           result = msol.lookup(token)
           if result == 'negative' or result == 'positive':
               polarity[result] += 1
    all_wn = dict()
    if 'wn' in features:
        for token in tokens_body + tokens_subj:
            result = wn.lookup(token)
            if len(result) > 0:
                for element in result:
                    all_wn['WN_{}'.format(element)] = True
    return combine_dicts(all_bow,
                         subj_bow,
                         body_bow,
                         all_wn,
                         prepend_key(subj_liwc, 'SUBJLIWC'),
                         prepend_key(body_liwc, 'BODYLIWC'),
                         polarity,
                        )

def create_training_instances(vectorizer, stemmer=None, features=[]):
    directory = 'bare'
    spam_message = re.compile(r'spmsg')
    subject = re.compile(r'Subject:\s*(.*)')
    instances = list()
    targets = list()
    t0 = time.time()
    print 'Building Features'
    wn = WN()
    for folder in os.listdir(directory):
        for filename in os.listdir('{}/{}'.format(directory, folder)):
            if spam_message.search(filename):
                targets.append(True)
            else:
                targets.append(False)
            _h, _b = open('{}/{}/{}'.format(directory, folder, filename)).read().split('\n\n')
            _s = subject.match(_h).group(1).lower()
            instances.append(build_features(_b, _s, stemmer, wn=wn, features=features))
        #if True in targets: break
    print 'Finished building features in {}s'.format(time.time() - t0)
    return vectorizer.fit_transform(instances).toarray(), targets

def train_classifier(classifier, instances, targets):
    print 'Training classifier'
    t0 = time.time()
    classifier.fit(instances, targets)
    print 'Finished Training Classifier in {}s'.format(time.time() - t0)
    return classifier

def write_email(message, spam=False):
    messageid, message = message
    directory = 'emails'
    subdir = 'spam' if spam else 'ham'
    fd = open('{}/{}/{}'.format(directory, subdir, messageid), 'wb')

def get_messages(spam_classifier, vectorizer):
    email_client = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    username = raw_input('Enter your username:')
    password = raw_input('Password:')
    email_client.login(user=username, password=password)
    email_client.select('inbox')
    result, data = email_client.search(None, 'ALL')
    for messageid in data[0].split():
        raw_data = email_client.fetch(messageid, '(RFC822)')
        email_message = email.message_from_string(raw_data[1][0][1])
        content_type = email_message.get_content_maintype()
        message = ''
        try:
            if content_type == 'multipart':
                for part in email_message.get_payload():
                    if part.get_content_maintype() == 'text':
                        message += '\n' + strip_html(part.get_payload()).encode('UTF-8')
            elif content_type == 'text':
                message += strip_html(email_message.get_payload().encode('UTF-8'))
            vector = build_features(message, email_message['Subject'].encode('UTF-8'))
            vector = vectorizer.transform(vector).toarray()
            prediction, = spam_classifier.predict(vector)

            print prediction, email_message['Subject'][:60]
        except UnicodeDecodeError:
            print 'Err:', email_message['Subject']

def strip_html(message):
    try:
        message = lxml.html.fromstring(message)
        message = ' '.join(message.xpath('text()'))
        return re.subn(r' +', ' ', message)[0]
    except lxml.etree.XMLSyntaxError:
        return message
    except IndexError:
        return message

def test_classifier(classifier, instances, targets, metric):
    from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support
    from sklearn.cross_validation import cross_val_score, ShuffleSplit
    cv = ShuffleSplit(len(instances), n_iterations=10, test_size=.2, random_state=0)
    t0 = time.time()
    print 'Testing the classifier with cross validation'
    scores = cross_val_score(classifier, instances, targets, cv=3, score_func=metric)#precision_recall_fscore_support)#recall_score)#f1_score)
    print 'Finishing testing in {}s'.format(time.time() - t0)
    return scores.mean()

def save_data(instances, targets, filename='precomputed_'):
    import pickle
    t0 = time.time()
    print 'Saving Data'
    pickle.dump(instances, open(filename +'instances', 'wb'))
    pickle.dump(targets, open(filename +'targets', 'wb'))
    print 'Took {}s to save data'.format(time.time() - t0)

def load_data(filename='precomputed_'):
    import pickle
    instances = pickle.load(open(filename+'instances'))
    targets = pickle.load(open(filename + 'targets'))
    return instances, targets

def run_experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectFpr, f_classif, chi2
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import precision_score, recall_score
    from itertools import combinations
    anova_filter = SelectFpr(f_classif)
    fpr_filter = SelectFpr(chi2)
    vectorizer = DictVectorizer()
    stemmer = SnowballStemmer('english')

    feature_set = ['bow_desc', 'wn', 'polarity', 'liwc']
    classifiers = [('naivebayes', BernoulliNB()), ('svmlinear', SVC(kernel='linear')), ('logistic', LogisticRegression())]
    for (_name, _classifier) in classifiers:
        for nr in xrange(len(feature_set) + 1):
            for features in combinations(feature_set, nr):
                classifier = Pipeline([('chi2', fpr_filter), (_name, _classifier)])
                instances, targets = create_training_instances(vectorizer=vectorizer, stemmer=stemmer, features=features)
                with open('results/{}__{}.txt'.format(_name, '_'.join(features)), 'wb') as fd:
                    for metric_name, metric_func in [('precision', precision_score), ('recall', recall_score)]:
                        fd.write('{}:'.format(metric_name))
                        result = test_classifier(classifier=classifier, instances=instances, targets=targets, metric=metric_func)
                        fd.write('{}\n'.format(result))
                        print '{}:{}'.format(metric_name, result)
                    del instances; del targets
                    #classifier = train_classifier(classifier=classifier, instances=instances, targets=targets)
                    #get_messages(spam_classifier=classifier, vectorizer=vectorizer)

if __name__ == '__main__':
    run_experiment() #used in the paper for generating the optimal model
    """
    #uncomment this block of text to run the spam filter on an actual gmail email
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_selection import SelectFpr, chi2
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import BernoulliNB

    vectorizer = DictVectorizer()
    stemmer = SnowballStemmer('english')
    features = [] #use bag of words it performed the best
    instances, targets = create_training_instances(vectorizer=vectorizer, stemmer=stemmer, features=features)
    classifier = Pipeline([('chi2', SelectFpr(chi2)), ('nb', BernoulliNB())])
    classifier = train_classifer(classifier, instances, targets)

    get_messages(classifier, vectorizer) #will try to open your gmail email asking for credentials
    """
