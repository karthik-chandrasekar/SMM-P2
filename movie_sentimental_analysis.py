import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk, os, logging, json, ConfigParser, codecs
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report

class movie_sentiment:
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.read("senti_analysis.config")

        #File names:
        cur_dir = os.getcwd()
        rel_dir_path = self.config.get('GLOBAL', 'reviews_file_dir')

        self.reviews_file_dir = os.path.join(cur_dir, rel_dir_path)
        
        self.d2_pos_reviews_file_dir = os.path.join(cur_dir, self.config.get('GLOBAL', 'pos_reviews_dir'))
        self.d2_neg_reviews_file_dir = os.path.join(cur_dir, self.config.get('GLOBAL', 'neg_reviews_dir'))

        self.pos_rev_file_name = self.config.get('GLOBAL', 'pos_reviews_file_name') 
        self.pos_rev_file = os.path.join(self.reviews_file_dir, self.pos_rev_file_name)

        self.neg_rev_file_name = self.config.get('GLOBAL', 'neg_reviews_file_name') 
        self.neg_rev_file = os.path.join(self.reviews_file_dir, self.neg_rev_file_name)

        self.logger_file = os.path.join("OUTPUT", "senti_analysis.log")

        #Dataset 2

        self.bow_file_dir = self.config.get('GLOBAL','bag_of_words_file_dir')
        self.bow_file_name = self.config.get('GLOBAL', 'bag_of_words_file_name')
        self.bow_file = os.path.join(self.bow_file_dir, self.bow_file_name)

        self.train_feat_file_dir = self.config.get('GLOBAL', 'feat_file_dir')
        self.train_feat_file_name = self.config.get('GLOBAL', 'feat_file_name')
        self.train_file = os.path.join(self.train_feat_file_dir, self.train_feat_file_name)

        self.test_feat_file_dir = self.config.get('GLOBAL', 'test_feat_file_dir')
        self.test_feat_file_name = self.config.get('GLOBAL', 'test_feat_file_name')
        self.test_file = os.path.join(self.test_feat_file_dir, self.test_feat_file_name)

        #Global ds
        self.pos_reviews_list = []
        self.neg_reviews_list = []
        self.bow_dict = {}

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")

    def run_main(self):
        self.preprocessing()
        self.feature_extraction()
        self.classification()
        self.testing()

    def preprocessing(self):
        self.initialize_logger()
        self.open_files()
        self.load_data()
        self.close_files()        

    def open_files(self):
        self.pos_rev_fd = open(self.pos_rev_file, 'r')
        self.neg_rev_fd = open(self.neg_rev_file, 'r')
        self.bow_fd = open(self.bow_file, 'r')
        self.train_file_fd = open(self.train_file, 'r')
        self.test_file_fd = open(self.test_file, 'r')  

    def load_data(self):
        self.load_bow()
        self.load_reviews()

    def load_bow(self):
        counter = 0

        for word in self.bow_fd.readlines():
            self.bow_dict[counter] = word and word.strip()     
            counter += 1 

    def load_reviews(self):
        #Loading pos reviews
        for review in self.pos_rev_fd.readlines():
            self.pos_reviews_list.append(review)

        #Loading neg reviews
        for review in self.neg_rev_fd.readlines():
            self.neg_reviews_list.append(review)

        #self.load_dataset_two()
        #self.load_d2_reviews()


    def load_d2_reviews(self):
        
        d2_pos_reviews = []
        d2_neg_reviews = []       

        pos_files = os.listdir(self.d2_pos_reviews_file_dir)
        neg_files = os.listdir(self.d2_neg_reviews_file_dir)

        for pos_file in pos_files:
            pos_filename = os.path.join(self.d2_pos_reviews_file_dir, pos_file)
            pos_fd = open(pos_filename, 'r')
            for lines in pos_fd.readlines():
                d2_pos_reviews.append(lines)


        for neg_file in neg_files:
            neg_filename = os.path.join(self.d2_neg_reviews_file_dir, neg_file)
            neg_fd = open(neg_filename, 'r')
            for lines in neg_fd.readlines():
                d2_neg_reviews.append(lines)
    

        self.pos_reviews_list.extend(d2_pos_reviews[:1000])
        self.neg_reviews_list.extend(d2_neg_reviews[:5000])


    def load_dataset_two(self):
        d2_pos_reviews_list = []
        d2_neg_reviews_list = []        


        for review in self.train_file_fd.readlines():
            label = review[0]
            if int(label) >= 7:
                kv_list = review.split(" ")[1:]
                sent = ""                       
                for kv in kv_list:
                    sent = sent + " " + self.bow_dict.get(int(kv.split(":")[0]))
                d2_pos_reviews_list.append(sent)
    

            if int(label) <= 4:
                kv_list = review.split(" ")[1:]
                sent=""
                for kv in kv_list:
                    sent = sent + " " + self.bow_dict.get(int(kv.split(":")[0]))
                d2_neg_reviews_list.append(sent)            

        self.pos_reviews_list.extend(d2_pos_reviews_list)
        self.neg_reviews_list.extend(d2_neg_reviews_list)

    def close_files(self):
        self.pos_rev_fd.close()
        self.neg_rev_fd.close()
        self.bow_fd.close()
        self.train_file_fd.close()
        self.test_file_fd.close()  

    def feature_selection(self, features_list):
        selected_feat_list = []
        for feat in features_list:
            selected_feat_list.append((feat, True))
        return dict(selected_feat_list)
       
    def feature_extraction(self):
        self.pos_feat_extraction()
        self.neg_feat_extraction()


    def pos_feat_extraction(self):
        self.selected_pos_feats = []

        #Select positive features
        for review in self.pos_reviews_list:
            review_words = review.split(" ")
            selected_review_words = self.feature_selection(review_words)
            self.selected_pos_feats.append((selected_review_words, 'pos'))
    

    def neg_feat_extraction(self):
        self.selected_neg_feats = []

        #Selecte negative features
        for review in self.neg_reviews_list:
            review_words = review.split(" ")
            selected_review_words = self.feature_selection(review_words) 
            self.selected_neg_feats.append((selected_review_words, 'neg'))

    def classification(self):
       
        pos_train_features = self.selected_pos_feats[:4000]
        neg_train_features = self.selected_neg_feats[:4000]        
 
        pos_test_features = self.selected_pos_feats[4000:]
        neg_test_features = self.selected_neg_feats[4000:]

        self.train_features = pos_train_features + neg_train_features
        self.test_features = pos_test_features + neg_test_features
        
        #NaiveBayesClassfication
        self.NaiveBayesClassification(self.train_features, self.test_features)
        
        #Support vecotr machine
        self.SVMClassification(self.train_features, self.test_features)

    def NaiveBayesClassification(self, train_features, test_features):

        #Training
        self.nb_classifier = NaiveBayesClassifier.train(train_features)

        #Testing
        print 'accuracy:', nltk.classify.util.accuracy(self.nb_classifier, test_features)
        self.nb_classifier.show_most_informative_features()

    def SVMClassification(self, train_features, test_features):
        test_feat_list = []
        test_feat_labels_list = []        

        #Training
        self.svm_classifier = SklearnClassifier(LinearSVC()) 
        self.svm_classifier.train(train_features)
        

        #Testing
        for test_feat in test_features:
            test_feat_list.append(test_feat[0])
            test_feat_labels_list.append(test_feat[1])            

        svm_test = self.svm_classifier.batch_classify(test_feat_list)
        
        print "SVM Classification"
        print classification_report(test_feat_labels_list, svm_test, labels=['pos','neg'], target_names=['pos', 'neg'])

    def testing(self):

        print "Naive Bayes \n"

        #Naive bayes
        actual_pol_dict, predicted_pol_dict = self.load_test_and_predicted_values(self.nb_classifier)
        pos_precision, neg_precision = self.find_precision(actual_pol_dict, predicted_pol_dict)
        pos_recall, neg_recall = self.find_recall(actual_pol_dict, predicted_pol_dict)
        self.find_fmeasure(pos_precision, neg_precision, pos_recall, neg_recall)


        print " Support vector machine \n"

        #Support Vector Machine
        actual_pol_dict, predicted_pol_dict = self.load_test_and_predicted_values(self.svm_classifier)
        pos_precision, neg_precision = self.find_precision(actual_pol_dict, predicted_pol_dict)
        pos_recall, neg_recall = self.find_recall(actual_pol_dict, predicted_pol_dict)
        self.find_fmeasure(pos_precision, neg_precision, pos_recall, neg_recall)
        

    def load_test_and_predicted_values(self, classifier):
        #Find the precision and recall
        actual_polarity_dict = {}
        predicted_polarity_dict = {}

        for i, (features, label) in enumerate(self.test_features):
            actual_polarity_dict.setdefault(label, set()).add(i)
            predicted_polarity = classifier.classify(features)
            predicted_polarity_dict.setdefault(predicted_polarity, set()).add(i)

        return (actual_polarity_dict, predicted_polarity_dict)


    def find_precision(self, actual_polarity_dict, predicted_polarity_dict):
        pos_precision = self.pos_precision(actual_polarity_dict, predicted_polarity_dict)
        neg_precision = self.neg_precision(actual_polarity_dict, predicted_polarity_dict)   
        return (pos_precision, neg_precision)

    def pos_precision(self, actual_polarity_dict, predicted_polarity_dict):
        pos_val_precision = nltk.metrics.precision(actual_polarity_dict['pos'], predicted_polarity_dict['pos'])
        print "Pos values preicsiion %s" % (pos_val_precision)
        return pos_val_precision

    def neg_precision(self, actual_polarity_dict, predicted_polarity_dict):
        neg_val_precision = nltk.metrics.precision(actual_polarity_dict['neg'], predicted_polarity_dict['neg'])
        print "Neg values preicsiion %s" % (neg_val_precision)
        return neg_val_precision

    def find_recall(self, actual_polarity_dict, predicted_polarity_dict):
        pos_recall = self.pos_recall(actual_polarity_dict, predicted_polarity_dict)
        neg_recall = self.neg_recall(actual_polarity_dict, predicted_polarity_dict)
        return (pos_recall, neg_recall)

    def pos_recall(self, actual_polarity_dict, predicted_polarity_dict):
        pos_val_recall = nltk.metrics.recall(actual_polarity_dict['pos'], predicted_polarity_dict['pos'])
        print "Pos values recall %s" % (pos_val_recall) 
        return pos_val_recall

    def neg_recall(self, actual_polarity_dict, predicted_polarity_dict):
        neg_val_recall = nltk.metrics.recall(actual_polarity_dict['neg'], predicted_polarity_dict['neg'])
        print "Neg values recall %s" % (neg_val_recall) 
        return neg_val_recall

    def find_fmeasure(self, pos_precision, neg_precision, pos_recall, neg_recall):
        self.pos_fmeasure(pos_precision, pos_recall)
        self.neg_fmeasure(neg_precision, neg_recall)

    def pos_fmeasure(self, pos_precision, pos_recall):
        pos_fmeasure_val = 2 * (pos_precision * pos_recall) / float(pos_precision + pos_recall)          
        print "F measure for pos val %s" % (pos_fmeasure_val)           
 
    def neg_fmeasure(self, neg_precision, neg_recall):
        neg_fmeasure_val = 2 * (neg_precision * neg_recall) / float(neg_precision + neg_recall)          
        print "F measure for neg val %s" % (neg_fmeasure_val)

if __name__ == "__main__":
    ms = movie_sentiment()
    ms.run_main()
