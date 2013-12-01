import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk, os, logging, json, ConfigParser, codecs


class movie_sentiment:
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.read("senti_analysis.config")
        print "Karthik Chandrasekar"

        #File names:
        cur_dir = os.getcwd()
        rel_dir_path = self.config.get('GLOBAL', 'reviews_file_dir')
        self.reviews_file_dir = os.path.join(cur_dir, rel_dir_path)
        
        self.pos_rev_file_name = self.config.get('GLOBAL', 'pos_reviews_file_name') 
        self.pos_rev_file = os.path.join(self.reviews_file_dir, self.pos_rev_file_name)

        self.neg_rev_file_name = self.config.get('GLOBAL', 'neg_reviews_file_name') 
        self.neg_rev_file = os.path.join(self.reviews_file_dir, self.neg_rev_file_name)

        self.logger_file = os.path.join("OUTPUT", "senti_analysis.log")

        #Global ds
        self.pos_reviews_list = []
        self.neg_reviews_list = []


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

    def load_data(self):
        #Loading pos reviews
        for review in self.pos_rev_fd.readlines():
            self.pos_reviews_list.append(review)

        #Loading neg reviews
        for review in self.neg_rev_fd.readlines():
            self.neg_reviews_list.append(review)

    def close_files(self):
        self.pos_rev_fd.close()
        self.neg_rev_fd.close()

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

        train_features = pos_train_features + neg_train_features
        test_features = pos_test_features + neg_test_features


        classifier = NaiveBayesClassifier.train(train_features)
        print 'accuracy:', nltk.classify.util.accuracy(classifier, test_features)
        classifier.show_most_informative_features()

    def testing(self):
        pass


if __name__ == "__main__":
    ms = movie_sentiment()
    ms.run_main()
