import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk, os, logging, json, ConfigParser, codecs


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

        self.load_dataset_two()
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

        self.classifier = NaiveBayesClassifier.train(self.train_features)
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.test_features)
        self.classifier.show_most_informative_features()

    def testing(self):
        self.load_test_and_predicted_values()
        self.find_precision()
        self.find_recall()
        self.find_fmeasure()

    def load_test_and_predicted_values(self):
        #Find the precision and recall
        self.actual_polarity_dict = {}
        self.predicted_polarity_dict = {}

        for i, (features, label) in enumerate(self.test_features):
            self.actual_polarity_dict.setdefault(label, set()).add(i)
            predicted_polarity = self.classifier.classify(features)
            self.predicted_polarity_dict.setdefault(predicted_polarity, set()).add(i)

    def find_precision(self):
        self.pos_precision()
        self.neg_precision()   

    def pos_precision(self):
        self.pos_val_precision = nltk.metrics.precision(self.actual_polarity_dict['pos'], self.predicted_polarity_dict['pos'])
        print "Pos values preicsiion %s" % (self.pos_val_precision)

    def neg_precision(self):
        self.neg_val_precision = nltk.metrics.precision(self.actual_polarity_dict['neg'], self.predicted_polarity_dict['neg'])
        print "Neg values preicsiion %s" % (self.neg_val_precision)


    def find_recall(self):
        self.pos_recall()
        self.neg_recall()

    def pos_recall(self):
        self.pos_val_recall = nltk.metrics.recall(self.actual_polarity_dict['pos'], self.predicted_polarity_dict['pos'])
        print "Pos values recall %s" % (self.pos_val_recall)

    def neg_recall(self):
        self.neg_val_recall = nltk.metrics.recall(self.actual_polarity_dict['neg'], self.predicted_polarity_dict['neg'])
        print "Neg values recall %s" % (self.neg_val_recall) 

    def find_fmeasure(self):
        self.pos_fmeasure()
        self.neg_fmeasure()

    def pos_fmeasure(self):
        pos_fmeasure_val = 2 * (self.pos_val_precision * self.pos_val_recall) / float(self.pos_val_precision + self.pos_val_recall)          
        print "F measure for pos val %s" % (pos_fmeasure_val)           
 
    def neg_fmeasure(self):
        neg_fmeasure_val = 2 * (self.neg_val_precision * self.neg_val_recall) / float(self.neg_val_precision + self.neg_val_recall)          
        print "F measure for neg val %s" % (neg_fmeasure_val)

if __name__ == "__main__":
    ms = movie_sentiment()
    ms.run_main()
