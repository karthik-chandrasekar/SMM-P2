import nltk, os, logging, json, ConfigParser, codecs
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

class sample_nltk:
    def __init__(self):

        self.config = ConfigParser.ConfigParser()
        self.config.read("senti_analysis.config")

        #File names
        self.bag_of_words_file_dir = self.config.get('GLOBAL','bag_of_words_file_dir')
        self.bag_of_words_file_name = self.config.get('GLOBAL', 'bag_of_words_file_name')
        self.feat_file_dir = self.config.get('GLOBAL', 'feat_file_dir')
        self.feat_file_name = self.config.get('GLOBAL', 'feat_file_name')
        self.test_feat_file_dir = self.config.get('GLOBAL', 'test_feat_file_dir')
        self.test_feat_file_name = self.config.get('GLOBAL', 'test_feat_file_name')

        self.preprocessing_results_dir = self.config.get('PRE_PROCESSING', 'pre_processing_results_dir')
        self.preprocessing_results_file_name = self.config.get('PRE_PROCESSING', 'pre_processing_results_file_name')
        self.preprocessing_results_file = os.path.join(self.preprocessing_results_dir, self.preprocessing_results_file_name)

        self.logger_file = os.path.join("OUTPUT", "senti_analysis.log")

        #Global DS
        self.id_to_word_dict = {}
        self.id_to_bow_dict = {}

        self.tokenized_without_stop_words_list = []
        self.stemmed_word_list = []
 
        self.stop_words_set = set()
    
    def run_main(self):
        self.preprocessing()
        self.feature_selection()
        self.feature_extraction()
        self.classification()

    def preprocessing(self):
        self.initialize_logger()
        self.parse_reviews()
        #self.remove_stop_words()
        #self.do_stemming()
        #self.dump_preprocessing()

    def feature_selection(self):

        self.pos_reviews, self.neg_reviews = self.get_labelled_reviews_words(self.train_reviews_list)
        self.test_pos_reviews, self.test_neg_reviews = self.get_labelled_reviews_words(self.test_reviews_list)
      
        self.pos_tagged_reviews, self.neg_tagged_reviews = self.tag_words_with_labels(self.pos_reviews, self.neg_reviews) 
        self.test_pos_tagged_reviews, self.test_neg_tagged_reviews = self.tag_words_with_labels(self.test_pos_reviews, self.test_neg_reviews)    
        self.train_reviews = self.pos_tagged_reviews + self.neg_tagged_reviews
        self.test_reviews = self.test_pos_tagged_reviews + self.test_neg_tagged_reviews  

    def feature_extraction(self):
        pass

    def classification(self):

        import pdb;pdb.set_trace()
        classifier = NaiveBayesClassifier.train(self.train_reviews)

        print 'accuracy:', nltk.classify.util.accuracy(classifier, self.test_reviews)
        classifier.show_most_informative_features() 


    def get_labelled_reviews_words(self, reviews_list):
        pos_reviews_words_list = neg_reviews_words_list = []

        for review in reviews_list:
            words_list = []

            if not review:
                continue

            label = review[0]
            if int(label) >= 7:
                label = True
            elif int(label) <=4:
                label = False
            else:
                label = False

            if label:
                for word_freq_pair in review.split(" ")[1:]:
                    if not word_freq_pair:
                        continue
                    word_id = str(word_freq_pair.split(":")[0])
                    word = self.id_to_word_dict.get(int(word_id))
                    if not word:
                        continue
                    words_list.append(word)
                pos_reviews_words_list.append(words_list)
            else: 
                for word_freq_pair in review.split(" ")[1:]:
                    if not word_freq_pair:
                        continue
                    word_id = str(word_freq_pair.split(":")[0])
                    word = self.id_to_word_dict.get(int(word_id))
                    if not word:
                        continue
                    words_list.append(word)
                neg_reviews_words_list.append(word)
        return (pos_reviews_words_list, neg_reviews_words_list)


    def tag_words_with_labels(self, pos_reviews_words, neg_reviews_words):
        pos_tagged_words = []
        neg_tagged_words = []


        for word_list in pos_reviews_words:
            word_dict = self.tag_words(word_list)
            if not word_dict:
                continue
            pos_tagged_words.append([word_dict, 'pos'])
      
        for word_list in neg_reviews_words:
            neg_word_dict = self.tag_words(word_list)
            if not neg_word_dict:
                continue
            neg_tagged_words.append([neg_word_dict, 'neg'])

        return(pos_tagged_words, neg_tagged_words) 


    def tag_words(self, words_list):
        tag_word_list = []

        if type(words_list) != list:
            return dict()

        for word in words_list:
            word_tuple = (word, True)
            tag_word_list.append(word_tuple)

        return dict(tag_word_list)

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")

    def parse_reviews(self):
        self.open_files()
        self.load_data()
        self.close_files()

    def open_files(self):
        self.bow = open(os.path.join(self.bag_of_words_file_dir, self.bag_of_words_file_name), 'r')
        self.feat = open(os.path.join(self.feat_file_dir, self.feat_file_name), 'r')
        self.test_feat = open(os.path.join(self.test_feat_file_dir, self.test_feat_file_name), 'r')

    def load_data(self):
        self.load_bow()
        self.train_reviews_list = self.load_feat(self.feat)
        self.test_reviews_list = self.load_feat(self.test_feat)
        self.load_stop_words() 

    def load_bow(self):
        uniq_id = 0
        for line in self.bow.readlines():
            if not line:    
                continue
            
            self.id_to_word_dict[uniq_id] = line.strip()           
            uniq_id += 1
        logging.info("id_to_word_dict - length - %s" % (len(self.id_to_word_dict)))   

 
    def load_feat(self, fd):
        uniq_id = 0
        reviews_list = []

        for line in fd.readlines():
            if not line:
                continue
            reviews_list.append(line.strip())  #First value in this line hold the labeled value
            uniq_id += 1            
        logging.info("total reviews  - length - %s" % (len(reviews_list)))
        return reviews_list

    def load_stop_words(self):
        self.stop_words_set = set(nltk.corpus.stopwords.words())

    def close_files(self):
        self.bow.close()
        self.feat.close()
        self.test_feat.close()

    def remove_stop_words(self):
        for uniq_id, word in self.id_to_word_dict.iteritems():
            if word in self.stop_words_set:
                self.id_to_word_dict[uniq_id] = '' #Removed stop words will be replaced with null

    def do_stemming(self):
        stemmer = nltk.stem.PorterStemmer()
        for uniq_id, word in self.id_to_word_dict.iteritems():
            self.id_to_word_dict[uniq_id] = stemmer.stem(word)

    def dump_preprocessing(self):
        fd = open(self.preprocessing_results_file, 'w', 'utf-8')
        fd.write(json.dumps(self.id_to_word_dict))
        fd.close()

if __name__ == "__main__":
    sn = sample_nltk()
    sn.run_main()
