import nltk, os, logging

class sample_nltk:
    def __init__(self):

        #File names
        self.bag_of_words_file_dir = "/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_PROJECT_2/INPUT/aclImdb"
        self.bag_of_words_file_name = "imdb.vocab"

        self.feat_file_dir = "/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_PROJECT_2/INPUT/aclImdb/train"
        self.feat_file_name = "labeledBow.feat"

        self.logger_file = os.path.join("OUTPUT", "senti_analysis.log")

        #Global DS
        self.id_to_word_dict = {}
        self.id_to_bow_dict = {}
        self.tokenized_without_stop_words_list = []
        self.stemmed_word_list = []
    
    def run_main(self):
        self.initialize_logger()
        self.parse_reviews()
        self.do_tokenize()
        self.do_stemming()
    
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

    def load_data(self):
        self.load_bow()
        self.load_feat()
    

    def load_bow(self):
        uniq_id = 0
        for line in self.bow.readlines():
            if not line:    
                continue
            
            self.id_to_word_dict[uniq_id] = line.strip()           
            uniq_id += 1
        logging.info("id_to_word_dict - length - %s" % (len(self.id_to_word_dict)))   

 
    def load_feat(self):
        uniq_id = 0
        for line in self.feat.readlines():
            if not line:
                continue
            self.id_to_bow_dict[uniq_id] =  line.strip().split(" ")  #First value in this line hold the labeled value
            uniq_id += 1            
        logging.info("id_to_bow_dict - length - %s" % (len(self.id_to_bow_dict)))

    def close_files(self):
        self.bow.close()
        self.feat.close()

    def do_tokenize(self):
        tokenized_words_list = nltk.word_tokenize("hello ")
        stop_words_set = set(nltk.corpus.stopwords.words())
    
        for word in tokenized_words_list:
            if word not in stop_words_set:
                self.tokenized_without_stop_words_list.append(word)

        return self.tokenized_without_stop_words_list
    

    def do_stemming(self):
        stemmer = nltk.stem.PorterStemmer()

        for word in self.tokenized_without_stop_words_list:
            self.stemmed_word_list.append(stemmer.stem(word))


if __name__ == "__main__":
    sn = sample_nltk()
    sn.run_main()
