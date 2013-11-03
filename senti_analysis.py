import nltk

class sample_nltk:
    def __init__(self):
        self.samp_review = "No twist and turn. Just thala's presence made everything simple. BGM was classic. Aarambbam - a tribute to real heros, to and for thala fans. #bitch pls, it has just begun"        
        self.pre_processed_words = []


    def run_main(self):
        self.do_tokenize()
        self.do_stemming()


    def do_tokenize(self):
            tokenized_words = nltk.word_tokenize(self.samp_review)
            tokenized_words_set = set(tokenized_words)
        

            stop_words_set = set(nltk.corpus.stopwords.words())
        

            for word in tokenized_words:
                if word not in stop_words_set:
                    self.pre_processed_words.append(word)
     
            print tokenized_words
            print self.pre_processed_words

    def do_stemming(self):
            pass


if __name__ == "__main__":
    sn = sample_nltk()
    sn.run_main()
