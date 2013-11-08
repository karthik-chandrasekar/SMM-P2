import nltk

class sample_nltk:
    def __init__(self):
        self.samp_review = "No twist and turn. Just thala's presence made everything simple. BGM was classic. Aarambbam - a tribute to real heros, to and for thala fans. #bitch pls, it has just begun"        

    def run_main(self):
        tokenized_without_stop_words_list = self.do_tokenize()
        self.do_stemming(tokenized_without_stop_words_list)


    def do_tokenize(self):

        #Local ds
        tokenized_without_stop_words_list = []

        tokenized_words_list = nltk.word_tokenize(self.samp_review)
        stop_words_set = set(nltk.corpus.stopwords.words())
    
        for word in tokenized_words_list:
            if word not in stop_words_set:
                tokenized_without_stop_words_list.append(word)

        return tokenized_without_stop_words_list
    

    def do_stemming(self, tokenized_without_stop_words_list):
        #Local ds
        stemmed_word_list = [] 
        stemmer = nltk.stem.PorterStemmer()
   
        print "Before stemming"
        print tokenized_without_stop_words_list 

        for word in tokenized_without_stop_words_list:
            stemmed_word_list.append(stemmer.stem(word))


        print stemmed_word_list

if __name__ == "__main__":
    sn = sample_nltk()
    sn.run_main()
