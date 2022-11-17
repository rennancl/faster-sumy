# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re

from ._summarizer import AbstractSummarizer

class SumFocusSummarizer(AbstractSummarizer):
    """
    SumBasic: a frequency-based summarization system that adjusts word frequencies as
    sentences are extracted.
    Source: http://www.cis.upenn.edu/~nenkova/papers/ipm.pdf
    """
    _stop_words = frozenset()
    __lambda = 0.9
    _diversity = "default"
    _volume = 1

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))
        
    @property
    def _lambda(self):
        return self.__lambda

    @_lambda.setter
    def _lambda(self, _lambda):
        self.__lambda = _lambda

    @property
    def diversity(self):
        return self._diversity

    @diversity.setter
    def diversity(self, div):
        self._diversity = div      
        
    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, vol):
        self._volume = vol     

    def __call__(self, document, sentences_count, query):
        sentences = document.sentences
        ratings = self._compute_ratings(sentences, query)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _get_all_words_in_doc(self, sentences):
        return self._normalize_words([w for s in sentences for w in s.words])

    def _get_content_words_in_sentence(self, sentence):
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        stemmed_normalized_content_words = self._normalize_words(normalized_content_words)
        return stemmed_normalized_content_words

    def _normalize_words(self, words):
        return [self.stem_word(self.normalize_word(w)) for w in words]

    def _filter_out_stop_words(self, words):
        return [w for w in words if w not in self.stop_words]

    @staticmethod
    def _compute_word_freq(list_of_words):
        word_freq = {}
        for w in list_of_words:
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def _get_all_content_words_in_doc(self, sentences):
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words

    def _get_query_words(self, query):
        query = re.sub(r'[^\w\s]', '', query)
        query_words = query.split(" ")
        normalized_query_words = self._normalize_words(query_words)
        return normalized_query_words
    
    def _compute_tf(self, sentences, query):
        """
        Computes the normalized term frequency as explained in http://www.tfidf.com/
        """
        # tratar palavras aqui
        query_words = self._get_query_words(query)
        
        content_words = self._get_all_content_words_in_doc(sentences)
        content_words_count = len(content_words)
        content_words_freq = self._compute_word_freq(content_words)
        content_word_tf = dict((k, (v / content_words_count) *(1 + self._lambda * int(k in query_words))) for (k, v) in content_words_freq.items())
        return content_word_tf

    def _compute_average_probability_of_words(self, word_freq_in_doc, content_words_in_sentence):
        content_words_count = len(content_words_in_sentence)
        if content_words_count > 0:
            word_freq_sum = sum([word_freq_in_doc[w] for w in content_words_in_sentence])
            word_freq_avg = word_freq_sum / pow(content_words_count, self.volume)
            return word_freq_avg
        else:
            return 0

    def _update_tf(self, word_freq, words_to_update):
        for w in words_to_update:
            if self.diversity == "max":
                # never select same word
                word_freq[w] = -1
            if self.diversity == "median":
                # might select same word
                word_freq[w] = 0
            if self.diversity == "default":
                # default diverisity
                word_freq[w] *= word_freq[w] 
            if self.diversity == "min":
                # doesnt update word
                word_freq[w] = word_freq[w] 
        return word_freq

    def _find_index_of_best_sentence(self, word_freq, sentences_as_words):
        min_possible_freq = -1
        max_value = min_possible_freq
        best_sentence_index = 0
        for i, words in enumerate(sentences_as_words):
            word_freq_avg = self._compute_average_probability_of_words(word_freq, words)
            if word_freq_avg > max_value:
                max_value = word_freq_avg
                best_sentence_index = i
        return best_sentence_index

    def _compute_ratings(self, sentences, query):
        word_freq = self._compute_tf(sentences, query)
        ratings = {}

        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [self._get_content_words_in_sentence(s) for s in sentences]

        # Removes one sentence per iteration by adding to summary
        while len(sentences_list) > 0:
            best_sentence_index = self._find_index_of_best_sentence(word_freq, sentences_as_words)
            best_sentence = sentences_list.pop(best_sentence_index)

            # value is the iteration in which it was removed multiplied by -1 so that the first sentences removed (the most important) have highest values
            ratings[best_sentence] = -len(ratings)

            # update probabilities
            best_sentence_words = sentences_as_words.pop(best_sentence_index)
            self._update_tf(word_freq, best_sentence_words)

        return ratings
