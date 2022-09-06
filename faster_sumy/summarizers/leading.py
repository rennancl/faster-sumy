# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from ._summarizer import AbstractSummarizer


class LeadingSummarizer(AbstractSummarizer):
    """Summarizer that picks the first sentences."""

    def __call__(self, document, sentences_count):
        sentences = document.sentences
        ratings = self._get_ratings(sentences)

        return self._get_best_sentences(sentences, sentences_count, ratings)

    def _get_ratings(self, sentences):
        ratings = list(range(len(sentences)))
        ratings.reverse()

        return dict((s, r) for s, r in zip(sentences, ratings))
