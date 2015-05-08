from __future__ import division
from mrjob.job import MRJob
import re
import math
from nltk import word_tokenize
from collections import Counter
import string

#WORD_RE = re.compile(r"[\w']+")

class MRTFIDF(MRJob):

    # mapper_init 0
    def gen_tf_init(self, args=None):
        self.line_number = 0

        self.vocabs = map(
            string.strip, open("reviews.txt", "r").readlines())

    # mapper 0
    def gen_tf(self, _, line):
        self.line_number += 1
        word_cnt = Counter()
        #words = WORD_RE.findall(line)
        words = word_tokenize(line)  # from nltk import word_tokenize
        for word in words:
            if word in self.vocabs[self.line_number-1]:
                word_cnt[word] += 1
        for word in word_cnt:
            yield word, (self.line_number, word_cnt[word]/len(word_cnt))

        del word_cnt

    # reducer 0
    def gen_tf_idf(self, word, word_cnt):
        word_cnt = list(word_cnt)
        df = len(word_cnt)
        idf = math.log(63483 / df, 10)
        for term, tf in word_cnt:
            tf_idf = tf * idf
            yield term, (word, tf_idf)

    def steps(self):
        return [
            self.mr(mapper_init=self.gen_tf_init,
                    mapper=self.gen_tf,
                    reducer=self.gen_tf_idf,
            )
        ]

if __name__ == '__main__':
    MRTFIDF.run()
