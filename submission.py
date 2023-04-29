# Sheng-Hung Kuan
from unittest import TestCase

from HMM import HMM


class submission(TestCase):

    def test_load(self):
        h = HMM("two_english.trans", "two_english.emit")
        print(h.transitions)
        print(h.emissions)

    def test_generate(self):
        h = HMM("partofspeech.browntags.trained.trans", "partofspeech.browntags.trained.emit")
        print(h.generate(20))

    def test_viterbi(self):
        h = HMM("partofspeech.browntags.trained.trans", "partofspeech.browntags.trained.emit")
        with open("ambiguous_sents.obs") as f:
            for line in f:
                o = ['-']
                o.extend(line.split())
                if len(o) <= 1:
                    continue
                h.viterbi(o)

