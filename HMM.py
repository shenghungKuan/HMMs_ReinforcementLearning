
import random
import argparse
import codecs
import os
import numpy
from collections import defaultdict

import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq ) +'\n ' +' '.join(self.outputseq ) +'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions, emissions):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        # self.transitions = transitions
        # self.emissions = emissions

        self.transitions = self.load(transitions)
        self.emissions = self.load(emissions)

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        d = defaultdict(dict)
        with open(basename) as f:
            for line in f:
                (prev, cur, prab) = line.split()
                d[prev][cur] = float(prab)

        return d



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        # transitions
        res = ['#']
        choices = list(self.transitions['#'].keys())
        for i in range(n):
            weights = []
            for c in choices:
                weights.append(self.transitions[res[-1]][c])
            choice = random.choices(choices, weights=weights)
            res.append(choice[0])
        # emissions
        observation = []
        for emit in res[1:]:
            choices = list(self.emissions[emit].keys())
            weights = []
            for c in choices:
                weights.append(self.emissions[emit][c])
            choice = random.choices(choices, weights=weights)
            observation.append(choice[0])
        # observation
        ob = Observation(res, observation)
        return ob



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        state = ['#']
        state.extend(list(self.transitions['#'].keys()))
        prob = np.zeros((len(observation), len(state)), dtype=float)
        backtrack = np.zeros((len(observation), len(state)), dtype=int)
        prob[0, 0] = 1
        for i in range(1, len(observation)):
            for j in range(len(state)):  # trans[#][ADV] * emit[ADV][I] * prob[i, j]
                val_list = list(self.transitions[state[k]].get(state[j], 0) * prob[i - 1, k] for k in range(len(state)))
                val = np.array(val_list).argmax()
                prob[i, j] = self.emissions[state[j]].get(observation[i], 0)\
                             * max(self.transitions[state[k]].get(state[j], 0)
                                   * prob[i - 1, k] for k in range(len(state)))
                backtrack[i, j] = int(val)

        for i in range(1, len(observation)):
            index = np.array(prob[i,]).argmax()
            print(state[index])
        print("-------------------------------------------")



