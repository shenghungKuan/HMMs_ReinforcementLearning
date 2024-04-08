# HMMs and Reinforcement Learning

## HMM.py (Hidden Markov Models)

In this assignment you'll be implementing two algorithms associated with Hidden Markov Models.

two-english.trans and two-english.emit represent a two-state HMM that was learned from English words. This HMM has two states: Consonant and Vowel.

The .trans files contain the transition probabilities. two_english models the transition between 'C' for Consonant and 'V' for Vowel in English. There's also a '#' which is used to represent the starting state.

The .emit files contain the probability of emitting a particular output from that state. These are learned from data, and so contain errors (especially two_english).

The .trans/.emit files are for the Brown corpus. These represent an HMM for recognizing parts of speech. This problem is larger, with 12 states and lots of possible emissions. 

ambiguous_sents.obs and ambiguous_sents.tagged.obs are used for testing the Viterbi implementation.

The transitions and emissions as dictionaries of dictionaries. e.g. {'#': {'C': 0.814506898514, 'V': 0.185493101486}, 'C': {'C': 0.625840873591, 'V': 0.374159126409}, 'V': {'C': 0.603126993184, 'V': 0.396873006816}}

Implement generate. It should take an integer n, and return a random observation of length n. To generate this, start in the initial state and repeatedly select successor states at random, using the probability as a weight, and then select an emission, again using the probability as a weight.

Run it with the pre-trained probabilities for the Brown corpus:
```
python hmm.py partofspeech.browntags.trained --generate 20
```
which generates 20 random observations.

Here are two sample observations:

DET ADJ . ADV ADJ VERB ADP DET ADJ NOUN VERB ADJ NOUN 

the semi-catatonic , quite several must of an western bridge cannot spectacular analyses 

DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN 

whose light for wall and the learned the hull postmaster trash in his peters

Implement Viterbi. This tells us, for a sequence of observations, the most likely sequence of states.
Run this like so:
```
python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs
```
This uses the HMM parameters in partofspeech.browntags.trained.{trans,emit} to compute the best sequence of part-of-speech tags for each sentence in ambiguous_sents.obs, and prints the results.

## approach.py (Reinforcement Learning for Approach)

I used Q-learning to learn the optimal policy through repeated play for the dice game "Approach".

"Approach": It works like this: There are two players, and they choose a target number (n). Player 1 repeatedly rolls a six-sided die and adds up their total. Whenever they want, they can "hold." Then player 2 must roll;' their goal is to beat player 1's score without going past n.

For player 1, for a given n and a particular total so far (called s) should they 'hold' (action 1) or 'roll' (action 2).

(Player 2's strategy in this game is keeping rolling until they either beat player 1's score or exceed n.)

Q-learning generates a table that stores Q(s,a). The function "approach". 
It takes in a limit n and computes the Q-table for [0,n] with the actions (hold, roll). 

For example, for n=10, you might get:

<pre>
sum:   hold     roll   [action]
0: 0.000000 0.493921 [roll]
1: 0.000000 0.477271 [roll]
2: 0.000000 0.479286 [roll]
3: 0.000000 0.505797 [roll]
4: 0.000000 0.580803 [roll]
5: 0.051270 0.498027 [roll]
6: 0.170403 0.427388 [roll]
7: 0.297536 0.365115 [roll]
8: 0.478196 0.285432 [hold]
9: 0.711051 0.164235 [hold]
10: 1.000000 0.000000 [hold]
</pre>

I did this using epsilon-soft on-policy control, a form of reinforcement learning. 
We will select the optimal action with probability (1-epsilon) and explore with probability epsilon.

Start by initializing the Q-table to small random values. 
Our agent should start with a random s and a and then either take the best action (prob. 1-epsilon), or the other action (prob. epsilon). At the end of the game, it receives reward 1 (if it wins) or 0 (if it loses). 
It then uses this to update the Q value for each state-action pair chosen. Run for 1000000 iterations and print out the Q-table, and optimal action for each state.
