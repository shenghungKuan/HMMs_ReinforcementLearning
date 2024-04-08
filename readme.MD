# HMMs and Reinforcement Learning

Part 2: Hidden Markov Models 

(Note: this is derived from an assignment in AAAI's Model Assignments workshop)

In this assignment you'll be implementing two algorithms associated with Hidden Markov Models.

You'll be building off of the code presented in HMM.py. There's also some included data to use.
The first set of files are called two-english.trans and two-english.emit. These represent a two-state HMM that was *learned* from English words.
This HMM has two states: Consonant and Vowel. It's here for practice. Since it only has two states, it's not going to be very accurate.

The first set of files are .trans files. They contain the transition probabilities. two_english models the transition between 'C' for Consonant and 'V' for Vowel in English. There's also a '#' which is 
used to represent the starting state.

The second set of files are .emit files. These contain the probability of emitting a particular output from that state. These are learned from data, and so contain errors (especially two_english).

The second set of .trans/.emit files are for the Brown corpus. These represent an HMM for recognizing parts of speech. This problem is larger, with 12 states and lots of possible emissions. 

The last pair of files is ambiguous_sents.obs and ambiguous_sents.tagged.obs. This is what you'll test your Viterbi implementation on.


**(5 points)**. Use the included code to implement load. Use two_english as a sample file to work with. You should be able to do:

model = HMM()
model.load('two_english')

You should store the transitions and emissions as dictionaries of dictionaries. e.g. {'#': {'C': 0.814506898514, 'V': 0.185493101486}, 'C': {'C': 0.625840873591, 'V': 0.374159126409}, 'V': {'C': 0.603126993184, 'V': 0.396873006816}}

**(10 points)** Implement generate. It should take an integer n, and return a random observation of length n. To generate this, start in the initial state and repeatedly select successor states at random, using the probability as a weight, and then select an emission, again using the probability as a weight. You may find either numpy.random.choice or random.choices very helpful here.
Be sure that you are using the transition probabilities to determine the next state, and not a uniform distribution!

You should be able to run it with the pre-trained probabilities for the Brown corpus, like so:

python hmm.py partofspeech.browntags.trained --generate 20

which generates 20 random observations.

Here are two sample observations:

DET ADJ . ADV ADJ VERB ADP DET ADJ NOUN VERB ADJ NOUN 

the semi-catatonic , quite several must of an western bridge cannot spectacular analyses 

DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN 

whose light for wall and the learned the hull postmaster trash in his peters

**(15 points)** Next, implement Viterbi. This tells us, for a sequence of observations, the most likely sequence of states. You should be able to run this like so:

python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs

This uses the HMM parameters in partofspeech.browntags.trained.{trans,emit} to compute the best sequence of part-of-speech tags for each sentence in ambiguous_sents.obs, and prints the results.

You might find it helpful to use a numpy array to hold the matrix.

You can compare your results to ambiguous_sents.tagged.obs.

**(35 points)** Part 3: Reinforcement Learning for Approach.

In this task, we'll return to Approach and use Q-learning to learn the optimal policy through repeated play.

Recall the dice game "Approach". It works like this: There are two players, and they choose a target number (n). Player 1 repeatedly rolls a six-sided die and adds up their total. Whenever they want, they can "hold." Then player 2 must roll;' their goal is to beat player 1's score without going past n.

The problem we want to solve is this:

For player 1, for a given n and a particular total so far (called s) should they 'hold' (action 1) or 'roll' (action 2).

(note that player 2's strategy in this game is pretty boring - they just roll until they either beat player 1's score or exceed n.)

Earlier, we used Monte Carlo simulation to determine the best value to hold. 
This is fine when there's a small number of potential policies to consider, but it's not very scalable. 
Let's now add in Q-learning.

Q-learning generates a table that stores Q(s,a). I've started a function called approach for you. 
It takes in a limit n and computes the Q-table for [0,n] with the actions (hold, roll). 
You will need to complete this.

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

We will do this using epsilon-soft on-policy control, a form of reinforcement learning. 
We will select the optimal action with probability (1-epsilon) and explore with probability epsilon. (use epsilon=0.1 for this.)

Start by initializing the Q-table to small random values. 
Our agent should start with a random s and a and then either take the best action (prob. 1-epsilon), or the other action (prob. epsilon). At the end of the game, it receives reward 1 (if it wins) or 0 (if it loses). 
It then uses this to update the Q value for each state-action pair chosen. Run for 1000000 iterations and print out the Q-table, and optimal action for each state.

# Approach.py


# This program uses reinforcement learning to determine the optimal policy
# for Approach.
# Recall that approach works like this:
# Both players agree on a limit n.
# Player 1 rolls first. They go until they either exceed n or hold.
# Then player 2 rolls. They go until they either exceed n or beat player 1's score.
# The player who is closest to n without going over wins.
# Note:
# We can reduce this to the problem of player 1 choosing the best value at which to hold.
# This is called a policy; once we know the best number to hold at, we can act optimally.