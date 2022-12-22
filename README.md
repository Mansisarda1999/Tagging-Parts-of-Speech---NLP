# Tagging-Parts-of-Speech---NLP
# Part of Speech tagging

## Problem Statement:
The problem here is the part of Natural Language Processing.
The goal is to implement parts of speech tagging using Bayes Networks.
1. Simplified Bayes Net to get a suitable tag for each word.
2. Richer Bayes Net using Viterbi to find maximum a posteriori (MAP) for the sentence.
3. We have to implement Gibbs Sampling to sample from the posterior distribution.

## Input Data:
To datasets is given bc.train, bc.test.  
The train dataset consists of about 50,000 sentences where in each sentence we have a word followed by its POS i.e ADJ (adjective),ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark).
The test dataset contains 20,000 sentences. We have managed to label the test dataset within 10 min.

## Algorithms:
**1. Simplified Bayes net: (SIMPLE)**
Here, it is assumed that each feature is independent of the other. We simply calculate the probability of a word given the part of speech using train dataset(emission probability) and use this 
posterior probability in the testing phase. 
It is a supervised machine learning algorithm that makes the use of Bayes theorem. 
Testing: Each word is taken and we get the argmax{(P(word|parts of speech_i)*P(parts of speech_i))} where i runs for all parts of speech.

**- Implementation/Design Decisions:**
  We create a dictionary where we store the probability of each word given its part of speech. 
  - P(POS|word) = (P(word|POS) * P(POS))
  The posterior probability of all word-tag is : P(POS1|word1)*P(POS2|word2)…P(POS_n|word_n) α (P(word1|POS1) * P(POS1)) * (P(word2|POS2) * P(POS2)) ….. (P(word_n|POS_n) * P(POS_n))
  If the word is not present in the dictionary, then I am returning noun which is the most common part of speech. 
  
- Assumption: Each word is independent of others.

![4](https://user-images.githubusercontent.com/60294261/205785252-6aca8206-33ef-4187-94d3-a8dfd4052059.png)

2. Richer Bayes Net using Viterbi to find maximum a posteriori (MAP) for the sentence. (HMM)
Here, we are using the concept of the hidden Markov Model. It uses the Dynamic Programming method to get the path of the most likely POS for each word using backtracking.
It estimates the most likely sequence of hidden states referred to as the Viterbi path (HMM).

- Implementation/Design Decisions:
   - Initial Probability is obtained for each POS that is the probability of getting a POS at the beginning of the sentence. It is an array of length 12.
   - Transition Probability is obtained in a 2D array where we calculate the transitioning probability of a POS to another POS. 
  At the time of testing, 
  We create a matrix with rows as words and columns as 12 POS. 
  - If the word is an initial word - We take P(word|POS)*Initial Probability(POS) for all POS and store that in the first row of the matrix.
  - For the rest of the words we take P(word|POS)*Transition Probability(POS)*P(prev_word|POS) for all POS and store that in the corresponding row.
  - We finally take the max probable POS in each row for each word and return that as labels.

-  Assumption: The pos for each word is dependent on the previous word POS.

![3](https://user-images.githubusercontent.com/60294261/205785208-d66bf5a9-7b5c-4132-b792-7ed61e3d5cf8.png)

3. We have to implement Gibbs Sampling to sample from the posterior distribution (Complex):

Here we assumed that every part of speech depends on the previous two parts of speech.
- Implementation/Design Decisions:
- Each word is assigned a random POS. Then we fix a word and compute the posterior given by the following formula for all POS:
- $P(S_{i}|S-S_{i}, W) = (P(S_{1})P(W_{1}|S_{1})P(S_{2}|S_{1})P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})$
- or 
  $P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}*{P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})}{P(W_{1}|S_{1})…P(W_{n}|S_{n})}$
- Second level transition probability is also calculated.

- Initial Sample: We use the labels returned by the simplified Naive Bayes Model.
- Number of iterations: To keep the code running within the time limit constraints I have taken 20 iterations and used initial labels from Naive Bayes. 
- Buring iterations is kept as 15.
- Log probabilities: To solve the problem of getting small probabilities that result in 0 in the program.

-  Assumption: The pos for each word is dependent on the previous 2 words POS.

![2](https://user-images.githubusercontent.com/60294261/205785118-03dd3091-07dd-4c60-876f-ed84720ffa75.png)

## Results: 

The following are the results we obtained using bc.test dataset for the following models:

![1](https://user-images.githubusercontent.com/60294261/205783754-e0b7bc9e-8133-42d6-90e5-f3782d009232.png)

## Observations:
The accuracy of the Complex algorithm is slightly higher than HMM, however, simplified Bayes Net is giving the highest word accuracy of 93.95%.

The following are the posterior probabilities and observations:

![image](https://user-images.githubusercontent.com/60294261/205784491-fe522f36-b99e-414d-b38f-dfc507ea10a4.png)


## Observations:
Simple returns the highest value of their probabilities but it is not very different from HMM and MCMC.

