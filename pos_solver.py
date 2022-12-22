###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids:
# 1. Mansi Sarda (msarda)
# 2. Shyam Makwana (smakwana)
# 3. Lakshay Madaan (lmadaan)
# (Based on skeleton code by D. Crandall)
#


import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):                               ## calcu;ating the posterior for each model
        if model == "Simple":
            output=0
            for ind,word in enumerate(sentence):
              if word not in prob_words:
                output+=math.log(0.0001)+math.log(initial_probaility[pos.index(label[ind].upper())])
              else:
                x=(prob_words[word][pos.index(label[ind].upper())])
                if x==0:
                  output+=math.log(0.0001)+math.log(initial_probaility[pos.index(label[ind].upper())])
                else:
                  output+=math.log(x)
            return output
        elif model == "HMM":
            output = self.func2(sentence,label)
            return output
        elif model == "Complex":
            output=0
            for i in range(len(sentence)):
              if i==0:
                if sentence[i] not in prob_words:
                  output+=math.log(1/1000000)+math.log(max(1/1000000,initial_probaility[pos.index(label[i].upper())]))
                else:
                  output+=math.log(max(1/1000000,prob_words[sentence[i]][pos.index(label[i].upper())]/10))+math.log(max(1/1000000,initial_probaility[pos.index(label[i].upper())]))
              elif i==1:
                if sentence[i] not in prob_words:
                  output+=math.log(1/1000000)+math.log(max(1/1000000,prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())]))
                else:
                  output+=math.log(max(1/1000000,prob_words[sentence[i]][pos.index(label[i].upper())]/10))+math.log(max(1/1000000,prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())]))
              else:
                if sentence[i] not in prob_words:
                  output+=math.log(1/1000000)+math.log(max(1/1000000,prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())]))+math.log(max(1/1000000,second_level_prob_matrix[pos.index(label[i-2].upper())][pos.index(label[i-1].upper())][pos.index(label[i].upper())]))
                else:
                  output+=math.log(max(1/1000000,prob_words[sentence[i]][pos.index(label[i].upper())]/10))+math.log(max(1/1000000,prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())]))+math.log(max(1/1000000,second_level_prob_matrix[pos.index(label[i-2].upper())][pos.index(label[i-1].upper())][pos.index(label[i].upper())]))
            return output
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def func2(self,sentence,label):
      output=0
      for i in range(len(sentence)):
        if i==0:
          if sentence[i] not in prob_words:
            output+=math.log(0.0001)+math.log(initial_probaility[pos.index(label[i].upper())])
          else:
            x=(prob_words[sentence[i]][pos.index(label[i].upper())]/10)
            if x==0:
              output+=math.log(0.0001)+math.log(initial_probaility[pos.index(label[i].upper())])
            else:
              output+=math.log(x)+math.log(initial_probaility[pos.index(label[i].upper())])
        else:
          if sentence[i] not in prob_words:
            output+=math.log(0.0001)+math.log(max(prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())],0.0000001))
          else:
            x=(prob_words[sentence[i]][pos.index(label[i].upper())]/10)
            if x<=0:
              output+=math.log(0.0001)+math.log(max(prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())],0.0000001))
            else:
              output+=math.log(x)+math.log(max(prob_matrix[pos.index(label[i-1].upper())][pos.index(label[i].upper())],0.0000001))
      return output
    def train(self, data):
      global words_dic
      words_dic = {}; 
      global prob_matrix
      prob_matrix = [[0.00000001 for i in range(12)] for j in range(12)]
      global second_level_prob_matrix
      second_level_prob_matrix = [[[0.00000001 for k in range(12)] for i in range(12)] for j in range(12)]
      global initial_probaility 
      initial_probaility = [0.00000001 for j in range(12)]
      global pos 
      pos = ['ADJ','ADV','ADP','CONJ','DET','NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']    ## for training we create as 12x12 transition matrix
      total_words = 0                                                                          ## a list of initial probability of length 12
      for sen in data:                                                                         ## a 12x12x12 matrix to get second level transition matrix
        words,pos_list = sen
        total_words += len(words)
        for i in range(len(words)):
          key = words[i].lower()
          if i==0:
            initial_probaility[pos.index(pos_list[i].upper())]+=1
          elif i==1:
            prob_matrix[prev][pos.index(pos_list[i].upper())] += 1
            before_prev = pos.index(pos_list[i-1].upper())
          else:
            prob_matrix[prev][pos.index(pos_list[i].upper())] += 1
            second_level_prob_matrix[before_prev][prev][pos.index(pos_list[i].upper())]+=1
            before_prev = pos.index(pos_list[i-1].upper())
          prev = pos.index(pos_list[i].upper())
          if key not in words_dic:
            words_dic[key] = [0]*12
            words_dic[key][pos.index(pos_list[i].upper())]+=1                                
          else:
            words_dic[key][pos.index(pos_list[i].upper())]+=1
      
      n = len(data)
      for i in range(12):
        for j in range(12):
            prob_matrix[i][j] = prob_matrix[i][j]/(total_words-n)                                ##  also calculating emission probability
      initial_probaility = [i/n for i in initial_probaility]
      
      for i in range(12):
        for j in range(12):
          s = sum(second_level_prob_matrix[i][j])
          for k in range(12):
            second_level_prob_matrix[i][j][k]=second_level_prob_matrix[i][j][k]/s
      
      #print(prob_matrix)
      def naive_bayes():
        global prob_words
        prob_words  = {}
        for i in words_dic:
          prob_words[i] = [0.00001]*12 
          s = sum(words_dic[i])
          for ind in range(12):
            prob_words[i][ind] =  words_dic[i][ind]/s
  
      

      
      naive_bayes()
      

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence): 
        pos_tag=[]   
        for w in sentence:
          if w in prob_words.keys():
            pos_tag.append( pos[prob_words[w].index(max(prob_words[w]))].lower())
          else:
            pos_tag.append('noun')

        return pos_tag

    def hmm_viterbi(self, sentence):
        global dp
        words = list(set(sentence))
        result = []
        dp = [[0 for i in range(12)] for j in range(len(sentence))]
        def HMM():
          for i in range(len(sentence)):
            w = sentence[i]
            if i==0:
              for index in range(12):
                try:
                  dp[i][index] = (prob_words[w][index]*initial_probaility[index],-1)
                except:
                  dp[i][index] = (0.0001*prob_matrix[index][0],-1)
            else:
              for index in range(12):
                li = []
                for k in range(12):
                  try:
                    li.append(prob_words[w][index]*prob_matrix[k][index]*dp[i-1][k][0])            ## probabilty by hmm viterbi method
                  except:
                    li.append(0.0001*prob_matrix[k][index]*dp[i-1][k][0])
                dp[i][index] = (max(li),li.index(max(li))) 
            prob = [p[0] for p in dp[i]]   
            result.append(pos[prob.index(max(prob))].lower())
          
          ## Backtracking was giving low accuracy
          '''
          l = dp[len(sentence)-1]
          
          prob = [p[0] for p in l]
          p_index= prob.index(max(prob))
          result.append(pos[prob.index(max(prob))].lower())
          for i in range((len(sentence)-2),-1,-1):
            p_index = dp[i+1][p_index][1]
            result.append(pos[p_index].lower())
          '''
        HMM()

          
        return result
    def func(self,sentence):
        global dp1
        r = []
        dp1 = [[0 for i in range(12)] for j in range(len(sentence))]
        for i in range(len(sentence)):
            w = sentence[i]
            if i==0:
              for index in range(12):
                try:
                  dp1[i][index] = (prob_words[w][index],-1)
                except:
                  dp1[i][index] = (0.001,-1)
            else:
              for index in range(12):
                li = []
                for k in range(12):
                  try:
                    li.append(prob_words[w][index]*prob_matrix[k][index]*dp1[i-1][k][0])
                  except:
                    li.append(0.001*prob_matrix[k][index]*dp1[i-1][k][0])
                dp1[i][index] = (max(li),li.index(max(li))) 
            prob = [p[0] for p in dp1[i]]   
            r.append(pos[prob.index(max(prob))].lower())
        return r

    def mcmc_prob(self,words,sample):
        s1 = pos.index(sample[0].upper())
        prob_s1 = math.log(initial_probaility[s1] / sum(initial_probaility),10)
        v1,v2,v3 = 0,0,0
        for i in range(len(sample)):
            try:
              v2 += math.log(max(prob_words[words[i]][i],0.0001),10)
            except:
              v2+=math.log(0.0001,10)
            if i != 0:
                v1 += math.log(max(prob_matrix[pos.index(sample[i - 1].upper())][pos.index(sample[i].upper())],0.0001),10)
            if i != 0 and i != 1:
                v3 += math.log(max(second_level_prob_matrix[pos.index(sample[i - 2].upper())][pos.index(sample[i - 1].upper())][pos.index(sample[i].upper())],0.0001),10)
        return prob_s1+v1+v2+v3
        
    def sampling(self, words, s1):
        pos_tag = pos
        for ind in range(len(words)):
            probability_list,log_probability_list = [0] * len(pos_tag) , [0] * len(pos_tag)
            for k in range(len(pos_tag)):
                s1[ind] = pos_tag[k]
                log_probability_list[k] = self.mcmc_prob(words, s1)

            a = min(log_probability_list)
            for l in range(len(log_probability_list)):
                log_probability_list[l] -= a
                probability_list[l] = math.pow(10, log_probability_list[l])

            s = sum(probability_list)
            probability_list = [x / s for x in probability_list]
            rand = random.random()
            v = 0
            for m in range(len(probability_list)):
                v += probability_list[m]
                if rand < v:
                    s1[ind] = pos_tag[m]
                    break
        return s1
    
    def complex_mcmc(self, sentence):

        samples_list = []
        count_pos_tag_array = []
        sample = self.simplified(sentence) 
        total_iter,burning_iter= 20 , 15 
        r2 = self.func(sentence)
        words = [w for w in sentence]
        for k in range(total_iter):
            sample = self.sampling(words, sample)
            if k >= burning_iter:
                samples_list.append(sample)

        for l in range(len(words)):
            count_pos_tag = {}
            for sample in samples_list:
                key = sample[l]
                try:
                    count_pos_tag[key] += 1
                except:
                    count_pos_tag[key] = 1
            count_pos_tag_array.append(count_pos_tag)

        final_pos_tag = []
        for ind in range(len(words)):
          final_pos_tag.append(max(count_pos_tag_array[ind], key = count_pos_tag_array[ind].get))
        re = [ pos_tag.lower() for pos_tag in final_pos_tag ]
        return r2



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")