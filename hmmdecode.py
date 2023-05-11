import sys 
import os 
import numpy as np
import re

def load():
    tag_dict = {}
    text_dict = {}
    tag_list = []
    transition_matrix = []
    emission_matrix = []

    with open('hmmmodel.txt','r', encoding = 'utf-8') as f:
        #print(f.read()[:80].split('\n'))
        #stuff = f.read().split('')
        len_text = int(f.readline())
        #print(len_text)
        len_tags = int(f.readline())
        #print(len_tags)
        i = 0
        while True:

            #print(i)
            i+=1
            line = f.readline().split('\n')[0]
            #print(line)
            if line == '':
            #print('hi')
            #print(line)
                break

            elif line == 'Transition Matrix':
            #print("ho")
                for _ in range(len_tags):
                    l = f.readline()
                    #print(l)
                    #break
                    l = np.fromiter(map(np.float64, l.split(' ')), dtype = np.float64)
                    transition_matrix.append(l)

            elif line == 'Emission Matrix':
                for _ in range(len_text-1):
                    l = f.readline()
                    l = np.fromiter(map(np.float64, l.split(' ')), dtype = np.float64)
                    emission_matrix.append(l)

            elif line == "Tags":
                for i in range(len_tags-1):
                    #print(i)
                    l = f.readline().split(' ')
                    #print(l)
                    tag, index = l

                    tag_list.append(tag)
                    tag_dict[tag] = int(index.split('\n')[0])

            elif line == "Text":
                for _ in range(len_text-1):
                    text, index = f.readline().split(' ')

                    text_dict[text] = int(index.split('\n')[0])


    return tag_dict, text_dict, tag_list, np.asarray(emission_matrix), np.asarray(transition_matrix)

def viterbi_algorithm(words, tag_dict, text_dict, tag_list, emission_matrix, transition_matrix):
  states = []
  n = len(tag_dict)
  m = len(words)
  tag_matrix = np.zeros((n,m))
  state_matrix = np.zeros((n,m))
  set_words = set()
  dummy = np.asarray([1/n for _ in range(n)])
  
  for key, word in enumerate(words):
    #try:
    #  print(emission_matrix[text_dict[word]])
    #if it is the first word i.e. start
    if key == 0:

      #there are no states before this we only calculate thetransition probability from start state to all the hidden states for the first word
      #elementwise multiplication between the transition matrix row of the start tag(transition probability from start to all other tags) and
      #the emission probability for the word wrt to all tags
      try:
        #print(text_dict[word])
        calc = np.multiply(transition_matrix[len(tag_dict),:-1],  emission_matrix[text_dict[word],:]).reshape((len(tag_dict),1))
      except KeyError:
        #print("HEY")
        set_words.add(word)
        calc = np.multiply(transition_matrix[len(tag_dict),:-1],  dummy).reshape((len(tag_dict),1))
      state_matrix[:,0] = calc.ravel()
      #print("...................")
      #print(calc)

    else:

      #if its not start then we need to calculate the max probability from all possible hidden states for previous word
      #with the current hidden state
      for tag in tag_dict:
        tag_key = tag_dict[tag]

        #basically we do elementwise multiplication
        #for all hidden states of previous word we multiply
        # transition probability of previous state to current, the maximum probability of the previous hidden state under consideration(δ) and
        #emission probaility of a particular word from the hidden state
        #refer to the forward pass video from above for the exact formula
        try:
          
          calc = np.multiply(state_matrix[:, key-1],transition_matrix[:-1,tag_key]) * emission_matrix[text_dict[word]][tag_key]
        
        except KeyError:
          #i = randint()
          #print(word, tag_key)
          set_words.add(word)
          calc = np.multiply(state_matrix[:, key-1],transition_matrix[:-1,tag_key]) * dummy[tag_key]

        state_matrix[tag_key][key] = np.amax(calc)   #find the maximum probability
        tag_matrix[tag_key][key] = np.argmax(calc)   #find the previous state that maximizes this one

  #print(tag_matrix)
  #finding the probability that a state would be an end state. We take the argmax for this to
  #find the maximum likelihood state for the last word
  max_state = np.argmax(np.multiply(transition_matrix[:-1, len(tag_dict)], state_matrix[:,-1])) 
  
  states.append(tag_list[int(max_state)])

  #refer backward state video
  #so now we do the backward pass where we iterate through tag_matrix backward.
  #We find the max state for a particular word. That is the previous hidden state that maximizes the current.
  #This way we move backward and keep storing the maximum likelihood tag in states
  for i in reversed(range(1,len(words))):
    max_state = int(tag_matrix[max_state][i])
    states.append(tag_list[max_state])

  #print(state_matrix)
  #print(tag_matrix)
  #print(states)

  #reverse states because we've appended it from last to first
  print(set_words)
  return states[::-1]

if __name__ == '__main__':
    tag_dict, text_dict, tag_list, emission_matrix, transition_matrix = load()

    path = sys.argv[1]


    with open(path, "r", encoding="utf-8") as f:
        corpus_test = f.read().split('\n')[:-1]

    sent = []

    for i in corpus_test:
        words = []
        
        i = i.split(' ')
        #print(i)
        for word in i:
            #word, test_tag = j.rsplit('/',1)
            words.append(word)
            #test_tags.append(test_tag)

        sent.append(words)
        #sent_tags.append(test_tags)

    tagged_sents = []
    for  i in range(len(sent)):
        ans = viterbi_algorithm(sent[i], tag_dict, text_dict, tag_list, emission_matrix, transition_matrix)

        for j in range(len(ans)):
            #print()
            sent[i][j] = '/'.join([sent[i][j],ans[j]])

        tagged_sents.append(' '.join(sent[i]))

    with open('hmmoutput.txt', 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(tagged_sents))
        #f.write('\n')