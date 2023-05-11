import sys 
import os 
import numpy as np
import re


def prepare(corpus):
  
  text_tag = []

  for  i in range(len(corpus)):
    n = len(corpus[i])
    for t in range(n):
      text_tag.append(corpus[i][t].rsplit('/',1))
      if t == 0:
        text_tag[-1].append(True)
      else:
        text_tag[-1].append(False)

      if t == n-1:
        text_tag[-1].append(True)
      else: text_tag[-1].append(False)

  return text_tag

def create_emission_matrix(text_tag, text_dict, tag_dict):
  emission_matrix = np.zeros((len(text_dict),len(tag_dict)))
  emission_matrix.fill(0)

  for i in text_tag:
    try:
        emission_matrix[text_dict[i[0]]][tag_dict[i[1]]] += 1
    except:
        print("error ",i)
        #break

  with np.errstate(divide='ignore', invalid='ignore'):

    emission_matrix =  emission_matrix/emission_matrix.sum(axis=0, keepdims=True)
    print(np.count_nonzero(np.isnan(emission_matrix)))
    #emission_matrix[emission_matrix == np.inf] = 0
    #emission_matrix = np.nan_to_num(emission_matrix)
  return emission_matrix

def create_transition_matrix(text_tag, text_dict, tag_dict):
  transition_matrix = np.zeros((len(tag_dict)+1,len(tag_dict)+1))
  transition_matrix.fill(0)
  corp_len = len(text_tag)
  point = len(tag_dict)
  epsilon = 0.000000001

  for i in range(len(text_tag)-1):
    #since we cant find the i+1 of the last element, we keep this flag
    #if i < corp_len-1:
      
      #if it is the start of a sentence then increment the transition start_tag -> current_tag
      if text_tag[i][2] == True:
        transition_matrix[point][tag_dict[text_tag[i][1]]] +=1

       
      elif text_tag[i+1][3] == True:
        #if pos_res[i][0] != '.':
        transition_matrix[tag_dict[text_tag[i][1]]][point] +=1

      #increment transition current tag -> next_tag
      #else:
      #  if pos_res[i][0] != '.':
      transition_matrix[tag_dict[text_tag[i][1]]][tag_dict[text_tag[i+1][1]]] +=1

  #same thing as in emission
  with np.errstate(divide='ignore', invalid='ignore'):
    
    #again same concept as in what was done in emission but this time row wise (axis=1)
    transition_matrix =  (transition_matrix+1)/(transition_matrix.sum(axis=1, keepdims=True) + (len(tag_dict)+1))
    #transition_matrix[transition_matrix == np.inf] = 0.00000000001
    #print(np.count_nonzero(transition_matrix == np.inf))
    transition_matrix = np.nan_to_num(transition_matrix)

  #for i in range(len(tag_dict)):
  #transition_matrix[transition_matrix == 0] = epsilon
  return transition_matrix

def save(text_dict, tag_dict, tag_list, emission_matrix, transition_matrix):
  a = len(text_dict)+1
  b = len(tag_dict)+1
  #tag_list.append('ST\ED')
  with open('hmmmodel.txt','w', encoding='utf-8') as f:

    f.write(f"{a}\n")
    f.write(f"{b}\n")
    f.write("Transition Matrix\n")
    for i in range(len(tag_dict)+1):
      row = ' '.join(str(num) for num in transition_matrix[i])
      #z = list(map(np.float64, z.split(' ')))
      f.write(row)
      f.write('\n')

    f.write("Emission Matrix\n")
    for i in range(len(text_dict)):
      row = ' '.join(str(num) for num in emission_matrix[i])
      #z = list(map(np.float64, z.split(' ')))
      f.write(row)
      f.write('\n')

    f.write("Tags\n")
    for tag, num in tag_dict.items():

      f.write(f"{tag} {num}\n")

    f.write("Text\n")
    for text, num in text_dict.items():

      f.write(f"{text} {num}\n")

def HMMlearn(path):
  with open(path, "r", encoding="utf-8") as f:
    #corpus.append(f.readline().split('\n')[0].split(' '))
    #corpus.append(f.read().split('\n'))
    corpus = f.read().split('\n')[:-2]

  for i in range(len(corpus)):
    corpus[i] = corpus[i].split(' ')

  text_tag = prepare(corpus)

  text_dict = {}
  tag_dict = {}
  tag_list = []
  text_count = 0
  tag_count = 0
  visited_text = set()
  visited_tag = set()
  for text in text_tag:

    if text[0] not in visited_text:
      text_dict[text[0]] = text_count
      visited_text.add(text[0])
      text_count += 1

    if text[1] not in visited_tag:
      tag_list.append(text[1])
      tag_dict[text[1]] = tag_count
      visited_tag.add(text[1])
      tag_count +=1


  emission_matrix = create_emission_matrix(text_tag, text_dict,tag_dict)
  transition_matrix = create_transition_matrix(text_tag, text_dict,tag_dict)

  save(text_dict, tag_dict, tag_list, emission_matrix, transition_matrix)
if __name__ == "__main__":
    HMMlearn(sys.argv[1])
