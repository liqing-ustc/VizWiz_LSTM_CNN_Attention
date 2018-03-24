# create json file for vocabulary
import json
import os
import nltk
from nltk.stem.snowball import *
from tqdm import *
from collections import Counter, OrderedDict
import string

root_dir = os.environ['data_dir']
splits = ['train']

## question
q_counter = Counter()
n_sample = 0
maxlen = 0
for split in splits:
	dataset = json.load(open(root_dir + 'Annotations/%s.json'%split))
	for one_data in tqdm(dataset):
		n_sample += 1
		question = one_data['question']
		question = question.lower()
		tokens = nltk.word_tokenize(question)
		token_len = len(tokens)
		maxlen = max([maxlen,token_len])
		q_counter.update(tokens)
print('number of sample = ' + str(n_sample))
print('max len = ' + str(maxlen))
q_word_counts = [x for x in q_counter.items()]
q_word_counts.sort(key=lambda x: x[1], reverse=True)
json.dump(q_word_counts, open('q_word_counts.json', "w"), indent=2)

### build vocabulary based on question
vocab = [x[0] for x in q_word_counts if x[1] >= 0]
unk_word = '<UNK>'
vocab = [unk_word] + vocab
vocab = OrderedDict(zip(vocab,range(len(vocab))))
json.dump(vocab, open('word2vocab_id.json', 'w'), indent=2)

## answer
ans_counter = Counter()
for split in splits:
	dataset = json.load(open(root_dir + 'Annotations/%s.json'%split))
	for annotation in tqdm(dataset):
		for answer in annotation['answers']:
			answer = answer['answer'].lower()
			ans_counter.update([answer]) # don't forget the [], counter.update input a list
ans_counts = [x for x in ans_counter.items()]
ans_counts.sort(key=lambda x: x[1], reverse=True)
json.dump(ans_counts, open('ans_counts.json', "w"), indent=2)

### build answer candidates
output_num = 3000
n_totoal = sum([x[1] for x in ans_counts])
ans_counts = ans_counts[:output_num]
n_cover = sum([x[1] for x in ans_counts])
print "we keep top %d most answers"%len(ans_counts)
print "coverage: %d/%d (%.4f)"%(n_cover, n_totoal, 1.0 * n_cover / n_totoal)
ans_list = [x[0] for x in ans_counts]
ans_dict = OrderedDict(zip(ans_list,range(len(ans_list))))
json.dump(ans_dict, open('answer2answer_id.json', 'w'), indent=2)




