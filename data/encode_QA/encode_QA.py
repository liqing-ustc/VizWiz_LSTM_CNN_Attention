import json
import numpy as np
import nltk
import os
from tqdm import *
from collections import Counter


def encode_sentence(sentence, vocab):
	unk_word = '<UNK>'
	tokens = nltk.word_tokenize(sentence.lower())
	tokens_id = [vocab.get(x, vocab[unk_word]) for x in tokens]
	return tokens_id

root_dir = os.environ['data_dir']
splits = ['train', 'val', 'test']


vocab = json.load(open('../create_vocab/word2vocab_id.json'))
answer2answer_id = json.load(open('../create_vocab/answer2answer_id.json'))
all_imgs = json.load(open('../extract_feat/all_imgs.json'))
image_id2image_feat_id = {img:idx for idx, img in enumerate(all_imgs)}

missing_images = []

for split in splits:
	## load annotation file and question file
	dataset = json.load(open(root_dir + 'Annotations/%s.json'%split))

	## encode QA
	question_id = []
	image_id = []
	image_feat_id = [] # this is the line number of the image feat matrix
	question_encode = []
	answer_label = []
	answerable = []
	
	for one_data in tqdm(dataset):
		ans_counter = Counter([x['answer'] for x in one_data['answers']])
		ans = ans_counter.most_common(1)[0][0]
		a_label = answer2answer_id.get(ans, -1)
		if split[0] == 'train' and a_label == -1:
			continue
			
		i_id = one_data['image']
		i_feat_id = image_id2image_feat_id[str(i_id)]
		q_id = ''
		q_encode = encode_sentence(one_data['question'], vocab) # remove the '?' at the end
		
		question_id.append(q_id)
		image_id.append(i_id)
		image_feat_id.append(i_feat_id)
		question_encode.append(q_encode)
		answer_label.append(a_label)
		answerable.append(one_data['answerable'])

        all_data = {'image_id': image_id, 'image_feat_id': image_feat_id, 'answerable':answerable, 
				'question': question_encode, 'answer': answer_label}
	json_file = r'%s.json'%split
	json.dump(all_data, open(json_file, 'w'))
