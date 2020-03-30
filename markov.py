### IMPORTS
import json 
import glob
import string
import random

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import markovify

### CONSTANTS/GLOBALS/LAMBDAS
SYMBOLS_TO_RM = tuple(list(string.punctuation) + ['\xad'])
NUMBERS_TO_RM = tuple(string.digits)

spacy.prefer_gpu()
NLP_ENGINE = spacy.load("en_core_web_sm")

def clean_word(word):
	word_chars = list(word)
	ignore_flag = False

	for s in SYMBOLS_TO_RM:
		if s in word_chars:
			ignore_flag = True
			break

	for n in NUMBERS_TO_RM:
		if n in word_chars:
			ignore_flag = True
			break

	if not ignore_flag and len(word) >= 1:
		return word.lower()
	else:
		return None

def clean_set(raw_set, by_letters=False):
	clean_set = []

	for l in raw_set:
		words = l.split(' ')[:-1]
		clean_sentence = []

		for w in words:
			cleaned_word = None

			if by_letters:
				cw_temp = clean_word(w)
				if cw_temp is None:
					continue
				cleaned_word = cw_temp
			else:
				cleaned_word = clean_word(w)

			if cleaned_word is not None:
				clean_sentence.append(cleaned_word)

		clean_sentence = ' '.join(clean_sentence)
		if clean_sentence != '':
			clean_set.append(clean_sentence)

	return clean_set

def gen_user_corpus(sender, wpath):
	parsed_mesgs = []

	for mesg_corpus_path in glob.glob('message_*.json'):
		with open(mesg_corpus_path) as rjson:
			raw_data = json.load(rjson)

			# parse only textual mesgs from given sender
			for mesg in raw_data['messages']:
				sname = mesg['sender_name']

				if sname == sender:
					text_mesg = mesg.get('content')

					if text_mesg is not None:
						#text_mesg = text_mesg.decode('utf-8')
						parsed_mesgs.append(text_mesg)

	cset = clean_set((pm for pm in parsed_mesgs))

	# derive corpus of only words
	word_set = set()
	for sent in cset:
		words = sent.split(' ')
		for word in words:
			word_set.add(word)

	cset.extend(word_set)

	# generate final corpus
	with open(wpath, 'w+') as corpus:
		for mesg in cset:
			corpus.write(mesg + '\n')

def build_mm_for_user(sender, corpus_path):
	with open(corpus_path, 'r') as corpus:
		cread = corpus.read()
		model = markovify.NewlineText(cread)
		return model.compile()

def gen_valid_sent(model, init_state=None):
	if init_state is not None:
		init_state = ('___BEGIN__', init_state)

	sent = model.make_sentence(init_state=init_state)
	while sent is None:
		sent = model.make_sentence(init_state=init_state)

	return sent

def get_next_sent_subj(sent):
	doc = NLP_ENGINE(sent)
	subj_toks = [tok.text.lower() for tok in doc]
	subj_toks = [NLP_ENGINE.vocab[tok] for tok in subj_toks]
	subj_toks = [tok.text for tok in subj_toks if not tok.is_stop]

	no_stop_str = ' '.join(subj_toks)
	no_stop_doc = NLP_ENGINE(no_stop_str)
	subjs = [tok.text for tok in no_stop_doc if tok.pos_ == 'NOUN']

	if len(subjs) == 0:
		return None
	else:
		return random.choice(subjs)


if __name__ == '__main__':
	mu = gen_user_corpus('Michael Usachenko', 'mu_corpus.txt')
	mu_model = build_mm_for_user('Michael Usachenko', 'mu_corpus.txt')

	js = gen_user_corpus('Jonathan Shobrook', 'js_corpus.txt')
	js_model = build_mm_for_user('Jonathan Shobrook', 'js_corpus.txt')

	# generate starting sentence
	init_sent = gen_valid_sent(mu_model)
	init_subj = get_next_sent_subj(init_sent)

	# WIP: back and forth conversation. need to modify markovify libs
	# works for a few cycles, then errors
	past_init = False
	prior_resp = None

	"""
	for i in range(100):
		if not past_init:
			past_init = True
			js_resp = gen_valid_sent(js_model, init_state=init_subj)
			print('JONATHAN:', js_resp)
			prior_resp = js_resp
		else:
			next_subj = get_next_sent_subj(prior_resp)
			mu_resp = gen_valid_sent(mu_model, init_state=next_subj)
			print('MICHAEL:', mu_resp)

			next_subj = get_next_sent_subj(mu_resp)
			js_resp = gen_valid_sent(js_model, init_state=next_subj)
			print('JONATHAN:', js_resp)
			prior_resp = js_resp
	"""

	for i in range(100):
		#next_subj = get_next_sent_subj(prior_resp)
		mu_resp = gen_valid_sent(mu_model)
		print('MICHAEL:', mu_resp)

		#next_subj = get_next_sent_subj(mu_resp)
		js_resp = gen_valid_sent(js_model)
		print('JONATHAN:', js_resp)
		#prior_resp = js_resp







