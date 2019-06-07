import os
import wget
import tarfile
import re
import collections
import pandas as pd
import pickle
import numpy as np
from itertools import chain
from konlpy.tag import Twitter

pos_tagger = Twitter()

def tokenize(doc):  ## 리스트형테의 데이터 안에서 리뷰부분을 토크나이징해서 다시 리스트로 저장

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]



def read_data(filename):
    with open(filename, 'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외 #
    return data


def random_row_sampling(df,n): ## 데이터 내에서 몇개 뽑아서 쓸건지. random_sampling해줌
    return df.ix[np.random.random_integers(0,len(df),n)]


def build_word_dict():
        train_data = read_data('movie_data/ratings_train.txt')
        test_data = read_data('movie_data/ratings_test.txt')

        train_tokens = [tokenize(row[1]) for row in train_data]
        test_tokens = [tokenize(row[1]) for row in train_data]

        total_tokens = train_tokens + test_tokens

        pos_tagger = Twitter()

        unlist_tokens = list(chain.from_iterable(total_tokens))



        words = list() ## word index를 주기 위해서 dictionary를 만든다 -> lookup table사용
        for word in unlist_tokens:
            words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word,_ in word_counter:
            word_dict[word] = len(word_dict)

        return word_dict


def build_word_dataset(step, word_dict, document_max_len):
    pos_tagger = Twitter()


    if step == "train":
        train_data = read_data('movie_data/ratings_train.txt')



        train_tokens = [tokenize(row[1]) for row in train_data]
        train_labels = [int(row[2]) for row in train_data]




        df = train_tokens
        df_y = train_labels


        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), df))
        x = list(map(lambda d: d[:50], x))
        x = list(map(lambda d: d + (50- len(d)) * [word_dict["<pad>"]], x))
        y = df_y
        
        return x, y

    else:

        test_data = read_data('movie_data/ratings_test.txt')

        test_tokens = [tokenize(row[1]) for row in test_data]
        test_labels = [int(row[2]) for row in test_data]

        df = test_tokens
        df_y = test_labels

    # Shuffle dataframe

        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), df))
        x = list(map(lambda d: d[:50], x))
        x = list(map(lambda d: d + (50- len(d)) * [word_dict["<pad>"]], x))

        y = df_y

        return x, y




def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
