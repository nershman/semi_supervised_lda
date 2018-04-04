#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 zc <newvalue92@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Train LDA model
"""


import argparse
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel as LdaModelOld
# from gensim.models.ldamulticore import LdaMulticore
from lda.ldamodel import LdaModel as LdaModelNew
from lda.ldamulticore import LdaMulticore as LdaMulticoreNew
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.mmcorpus import MmCorpus

# setup logging
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


from corpus import LabeledCorpus, IntCorpus, FastTextCorpus
import codecs

def train_model(corpus_path, dic_conf, lda_conf):
    logging.info('Loading corpus from file {}'.format(corpus_path))
    corpus = FastTextCorpus(corpus_path, bufsize=20000000, length=5926250)
    # corpus = LineSentence(corpus_path, 10000000)
    print '-'* 80
    if lda_conf["build_dict"]:
        logging.info("Building dictionary ...")
        dic = Dictionary(corpus)
        dic.filter_extremes(
            no_below=dic_conf["min_tf"],
            no_above=dic_conf["max_df"],
            keep_n=dic_conf["vocab_size"])
        dic.compactify()
        logging.info("Saving dictionary ...")
        dic.save(dic_conf["dic"])
    else:
        logging.info("Loading dictionary ..")
        dic = Dictionary.load(dic_conf["dic"])

    bow = IntCorpus(corpus, dic)
    l = len(bow)
    print l

    tfMod = TfidfModel.load(lda_conf["tfmod"])
    #save corpus to disk for later usage
    # logging.info("Saving corpus to disk ...")
    # MmCorpus.serialize("data/corpus.mm", bow)
    # bow = MmCorpus("data/large_corpus.mm")

    print '-'* 80
    if lda_conf["new"]:
        logging.info("Training new lda model")
        logging.info("Loading defined keywords ...")
        keywords = {}
        topics = []
        with codecs.open(lda_conf["kw_file"], "r", "utf-8") as f:
            for l in f:
                sp = l.strip().split(':')
                topic = int(sp[0])
                topics.append(sp[1])
                kws = sp[2].split(',')
                for kw in kws:
                    if kw not in keywords:
                        keywords[kw] = set([topic])
                    else:
                        keywords[kw].add(topic)
                    #keywords[kw.lower()] = topic

        logging.info("Number of defined keywords: {}".format(len(keywords)))
        if lda_conf["threads"] <=1:
            model = LdaModelNew (
                corpus=bow,
                id2word=dic,
                iterations=lda_conf["iterations"],
                num_topics=lda_conf["num_topics"],
                passes=lda_conf["passes"],
                chunksize=lda_conf["chunksize"],
                defined_kws=keywords,
                alpha='auto',
                eval_every=lda_conf["eval_every"]
            )
        else:
            logging.info("Training model using mutlicore lda version")
            model = LdaMulticoreNew (
                corpus=bow,
                id2word=dic,
                workers=lda_conf["threads"],
                iterations=lda_conf["iterations"],
                num_topics=lda_conf["num_topics"],
                passes=lda_conf["passes"],
                defined_kws=keywords,
                alpha='symmetric',
                chunksize=lda_conf["chunksize"],
                eval_every=lda_conf["eval_every"],
                tfMod=tfMod,
                topic_names=topics
            )

    else:
        logging.info("Training ldamodel implemented in gensim")
        model = LdaModelOld (
            corpus=bow,
            id2word=dic,
            iterations=lda_conf["iterations"],
            num_topics=lda_conf["num_topics"],
            passes=lda_conf["passes"],
            chunksize=lda_conf["chunksize"],
            alpha='auto',
            eval_every=lda_conf["eval_every"]
        )

    logging.info('Saving lda model to {}'.format(lda_conf["model_path"]))
    model.save(lda_conf["model_path"])
    logging.info('Saving model done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training lda model')
    parser.add_argument('-k', "--num_topics", type=int, default=28)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--path", type=str, default="data/processed_data.dat")
    parser.add_argument("--min_tf", type=int, default=5)
    parser.add_argument("--max_df", type=float, default=0.3)
    parser.add_argument("--vocab_size", type=int, default=2000000)
    parser.add_argument("--kw_file", type=str)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--new", action="store_true", dest="new", default=False)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--build_dict", action="store_true", dest="build_dict")
    parser.add_argument("--dic", type=str, default='models/dic_lda_large.mod')
    parser.add_argument("--tfmod", type=str, default='models/tfidf_large.mod')


    args = parser.parse_args()

    lda_conf = {
        "num_topics": args.num_topics,
        "iterations": args.iterations,
        "passes": args.passes,
        "eval_every": args.eval_every,
        "threads": args.threads,
        "kw_file": args.kw_file,
        "new": args.new,
        "model_path": args.model_path,
        "build_dict": args.build_dict,
        "chunksize": args.chunksize,
        "tfmod": args.tfmod
    }
    dic_conf = {
        "min_tf": args.min_tf,
        "max_df": args.max_df,
        "vocab_size": args.vocab_size,
        "dic": args.dic
    }
    train_model(args.path, dic_conf, lda_conf)
