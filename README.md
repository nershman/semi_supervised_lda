 # A semi supervised LDA

 ## Why a semi-supervised LDA ?

In many realworld problems, we want to classify document into defined classes. However, building a large corpus can be very costly. LDA is well known algorithm to discover topics in corpus but it. So why we don't learned a LDA that its topics will aligned to our defined topics? My idea is pretty simple. We use pre-defined keywords for each topic to initialize some prior distribution. Better initialization often leads to a better convergence, which is well-known in high dimensional space data.

## Requirements

This model is adapted based on [Gensim's LDA implementation](https://github.com/RaRe-Technologies/gensim). I developed this algorithm several years ago, so it might work only with old version of gensim. You need to run following command to install requirements:

```bash
$ pip install gensim==0.12.4
```

 ## Keyword preparation


 You need to define some keywords for each topic. See [sample](sample/keywords.txt) for the format of this file.

 ## Training a model

 Runing `run_lda_v2.py` to train a new lda model. The full arguments are as bellowing:

 ```
 usage: train_lda_v2.py [-h] [-k NUM_TOPICS] [--passes PASSES]
                       [--iterations ITERATIONS] [--eval_every EVAL_EVERY]
                       [--path PATH] [--min_tf MIN_TF] [--max_df MAX_DF]
                       [--vocab_size VOCAB_SIZE] [--kw_file KW_FILE]
                       [--threads THREADS] [--chunksize CHUNKSIZE] [--new]
                       --model_path MODEL_PATH [--build_dict] [--dic DIC]
                       [--tfmod TFMOD]

Training lda model

optional arguments:
  -h, --help            show this help message and exit
  -k NUM_TOPICS, --num_topics NUM_TOPICS
  --passes PASSES
  --iterations ITERATIONS
  --eval_every EVAL_EVERY
  --path PATH
  --min_tf MIN_TF
  --max_df MAX_DF
  --vocab_size VOCAB_SIZE
  --kw_file KW_FILE
  --threads THREADS
  --chunksize CHUNKSIZE
  --new
  --model_path MODEL_PATH
  --build_dict
  --dic DIC
  --tfmod TFMOD

 ```

## Disclaimer
 This algorithm works with my case, but it might not work with your case. And I won't support any technical problems related to this library. So use it at your own risk.
 
 Hope it can help others to develop new algorithms :).
