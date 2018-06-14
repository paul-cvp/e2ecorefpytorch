

#Added mentions for reimplementation

## Singled out relevant classes for training:
* coref_ops.py - operations originally in C (extended from tensorflow source code), now rewritten 
in python but not tested yet
* util.py - utility functions including the CNN, FFNN and CustomLSTM definitions
* coref_model.py - the model class where the training and evaluation was implemented with tensorflow
* singleton.py - the training wrapper

## Other classes
At the core of this code are the 3 classes: coref_ops.py + util.py + coref_model.py.
Coref_model.py is more tightly bound with coref_ops.py and util.py
The other classes use coref_model.py and some functions from util.py and they are:
* evaluator.py
* test-single.py
* demo.py

## The rest of the code
The remaining classes of this code have not been changed, besides being rewritten for python 3 and pytorch.

### New Requirements
* Python 3.5 or above
  * Pytorch 0.40 (compiled with GPU-cuda)
  * pyhocon (for parsing the configurations)
  * NLTK (for sentence splitting and tokenization in the demo)

# The original End-to-end Neural Coreference Resolution

### Introduction
This repository contains the code for replicating results from

* [End-to-end Neural Coreference Resolution](https://homes.cs.washington.edu/~kentonl/pub/lhlz-emnlp.2017.pdf)
* [Kenton Lee](https://homes.cs.washington.edu/~kentonl), [Luheng He](https://homes.cs.washington.edu/~luheng), [Mike Lewis](https://research.fb.com/people/lewis-mike) and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In Proceedings of the Conference on Empirical Methods in Natural Language Process (EMNLP), 2017


