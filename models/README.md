# CFPB Actions
This project contains code that produces an analysis of the CFPB complaints database, looking specifically at how to identify the product referenced by a complaint using the language of a consumer's experience.

Project includes the following tools and technologies:
- Natural Language Preprocessing - performs tokenization, stemming (using Porter's stemming algorithm)
- _Latent Dirichlet Allocation_ - identifies topics in the data as a dimensionality reduction technique
- Doc2Vec - trains a corpus of words using the gensim library's Doc2Vec class on the words at hand in the document
