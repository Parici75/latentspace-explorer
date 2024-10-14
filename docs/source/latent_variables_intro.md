# What are latent variables?
*A high-level introduction to the concept.*

Latent variables are "hidden" factors that underly the relationship between observed variables. They are not directly measurable or observable, but they can be inferred by making sense of the patterns in the data. These variables capture underlying structures within the data that are not immediately apparent, and can be thought of as intermediate layers of explanation between what we observe, and the underlying mechanisms that govern the behavior of a system.

A down to earth illustration of the concept can be found in the way we assess personnality of our fellow human beings: In real-world setting, we are never able to directly measure someone's personality. Rather, we infer certain personality characteristics using cues such as their outlook, behavior, interests, interactions with others, and so on. In this case, the unobservable personality traits (e.g. agreableness) are the latent variables that underlie the patterns of behaviour we can observe on a day to day basis (e.g., the person's cheerfulness, generosity, irritability, etc). Note that this is not a problem of competency, tools, or point of view : seasoned psychatrists are only able to *infer* personnality traits from a *quantitative analysis* of the patterns of scores to various personality assessement questionaires and psychological tests.


## Why are latent variables fundamental in Statistics and Machine learning?
Latent variables are everywhere, including, *ipso facto* in Statistics and Machine learning.

Working out the latent variables of a dataset generally serves two distinct, yet possibly concomitant, purposes:
- **Reducing** the dimensionality of the feature space. The ultimate goal in this case is to obtain a *better representation* of the dataset, whether it be to decipher hidden patterns through analysis (e.g., by clustering or segmentation), or improve the quality of linear modeling (e.g., by reducing the degree of freedom available to the model, capturing "hidden" relationship between predictors or redistributing variance to supress multicolinearity). Techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), Non-negative Matrix Factorization (NMF), and vanilla Autoencoders extract such latent variables to reduce the dimensionality of high-dimensional data.
- Learning **new representations** from the original feature space. Here the goal is also to obtain a *better representation* of the dataset by mapping original variables onto latent variables, but any potential compression that comes with it is incidental. In fact, the compression (or expansion) ratio depends on the nature of the data, learning task, and statistical choices.


## Compression or inflation of the feature space?
With images, for example, which are inherently high dimensional, convolutional neural network (CNN) architectures tend to produce lower dimension latent space: if we consider the most high-level *feature maps* of CNNs to be the latent variables encoding relevant characteristics for the learning task at stake, the original LeNet5[^1] implementation indeed compresses 32x32 (=1024) features into a high-level, low-dimensional space of size 5x5x10 (=400), that is, a compression rate of ~2.5. As architecture becomes more complex however, this compression rate drops: for example, VGG16 compresses images of size 224x224x3(=150528) into a latent representation of size 14x14x512 (=100352), which amounts to a compression rate of "only" 1.5.

In pratice, the information is always compressed within CNN, because these architectures are designed to extract deep, hierarchical representations of the original pixel space by stacking more and more convolutional layers, while limiting the "width" of the layers (i.e., the number of convolutional filters) to avoid overfitting.

But Transformer architectures provide an extreme example of the opposite procedure: inflating the dimensionality of the feature space. Indeed, from an original unidimensional feature space constitued of *tokens* (where one word or *subword* is mapped onto a numeric value), a large language encoder model like BERT produces embeddings of size 1024 (BERT Large) or 768 (BERT Base)[^2]. These embeddings are the hidden latent variables encoding at once, the denotation of the word and its semantic context, hence its meaning.

These deep embeddings solve the problem that the meaning of the word can not be directly inferred from the sole occurence of a word in a sentence: meaning depends not only on the denotation of the word, but also on the surrounding words, the syntactic structure, and even the broader context endowed by the preceding and subsequent sentences.

Large language model's embeddings are thus the hidden latent variables of words that reflect how they are used by humans to convey meaning.

Similarly, Matrix Factorization techniques used in Recommender services[^3] build user and item latent spaces which encode the proximity between them, thus allowing to make recommendation on the straightforward and efficient assumptions that 1) item interactions define user similarity and 2) similar users share taste.

As Data scientists / ML engineers, our job is to exploit these latent variables, explicitely or under the hood, to solve a business problem.

References
-
[^1]: [Lecun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE. 86 (11): 2278–2324](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

[^2]: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

[^3]: [Tutorial on Collaborative Filtering and Matrix Factorization in Python](https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/)
