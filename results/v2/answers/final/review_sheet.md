# Review sheet: results/v2/answers/final

## final-ans00 (answerable, expected: answer)

**Q:** What manual Pyramid scores are used?  
**Paper:** Revisiting Summarization Evaluation for Scientific Articles

- REF:  higher tiers of the pyramid
- REF: following the pyramid framework, we design an annotation scheme

**Answer** (answered): The manual Pyramid scores are used as semi-manual evaluation scores in the TAC 2014 scientific summarization dataset [4]. These scores are based on the pyramid framework, where content units (nuggets) are organized in tiers representing their importance [2].

- claim0 cites [4]: The manual Pyramid scores are used as semi-manual evaluation scores in the TAC 2014 scientific summarization dataset
  - [4] Revisiting Summarization Evaluation for Scientific Articles / Introduction: maries for scientific articles. Scientific articles are much different than news articles in elements such as length, complexity and structure. Thus, effective summarization approaches usually have much higher compression rate, terminology variations and paraphrasing BIBREF4 .

Scientific summarization has attracted more attention recently (examples include works by abu2011coherent, qazvinian2013generating, and cohan2015scientific). Thus, it is important to study the validity of existing methodologies applied to the evaluation of news article summarization for this task. In particular, we raise the important question of how effective is Rouge, as an evaluation metric for scientific summarization? We answer this question by comparing Rouge scores with semi-manual evaluation score (Pyramid) in TAC 2014 scientific summarization dataset[1]. Results reveal that, contrary to the common belief, correlations between Rouge and the Pyramid scores are weak, which challenges its effectiveness for scientific summarization. Furthermore, we show a large variance of correlations between different Rouge variants and the manual evaluations which further makes the reliability of Rouge for evaluating scientific summaries less clear. We then propose an evaluation metric based on relevance analysis of summaries which aims to overcome the limitation of high lexical dependence in Rouge. We call our metric Sera (Summarization Evaluation by Relevance Analysis). Results show that the proposed metric achieves higher and more consistent correlations with semi-manual assessment scores.

[1]http://www.nist.gov/tac/2014/BiomedSumm/

Our contributions are as follows:

[2]The annotations can be accessed via the following repository: https://github.com/acohan/TAC-pyramid-Annotations/
- claim1 cites [2]: These scores are based on the pyramid framework, where content units (nuggets) are organized in tiers representing their importance
  - [2] Revisiting Summarization Evaluation for Scientific Articles / Annotations: In the TAC 2014 summarization track, Rouge was suggested as the evaluation metric for summarization and no human assessment was provided for the topics. Therefore, to study the effectiveness of the evaluation metrics, we use the semi-manual Pyramid evaluation framework BIBREF7 , BIBREF8 . In the pyramid scoring, the content units in the gold human written summaries are organized in a pyramid. In this pyramid, the content units are organized in tiers and higher tiers of the pyramid indicate higher importance. The content quality of a given candidate summary is evaluated with respect to this pyramid.

To analyze the quality of the evaluation metrics, following the pyramid framework, we design an annotation scheme that is based on identification of important content units. Consider the following example:

Endogeneous small RNAs (miRNA) were genetically screened and studied to find the miRNAs which are related to tumorigenesis.

In the above example, the underlined expressions are the content units that convey the main meaning of the text. We call these small units, nuggets which are phrases or concepts that are the main contributors to the content quality of the summary.

We asked two human annotators to review the gold summaries and extract content units in these summaries. The pyramid tiers represent the occurrences of nuggets across all the human written gold-standard summaries, and therefore the nuggets are weighted based on these tiers. The intuition is that, if a nugget occurs more frequently in the human summaries, it is a more important contributor (thus belongs to higher tier in the pyramid). Thus, if a candidate summary contains this nugget, it should be rewarded more. An example of the nuggets annotations in pyramid framework is shown in Table TABREF12 . In this example, the nugget “cell mutation” belongs to the 4th tier and it suggests that the “cell mutation” nugget is a very important representative of the content of the corresponding document.

## final-ans01 (answerable, expected: answer)

**Q:** What is the combination of rewards for reinforcement learning?  
**Paper:** A Neural Approach to Irony Generation

- REF: irony accuracy, sentiment preservation
- REF:  irony accuracy and sentiment preservation

**Answer** (answered): The combination of rewards for reinforcement learning includes the irony reward and the sentiment preservation reward [2].

- claim0 cites [2]: The combination of rewards for reinforcement learning includes the irony reward and the sentiment preservation reward
  - [2] A Neural Approach to Irony Generation / Reinforcement Learning: Since the gold transferred result of input is unavailable, we cannot evaluate the quality of the generated sentence directly. Therefore, we implement reinforcement learning and elaborately design two rewards to describe the irony accuracy and sentiment preservation, respectively.

A pre-trained binary irony classifier based on CNN BIBREF20 is used to evaluate how ironic a sentence is. We denote the parameter of the classifier as INLINEFORM0 and it is fixed during the training process.

In order to facilitate the transformation, we design the irony reward as the difference between the irony score of the input sentence and that of the output sentence. Formally, when we input a non-ironic sentence INLINEFORM0 and transfer it to an ironic sentence INLINEFORM1 , our irony reward is defined as: DISPLAYFORM0

where INLINEFORM0 denotes ironic style and INLINEFORM1 is the probability of that a sentence INLINEFORM2 is ironic.

To preserve the sentiment polarity of the input sentence, we also need to use classifiers to evaluate the sentiment polarity of the sentences. However, the sentiment analysis of ironic sentences and non-ironic sentences are different. In the case of figurative languages such as irony, sarcasm or metaphor, the sentiment polarity of the literal meaning may differ significantly from that of the intended figurative meaning BIBREF0 . As we aim to train our model to transfer sentences from non-ironic to ironic, using only one classifier is not enough. As a result, we implement two pre-trained sentiment classifiers for non-ironic sentences and ironic sentences respectively. We denote the parameter of the sentiment classifier for ironic sentences as INLINEFORM0 and that of the sentiment classifier for non-ironic sentences as INLINEFORM1 .

## final-ans02 (answerable, expected: answer)

**Q:** How do they measure which words are under-translated by NMT models?  
**Paper:** Towards Understanding Neural Machine Translation with Word Importance

- REF: They measured the under-translated words with low word importance score as calculated by Attribution.
method
- REF: we ask ten human annotators to manually label the under-translated input words, and at least two annotators label each input-hypothesis pair

**Answer** (answered): They measure under-translated words by comparing the importance of words calculated using the gradient-based method with human-annotated under-translated words [1]. The method identifies under-translated words as those with much smaller word importance [1].

- claim0 cites [1]: They measure under-translated words by comparing the importance of words calculated using the gradient-based method with human-annotated under-translated words
  - [1] Towards Understanding Neural Machine Translation with Word I / Analysis > Effect on Detecting Translation Errors: In this experiment, we propose to use the estimated word importance to detect the under-translated words by NMT models. Intuitively, under-translated input words should contribute little to the NMT outputs, yielding much smaller word importance. Given 500 Chinese$\Rightarrow $English sentence pairs translated by the Transformer model (BLEU 23.57), we ask ten human annotators to manually label the under-translated input words, and at least two annotators label each input-hypothesis pair. These annotators have at least six years of English study experience, whose native language is Chinese. Among these sentences, 178 sentences have under-translation errors with 553 under-translated words in total.

Table TABREF32 lists the accuracy of detecting under-translation errors by comparing words of least importance and human-annotated under-translated words. As seen, our Attribution method consistently and significantly outperforms both Erasure and Attention approaches. By exploiting the word importance calculated by Attribution method, we can identify the under-translation errors automatically without the involvement of human interpreters. Although the accuracy is not high, it is worth noting that our under-translation method is very simple and straightforward. This is potentially useful for debugging NMT models, e.g., automatic post-editing with constraint decoding BIBREF26, BIBREF27.
- claim1 cites [1]: The method identifies under-translated words as those with much smaller word importance
  - [1] Towards Understanding Neural Machine Translation with Word I / Analysis > Effect on Detecting Translation Errors: In this experiment, we propose to use the estimated word importance to detect the under-translated words by NMT models. Intuitively, under-translated input words should contribute little to the NMT outputs, yielding much smaller word importance. Given 500 Chinese$\Rightarrow $English sentence pairs translated by the Transformer model (BLEU 23.57), we ask ten human annotators to manually label the under-translated input words, and at least two annotators label each input-hypothesis pair. These annotators have at least six years of English study experience, whose native language is Chinese. Among these sentences, 178 sentences have under-translation errors with 553 under-translated words in total.

Table TABREF32 lists the accuracy of detecting under-translation errors by comparing words of least importance and human-annotated under-translated words. As seen, our Attribution method consistently and significantly outperforms both Erasure and Attention approaches. By exploiting the word importance calculated by Attribution method, we can identify the under-translation errors automatically without the involvement of human interpreters. Although the accuracy is not high, it is worth noting that our under-translation method is very simple and straightforward. This is potentially useful for debugging NMT models, e.g., automatic post-editing with constraint decoding BIBREF26, BIBREF27.

## final-ans03 (answerable, expected: answer)

**Q:** what kinds of male and female words are looked at?  
**Paper:** Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function

- REF: gendered word pairs like he and she

**Answer** (answered): The paper looks at gender-neutral occupations and gender-neutral words. These are used to evaluate co-occurrence bias and causal occupation bias [1][2].

- claim0 cites []: The paper looks at gender-neutral occupations and gender-neutral words
- claim1 cites [1, 2]: These are used to evaluate co-occurrence bias and causal occupation bias
  - [1] Reducing Gender Bias in Word-Level Language Models with a Ge / Model Evaluation: Language models are evaluated using perplexity, which is a standard measure of performance for unseen data. For bias evaluation, we use an array of metrics to provide a holistic diagnosis of the model behavior under debiasing treatment. These metrics are discussed in detail below. In all the evaluation metrics requiring gender pairs, we use gender pairs provided by BIBREF7 . This list contains 223 pairs, all other words are considered gender-neutral.

Co-occurrence bias is computed from the model-generated texts by comparing the occurrences of all gender-neutral words with female and male words. A word is considered to be biased towards a certain gender if it occurs more frequently with words of that gender. This definition was first used by BIBREF7 and later adapted by BIBREF5 . Using the definition of gender bias similar to the one used by BIBREF5 , we define gender bias as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral words, and INLINEFORM1 is the occurrences of a word INLINEFORM2 with words of gender INLINEFORM3 in the same window. This score is designed to capture unequal co-occurrences of neutral words with male and female words. Co-occurrences are computed using a sliding window of size 10 extending equally in both directions. Furthermore, we only consider words that occur more than 20 times with gendered words to exclude random effects.

We also evaluate a normalized version of INLINEFORM0 which we denote by conditional co-occurrence bias, INLINEFORM1 . This is defined as INLINEFORM2

where INLINEFORM0

INLINEFORM0 is less affected by the disparity in the general distribution of male and female words in the text. The disparity between the occurrences of the two genders means that text is more inclined to mention one over the other, so it can also be considered a form of bias. We report the ratio of occurrence of male and female words in the model generated text, INLINEFORM1 , as INLINEFORM2
  - [2] Reducing Gender Bias in Word-Level Language Models with a Ge / Model Evaluation: s means that text is more inclined to mention one over the other, so it can also be considered a form of bias. We report the ratio of occurrence of male and female words in the model generated text, INLINEFORM1 , as INLINEFORM2

Another way of quantifying bias in NLP models is based on the idea of causal testing. The model is exposed to paired samples which differ only in one attribute (e.g. gender) and the disparity in the output is interpreted as bias related to that attribute. BIBREF1 and BIBREF0 applied this method to measure bias in coreference resolution and BIBREF0 also used it for evaluating gender bias in language modelling.

Following the approach similar to BIBREF0 , we limit this bias evaluation to a set of gender-neutral occupations. We create a list of sentences based on a set of templates. There are two sets of templates used for evaluating causal occupation bias (Table TABREF7 ). The first set of templates is designed to measure how the probabilities of occupation words depend on the gender information in the seed. Below is an example of the first set of templates: INLINEFORM0

Here, the vertical bar separates the seed sequence that is fed into the language models from the target occupation, for which we observe the output softmax probability. We measure causal occupation bias conditioned on gender as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral occupations and INLINEFORM1 is the size of the gender pairs set. For example, INLINEFORM2 is the softmax probability of the word INLINEFORM3 where the seed sequence is He is a. The second set of templates like below, aims to capture how the probabilities of gendered words depend on the occupation words in the seed. INLINEFORM4

Causal occupation bias conditioned on occupation is represented as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral occupations and INLINEFORM1 is the size of the gender pairs set. For example, INLINEFORM2 is the softmax probability of man where the seed sequence is The doctor is a.

## final-ans04 (answerable, expected: answer)

**Q:** How is the dataset of hashtags sourced?  
**Paper:** Multi-task Pairwise Neural Ranking for Hashtag Segmentation

- REF: 1,268 randomly selected tweets in the Stanford Sentiment Analysis Dataset BIBREF36, all 12,594 unique English hashtags and their associated tweets from the same Stanford dataset
- REF: Stanford Sentiment Analysis Dataset BIBREF36

**Answer** (answered): The dataset of hashtags is sourced from the Stanford Sentiment Analysis Dataset. Specifically, it includes 1,108 unique English hashtags from 1,268 randomly selected tweets along with their crowdsourced segmentations and additional corrections, and 12,594 unique English hashtags and their associated tweets [1].

- claim0 cites []: The dataset of hashtags is sourced from the Stanford Sentiment Analysis Dataset
- claim1 cites [1]: Specifically, it includes 1,108 unique English hashtags from 1,268 randomly selected tweets along with their crowdsourced segmentations and additional corrections, and 12,594 unique English hashtags and their associated tweets
  - [1] Multi-task Pairwise Neural Ranking for Hashtag Segmentation / Hashtag Segmentation Data: We use two datasets for experiments (Table TABREF29 ): (a) STAN INLINEFORM0 , created by BIBREF10 BansalBV15, which consists of 1,108 unique English hashtags from 1,268 randomly selected tweets in the Stanford Sentiment Analysis Dataset BIBREF36 along with their crowdsourced segmentations and our additional corrections; and (b) STAN INLINEFORM1 , our new expert curated dataset, which includes all 12,594 unique English hashtags and their associated tweets from the same Stanford dataset.

## final-ans05 (answerable, expected: answer)

**Q:** Which annotated corpus did they use?  
**Paper:** Artificial Error Generation with Machine Translation and Syntactic Patterns

- REF:  FCE test data (41K tokens) and the two alternative annotations of the CoNLL 2014 Shared Task dataset (30K tokens) 
- REF: FCE ,  two alternative annotations of the CoNLL 2014 Shared Task dataset

**Answer** (answered): The annotated corpus used was the FCE and CoNLL 2014 datasets [2].

- claim0 cites [2]: The annotated corpus used was the FCE and CoNLL 2014 datasets
  - [2] Artificial Error Generation with Machine Translation and Syn / Conclusion: This paper investigated two AEG methods, in order to create additional training data for error detection. First, we explored a method using textual patterns learned from an annotated corpus, which are used for inserting errors into correct input text. In addition, we proposed formulating error generation as an MT framework, learning to translate from grammatically correct to incorrect sentences.

The addition of artificial data to the training process was evaluated on three error detection annotations, using the FCE and CoNLL 2014 datasets. Making use of artificial data provided improvements for all data generation methods. By relaxing the type restrictions and generating all types of errors, our pattern-based method consistently outperformed the system by Felice2014a. The combination of the pattern-based method with the machine translation approach gave further substantial improvements and the best performance on all datasets.

## final-ans06 (answerable, expected: answer)

**Q:** How many tweets did they collect?  
**Paper:** Stance Detection in Turkish Tweets

- REF: 700 
- REF: 700

**Answer** (answered): 700 tweets [2].

- claim0 cites [2]: 700 tweets
  - [2] Stance Detection in Turkish Tweets / A Stance Detection Data Set: the stance was the management of the club while in some others a particular footballer of the club is praised or criticised. Still, we have considered the club as the target of the stance in all of the cases and carried out our annotations accordingly.

At the end of the annotation process, we have annotated 700 tweets, where 175 tweets are in favor of and 175 tweets are against Target-1, and similarly 175 tweets are in favor of and 175 are against Target-2. Hence, our data set is a balanced one although it is currently limited in size. The corresponding stance annotations are made publicly available at http://ceng.metu.edu.tr/ INLINEFORM0 e120329/ Turkish_Stance_Detection_Tweet_Dataset.csv in Comma Separated Values (CSV) format. The file contains three columns with the corresponding headers. The first column is the tweet id of the corresponding tweet, the second column contains the name of the stance target, and the last column includes the stance of the tweet for the target as Favor or Against.

To the best of our knowledge, this is the first publicly-available stance-annotated data set for Turkish. Hence, it is a significant resource as there is a scarcity of annotated data sets, linguistic resources, and NLP tools available for Turkish. Additionally, to the best of our knowledge, it is also significant for being the first stance-annotated data set including sports-related tweets, as previous stance detection data sets mostly include on-line texts on political/ethical issues.

## final-ans07 (answerable, expected: answer)

**Q:** What other sentence embeddings methods are evaluated?  
**Paper:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

- REF: GloVe, BERT, Universal Sentence Encoder, TF-IDF, InferSent
- REF: Avg. GloVe embeddings, Avg. fast-text embeddings, Avg. BERT embeddings, BERT CLS-vector, InferSent - GloVe and Universal Sentence Encoder.

**Answer** (answered): InferSent [4], Universal Sentence Encoder [5], and averaging GloVe embeddings [2] are the other sentence embeddings methods evaluated.

- claim0 cites [2, 4, 5]: InferSent , Universal Sentence Encoder , and averaging GloVe embeddings  are the other sentence embeddings methods evaluated
  - [2] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Introduction: beddings) or by using the output of the first token (the [CLS] token). As we will show, this common practice yields rather bad sentence embeddings, often worse than averaging GloVe embeddings BIBREF2.

To alleviate this issue, we developed SBERT. The siamese network architecture enables that fixed-sized vectors for input sentences can be derived. Using a similarity measure like cosine-similarity or Manhatten / Euclidean distance, semantically similar sentences can be found. These similarity measures can be performed extremely efficient on modern hardware, allowing SBERT to be used for semantic similarity search as well as for clustering. The complexity for finding the most similar sentence pair in a collection of 10,000 sentences is reduced from 65 hours with BERT to the computation of 10,000 sentence embeddings (5 seconds with SBERT) and computing cosine-similarity (0.01 seconds). By using optimized index structures, finding the most similar Quora question can be reduced from 50 hours to a few milliseconds BIBREF3.

We fine-tune SBERT on NLI data, which creates sentence embeddings that significantly outperform other state-of-the-art sentence embedding methods like InferSent BIBREF4 and Universal Sentence Encoder BIBREF5. On seven Semantic Textual Similarity (STS) tasks, SBERT achieves an improvement of 11.7 points compared to InferSent and 5.5 points compared to Universal Sentence Encoder. On SentEval BIBREF6, an evaluation toolkit for sentence embeddings, we achieve an improvement of 2.1 and 2.6 points, respectively.

SBERT can be adapted to a specific task. It sets new state-of-the-art performance on a challenging argument similarity dataset BIBREF7 and on a triplet dataset to distinguish sentences from different sections of a Wikipedia article BIBREF8.
  - [4] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Related Work: ceembeddings3). These two options are also provided by the popular bert-as-a-service-repository. Up to our knowledge, there is so far no evaluation if these methods lead to useful sentence embeddings.

Sentence embeddings are a well studied area with dozens of proposed methods. Skip-Thought BIBREF12 trains an encoder-decoder architecture to predict the surrounding sentences. InferSent BIBREF4 uses labeled data of the Stanford Natural Language Inference dataset BIBREF13 and the Multi-Genre NLI dataset BIBREF14 to train a siamese BiLSTM network with max-pooling over the output. Conneau et al. showed, that InferSent consistently outperforms unsupervised methods like SkipThought. Universal Sentence Encoder BIBREF5 trains a transformer network and augments unsupervised learning with training on SNLI. hill-etal-2016-learning showed, that the task on which sentence embeddings are trained significantly impacts their quality. Previous work BIBREF4, BIBREF5 found that the SNLI datasets are suitable for training sentence embeddings. yang-2018-learning presented a method to train on conversations from Reddit using siamese DAN and siamese transformer networks, which yielded good results on the STS benchmark dataset.

polyencoders addresses the run-time overhead of the cross-encoder from BERT and present a method (poly-encoders) to compute a score between $m$ context vectors and pre-computed candidate embeddings using attention. This idea works for finding the highest scoring sentence in a larger collection. However, poly-encoders have the drawback that the score function is not symmetric and the computational overhead is too large for use-cases like clustering, which would require $O(n^2)$ score computations.
  - [5] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Evaluation - Semantic Textual Similarity > Unsupervised STS: We evaluate the performance of SBERT for STS without using any STS specific training data. We use the STS tasks 2012 - 2016 BIBREF16, BIBREF17, BIBREF18, BIBREF19, BIBREF20, the STS benchmark BIBREF10, and the SICK-Relatedness dataset BIBREF21. These datasets provide labels between 0 and 5 on the semantic relatedness of sentence pairs. We showed in BIBREF22 that Pearson correlation is badly suited for STS. Instead, we compute the Spearman's rank correlation between the cosine-similarity of the sentence embeddings and the gold labels. The setup for the other sentence embedding methods is equivalent, the similarity is computed by cosine-similarity. The results are depicted in Table TABREF6.

The results shows that directly using the output of BERT leads to rather poor performances. Averaging the BERT embeddings achieves an average correlation of only 54.81, and using the CLS-token output only achieves an average correlation of 29.19. Both are worse than computing average GloVe embeddings.

Using the described siamese network structure and fine-tuning mechanism substantially improves the correlation, outperforming both InferSent and Universal Sentence Encoder substantially. The only dataset where SBERT performs worse than Universal Sentence Encoder is SICK-R. Universal Sentence Encoder was trained on various datasets, including news, question-answer pages and discussion forums, which appears to be more suitable to the data of SICK-R. In contrast, SBERT was pre-trained only on Wikipedia (via BERT) and on NLI data.

While RoBERTa was able to improve the performance for several supervised tasks, we only observe minor difference between SBERT and SRoBERTa for generating sentence embeddings.

## final-ans08 (answerable, expected: answer)

**Q:** How are the siamese networks trained?  
**Paper:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

- REF: update the weights such that the produced sentence embeddings are semantically meaningful and can be compared with cosine-similarity., Classification Objective Function, Regression Objective Function, Triplet Objective Function

**Answer** (answered): The siamese networks are trained using different objective functions, including the classification, regression, and triplet objective functions. For the regression objective function, cosine-similarity between the sentence embeddings is computed, and mean-squared-error loss is used as the objective function [3]. For the classification objective function, the element-wise difference $|u-v|$ is concatenated with the sentence embeddings $u$ and $v$, and cross-entropy loss is optimized [3]. The training data and objective function depend on the annotated dataset, with the regression objective function trained on the STS benchmark dataset [4].

- claim0 cites []: The siamese networks are trained using different objective functions, including the classification, regression, and triplet objective functions
- claim1 cites [3]: For the regression objective function, cosine-similarity between the sentence embeddings is computed, and mean-squared-error loss is used as the objective function
  - [3] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Model: SBERT adds a pooling operation to the output of BERT / RoBERTa to derive a fixed sized sentence embedding. We experiment with three pooling strategies: Using the output of the CLS-token, computing the mean of all output vectors (MEAN-strategy), and computing a max-over-time of the output vectors (MAX-strategy). The default configuration is MEAN.

In order to fine-tune BERT / RoBERTa, we create siamese and triplet networks BIBREF15 to update the weights such that the produced sentence embeddings are semantically meaningful and can be compared with cosine-similarity.

The network structure depends on the available training data. We experiment with the following structures and objective functions.

Classification Objective Function. We concatenate the sentence embeddings $u$ and $v$ with the element-wise difference $|u-v|$ and multiply it with the trainable weight $W_t \in \mathbb {R}^{3n \times k}$:

where $n$ is the dimension of the sentence embeddings and $k$ the number of labels. We optimize cross-entropy loss. This structure is depicted in Figure FIGREF4.

Regression Objective Function. The cosine-similarity between the two sentence embeddings $u$ and $v$ is computed (Figure FIGREF5). We use mean-squared-error loss as the objective function.

Triplet Objective Function. Given an anchor sentence $a$, a positive sentence $p$, and a negative sentence $n$, triplet loss tunes the network such that the distance between $a$ and $p$ is smaller than the distance between $a$ and $n$. Mathematically, we minimize the following loss function:
- claim2 cites [3]: For the classification objective function, the element-wise difference $|u-v|$ is concatenated with the sentence embeddings $u$ and $v$, and cross-entropy loss is optimized
  - [3] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Model: SBERT adds a pooling operation to the output of BERT / RoBERTa to derive a fixed sized sentence embedding. We experiment with three pooling strategies: Using the output of the CLS-token, computing the mean of all output vectors (MEAN-strategy), and computing a max-over-time of the output vectors (MAX-strategy). The default configuration is MEAN.

In order to fine-tune BERT / RoBERTa, we create siamese and triplet networks BIBREF15 to update the weights such that the produced sentence embeddings are semantically meaningful and can be compared with cosine-similarity.

The network structure depends on the available training data. We experiment with the following structures and objective functions.

Classification Objective Function. We concatenate the sentence embeddings $u$ and $v$ with the element-wise difference $|u-v|$ and multiply it with the trainable weight $W_t \in \mathbb {R}^{3n \times k}$:

where $n$ is the dimension of the sentence embeddings and $k$ the number of labels. We optimize cross-entropy loss. This structure is depicted in Figure FIGREF4.

Regression Objective Function. The cosine-similarity between the two sentence embeddings $u$ and $v$ is computed (Figure FIGREF5). We use mean-squared-error loss as the objective function.

Triplet Objective Function. Given an anchor sentence $a$, a positive sentence $p$, and a negative sentence $n$, triplet loss tunes the network such that the distance between $a$ and $p$ is smaller than the distance between $a$ and $n$. Mathematically, we minimize the following loss function:
- claim3 cites [4]: The training data and objective function depend on the annotated dataset, with the regression objective function trained on the STS benchmark dataset
  - [4] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networ / Ablation Study: We have demonstrated strong empirical results for the quality of SBERT sentence embeddings. In this section, we perform an ablation study of different aspects of SBERT in order to get a better understanding of their relative importance.

We evaluated different pooling strategies (MEAN, MAX, and CLS). For the classification objective function, we evaluate different concatenation methods. For each possible configuration, we train SBERT with 10 different random seeds and average the performances.

The objective function (classification vs. regression) depends on the annotated dataset. For the classification objective function, we train SBERT-base on the SNLI and the Multi-NLI dataset. For the regression objective function, we train on the training set of the STS benchmark dataset. Performances are measured on the development split of the STS benchmark dataset. Results are shown in Table TABREF23.

When trained with the classification objective function on NLI data, the pooling strategy has a rather minor impact. The impact of the concatenation mode is much larger. InferSent BIBREF4 and Universal Sentence Encoder BIBREF5 both use $(u, v, |u-v|, u*v)$ as input for a softmax classifier. However, in our architecture, adding the element-wise $u*v$ decreased the performance.

The most important component is the element-wise difference $|u-v|$. Note, that the concatenation mode is only relevant for training the softmax classifier. At inference, when predicting similarities for the STS benchmark dataset, only the sentence embeddings $u$ and $v$ are used in combination with cosine-similarity. The element-wise difference measures the distance between the dimensions of the two sentence embeddings, ensuring that similar pairs are closer and dissimilar pairs are further apart.

## final-ans09 (answerable, expected: answer)

**Q:** Which languages are explored in this paper?  
**Paper:** Artificial Error Generation with Machine Translation and Syntactic Patterns

- REF: English 
- REF: English 

**Answer** (model_declined): I can't find this in the papers.


## final-ans10 (answerable, expected: answer)

**Q:** what bias evaluation metrics are used?  
**Paper:** Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function

- REF: gender bias, normalized version of INLINEFORM0, ratio of occurrence of male and female words in the model generated text, Causal occupation bias conditioned on occupation, causal occupation bias conditioned on gender, INLINEFORM1

**Answer** (answered): Perplexity and an array of metrics to provide a holistic diagnosis of the model behavior under debiasing treatment [1]. These include co-occurrence bias and causal occupation bias [1][3].

- claim0 cites [1]: Perplexity and an array of metrics to provide a holistic diagnosis of the model behavior under debiasing treatment
  - [1] Reducing Gender Bias in Word-Level Language Models with a Ge / Model Evaluation: Language models are evaluated using perplexity, which is a standard measure of performance for unseen data. For bias evaluation, we use an array of metrics to provide a holistic diagnosis of the model behavior under debiasing treatment. These metrics are discussed in detail below. In all the evaluation metrics requiring gender pairs, we use gender pairs provided by BIBREF7 . This list contains 223 pairs, all other words are considered gender-neutral.

Co-occurrence bias is computed from the model-generated texts by comparing the occurrences of all gender-neutral words with female and male words. A word is considered to be biased towards a certain gender if it occurs more frequently with words of that gender. This definition was first used by BIBREF7 and later adapted by BIBREF5 . Using the definition of gender bias similar to the one used by BIBREF5 , we define gender bias as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral words, and INLINEFORM1 is the occurrences of a word INLINEFORM2 with words of gender INLINEFORM3 in the same window. This score is designed to capture unequal co-occurrences of neutral words with male and female words. Co-occurrences are computed using a sliding window of size 10 extending equally in both directions. Furthermore, we only consider words that occur more than 20 times with gendered words to exclude random effects.

We also evaluate a normalized version of INLINEFORM0 which we denote by conditional co-occurrence bias, INLINEFORM1 . This is defined as INLINEFORM2

where INLINEFORM0

INLINEFORM0 is less affected by the disparity in the general distribution of male and female words in the text. The disparity between the occurrences of the two genders means that text is more inclined to mention one over the other, so it can also be considered a form of bias. We report the ratio of occurrence of male and female words in the model generated text, INLINEFORM1 , as INLINEFORM2
- claim1 cites [1, 3]: These include co-occurrence bias and causal occupation bias
  - [1] Reducing Gender Bias in Word-Level Language Models with a Ge / Model Evaluation: Language models are evaluated using perplexity, which is a standard measure of performance for unseen data. For bias evaluation, we use an array of metrics to provide a holistic diagnosis of the model behavior under debiasing treatment. These metrics are discussed in detail below. In all the evaluation metrics requiring gender pairs, we use gender pairs provided by BIBREF7 . This list contains 223 pairs, all other words are considered gender-neutral.

Co-occurrence bias is computed from the model-generated texts by comparing the occurrences of all gender-neutral words with female and male words. A word is considered to be biased towards a certain gender if it occurs more frequently with words of that gender. This definition was first used by BIBREF7 and later adapted by BIBREF5 . Using the definition of gender bias similar to the one used by BIBREF5 , we define gender bias as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral words, and INLINEFORM1 is the occurrences of a word INLINEFORM2 with words of gender INLINEFORM3 in the same window. This score is designed to capture unequal co-occurrences of neutral words with male and female words. Co-occurrences are computed using a sliding window of size 10 extending equally in both directions. Furthermore, we only consider words that occur more than 20 times with gendered words to exclude random effects.

We also evaluate a normalized version of INLINEFORM0 which we denote by conditional co-occurrence bias, INLINEFORM1 . This is defined as INLINEFORM2

where INLINEFORM0

INLINEFORM0 is less affected by the disparity in the general distribution of male and female words in the text. The disparity between the occurrences of the two genders means that text is more inclined to mention one over the other, so it can also be considered a form of bias. We report the ratio of occurrence of male and female words in the model generated text, INLINEFORM1 , as INLINEFORM2
  - [3] Reducing Gender Bias in Word-Level Language Models with a Ge / Model Evaluation: s means that text is more inclined to mention one over the other, so it can also be considered a form of bias. We report the ratio of occurrence of male and female words in the model generated text, INLINEFORM1 , as INLINEFORM2

Another way of quantifying bias in NLP models is based on the idea of causal testing. The model is exposed to paired samples which differ only in one attribute (e.g. gender) and the disparity in the output is interpreted as bias related to that attribute. BIBREF1 and BIBREF0 applied this method to measure bias in coreference resolution and BIBREF0 also used it for evaluating gender bias in language modelling.

Following the approach similar to BIBREF0 , we limit this bias evaluation to a set of gender-neutral occupations. We create a list of sentences based on a set of templates. There are two sets of templates used for evaluating causal occupation bias (Table TABREF7 ). The first set of templates is designed to measure how the probabilities of occupation words depend on the gender information in the seed. Below is an example of the first set of templates: INLINEFORM0

Here, the vertical bar separates the seed sequence that is fed into the language models from the target occupation, for which we observe the output softmax probability. We measure causal occupation bias conditioned on gender as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral occupations and INLINEFORM1 is the size of the gender pairs set. For example, INLINEFORM2 is the softmax probability of the word INLINEFORM3 where the seed sequence is He is a. The second set of templates like below, aims to capture how the probabilities of gendered words depend on the occupation words in the seed. INLINEFORM4

Causal occupation bias conditioned on occupation is represented as INLINEFORM0

where INLINEFORM0 is a set of gender-neutral occupations and INLINEFORM1 is the size of the gender pairs set. For example, INLINEFORM2 is the softmax probability of man where the seed sequence is The doctor is a.

## final-ans11 (answerable, expected: answer)

**Q:** How many users do they look at?  
**Paper:** Predicting the Industry of Users on Social Media

- REF: 22,880 users
- REF: 20,000

**Answer** (answered): The dataset consists of 22,880 users [1].

- claim0 cites [1]: The dataset consists of 22,880 users
  - [1] Predicting the Industry of Users on Social Media / Dataset: We compile our industry-annotated dataset by identifying blogger profiles located in the U.S. on the profile finder on http://www.blogger.com, and scraping only those users that had the industry profile element completed.

For each of these bloggers, we retrieve all their blogs, and for each of these blogs we download the 21 most recent blog postings. We then clean these blog posts of HTML tags and tokenize them, and drop those bloggers whose cumulative textual content in their posts is less than 600 characters. Following these guidelines, we identified all the U.S. bloggers with completed industry information.

Traditionally, standardized industry taxonomies organize economic activities into groups based on similar production processes, products or services, delivery systems or behavior in financial markets. Following such assumptions and regardless of their many similarities, a tomato farmer would be categorized into a distinct industry from a tobacco farmer. As demonstrated in Preotiuc-Pietro et al. Preoctiuc15 such groupings can cause unwarranted misclassifications.

The Blogger platform provides a total of 39 different industry options. Even though a completed industry value is an implicit text annotation, we acknowledge the same problem noted in previous studies: some categories are too broad, while others are very similar. To remedy this and following Guibert et al. Guibert71, who argued that the denominations used in a classification must reflect the purpose of the study, we group the different Blogger industries based on similar educational background and similar technical terminology. To do that, we exclude very general categories and merge conceptually similar ones. Examples of broad categories are the Education and the Student options: a teacher could be teaching in any concentration, while a student could be enrolled in any discipline. Examples of conceptually similar categories are the Investment Banking and the Banking options.

The final set of categories is shown in Table TABREF1 , along with the number of users in each category. The resulting dataset consists of 22,880 users, 41,094 blogs, and 561,003 posts. Table TABREF2 presents additional statistics of our dataset.

## final-ans12 (answerable, expected: answer)

**Q:** What baseline approaches does this approach out-perform?  
**Paper:** Open Event Extraction from Online Text using a Generative Adversarial Network

- REF: K-means, LEM BIBREF13, DPEMM BIBREF14
- REF: K-means, LEM, DPEMM

**Answer** (answered): The baseline approaches that the AEM out-perform are LEM, DPEMM, and K-means [3].

- claim0 cites [3]: The baseline approaches that the AEM out-perform are LEM, DPEMM, and K-means
  - [3] Open Event Extraction from Online Text using a Generative Ad / Experimental Results: To evaluate the performance of the proposed approach, we use the evaluation metrics such as precision, recall and F-measure. Precision is defined as the proportion of the correctly identified events out of the model generated events. Recall is defined as the proportion of correctly identified true events. For calculating the precision of the 4-tuple, we use following criteria:

(1) Do the entity/organization, location, date/person and keyword that we have extracted refer to the same event?

(2) If the extracted representation contains keywords, are they informative enough to tell us what happened?

Table TABREF35 shows the event extraction results on the three datasets. The statistics are obtained with the default parameter setting that INLINEFORM0 is set to 5, number of hidden units INLINEFORM1 is set to 200, and INLINEFORM2 contains three fully-connected layers. The event number INLINEFORM3 for three datasets are set to 25, 25 and 35, respectively. The examples of extracted events are shown in Table. TABREF36 .

It can be observed that K-means performs the worst over all three datasets. On the social media datasets, AEM outpoerforms both LEM and DPEMM by 6.5% and 1.7% respectively in F-measure on the FSD dataset, and 4.4% and 3.7% in F-measure on the Twitter dataset. We can also observe that apart from K-means, all the approaches perform worse on the Twitter dataset compared to FSD, possibly due to the limited size of the Twitter dataset. Moreover, on the Google dataset, the proposed AEM performs significantly better than LEM and DPEMM. It improves upon LEM by 15.5% and upon DPEMM by more than 30% in F-measure. This is because: (1) the assumption made by LEM and DPEMM that all words in a document are generated from a single event is not suitable for long text such as news articles; (2) DPEMM generates too many irrelevant events which leads to a very low precision score. Overall, we see the superior performance of

## final-ans13 (answerable, expected: answer)

**Q:** How do their models decide how much improtance to give to the output words?  
**Paper:** Towards Understanding Neural Machine Translation with Word Importance

- REF: Given the contribution matrix, we can obtain the word importance of each input word to the entire output sentence. 
- REF: They compute the gradient of the output at each time step with respect to the input words to decide the importance.

**Answer** (answered): The models aggregate the contribution values of each input word to all output words using the sum operation and then normalize these sums through the Softmax function to determine the word importance [1].

- claim0 cites [1]: The models aggregate the contribution values of each input word to all output words using the sum operation and then normalize these sums through the Softmax function to determine the word importance
  - [1] Towards Understanding Neural Machine Translation with Word I / Approach > Integrated Gradients: x}$. In detail, the contribution of the $m^{th}$ word in $\textbf {x}$ to the prediction of $F(\textbf {x})_n$ is defined as follows.

where $\frac{\partial {F(\textbf {x})_n}}{\partial {\textbf {x}_m}}$ is the gradient of $F(\textbf {x})_n$ w.r.t. the embedding of the $m^{th}$ word. In this paper, as suggested, the baseline input $\textbf {x}^{\prime }$ is set as a sequence of zero embeddings that has the same sequence length $M$. In this way, we can compute the contribution of a specific input word to a designated output word. Since the above formula is intractable for deep neural models, we approximate it by summing the gradients along a multi-step path from baseline $\textbf {x}^{\prime }$ to the input x.

where $S$ denotes the number of steps that are uniformly distributed along the path. The IG will be more accurate if a larger S is used. In our preliminary experiments, we varied the steps and found 300 steps yielding fairly good performance.

Following the formula, we can calculate the contribution of every input word makes to every output word, forming a contribution matrix of size $M \times N$, where $N$ is the output sentence length. Given the contribution matrix, we can obtain the word importance of each input word to the entire output sentence. To this end, for each input word, we first aggregate its contribution values to all output words by the sum operation, and then normalize all sums through the Softmax function. Figure FIGREF13 illustrates an example of the calculated word importance and the contribution matrix, where an English sentence is translated into a French sentence using the Transformer model. A negative contribution value indicates that the input word has negative effects on the output word.

## final-ans14 (answerable, expected: answer)

**Q:** What baseline model is used?  
**Paper:** Attention-based method for categorizing different types of online harassment language

- REF:  LastStateRNN, AvgRNN, AttentionRNN
- REF: LastStateRNN, AvgRNN, AttentionRNN 

**Answer** (answered): The baseline model used is LastStateRNN [2].

- claim0 cites [2]: The baseline model used is LastStateRNN
  - [2] Attention-based method for categorizing different types of o / Experiments > Evaluation and Results: Each model produces four scores and each score is the probability that a tweet includes harassment language, indirect, physical and sexual harassment language respectively. For any tweet, we first check the score of the harassment language and if it is less than a specified threshold, then the harassment label is zero, so the other three labels are zero as well. If it is greater than or equal to that threshold, then the harassment label is one and the type of harassment is the one among these three having that has the greatest score (highest probability). We set this threshold equal to 0.33.

We compare eight different models in our experiments. Four of them have a Projected Layer (see Fig. FIGREF2), while the others do not have, and this is the only difference between these two groups of our models. So, we actually include four models in our experiments (having a projected layer or not). Firstly, LastStateRNN is the classic RNN model, where the last state passes through an MLP and then the LR Layer estimates the corresponding probability. In contrast, in the AvgRNN model we consider the average vector of all states that come out of the cells. The AttentionRNN model is the one that it has been presented in BIBREF9. Moreover, we introduce the MultiAttentionRNN model for the harassment language detection, which instead of one attention, it includes four attentions, one for each category.

We have evaluated our models considering the F1 Score, which is the harmonic mean of precision and recall. We have run ten times the experiment for each model and considered the average F1 Score. The results are mentioned in Table TABREF11. Considering F1 Macro the models that include the multi-attention mechanism outperform the others and particularly the one with the Projected Layer has the highest performance. In three out of four pairs of models, the ones with the Projected Layer achieved better performance, so in most cases the addition of the Projected Layer had a significant enhancement.

## final-ans15 (answerable, expected: answer)

**Q:** What different correlations result when using different variants of ROUGE scores?  
**Paper:** Revisiting Summarization Evaluation for Scientific Articles

- REF: we observe that many variants of Rouge scores do not have high correlations with human pyramid scores
- REF: Using Pearson corelation measure,  for example, ROUGE-1-P is 0.257 and ROUGE-3-F 0.878.

**Answer** (answered): Different variants of ROUGE scores result in different correlations with human pyramid scores. Specifically, Rouge-1 and Rouge-L have the lowest F-score correlations (0.454), while Rouge-3 has the highest overall correlation [1]. Rouge-W, Rouge-S, and Rouge-SU also show sub-optimal correlations [1]. Additionally, Rouge-2 and Rouge-3 are better correlated with manual judgments in the context of scientific summarization [3].

- claim0 cites []: Different variants of ROUGE scores result in different correlations with human pyramid scores
- claim1 cites [1]: Specifically, Rouge-1 and Rouge-L have the lowest F-score correlations (0.454), while Rouge-3 has the highest overall correlation
  - [1] Revisiting Summarization Evaluation for Scientific Articles / Rouge: Another important observation is regarding the effectiveness of Rouge scores (top part of Table TABREF23 ). Interestingly, we observe that many variants of Rouge scores do not have high correlations with human pyramid scores. The lowest F-score correlations are for Rouge-1 and Rouge-L (with INLINEFORM0 =0.454). Weak correlation of Rouge-1 shows that matching unigrams between the candidate summary and gold summaries is not accurate in quantifying the quality of the summary. On higher order n-grams, however, we can see that Rouge correlates better with pyramid. In fact, the highest overall INLINEFORM1 is obtained by Rouge-3. Rouge-L and its weighted version Rouge-W, both have weak correlations with pyramid. Skip-bigrams (Rouge-S) and its combination with unigrams (Rouge-SU) also show sub-optimal correlations. Note that INLINEFORM2 and INLINEFORM3 correlations are more reliable in our setup due to the small sample size.

These results confirm our initial hypothesis that Rouge is not accurate estimator of the quality of the summary in scientific summarization. We attribute this to the differences of scientific summarization with general domain summaries. When humans summarize a relatively long research paper, they might use different terminology and paraphrasing. Therefore, Rouge which only relies on term matching between a candidate and a gold summary, is not accurate in quantifying the quality of the candidate summary.
- claim2 cites [1]: Rouge-W, Rouge-S, and Rouge-SU also show sub-optimal correlations
  - [1] Revisiting Summarization Evaluation for Scientific Articles / Rouge: Another important observation is regarding the effectiveness of Rouge scores (top part of Table TABREF23 ). Interestingly, we observe that many variants of Rouge scores do not have high correlations with human pyramid scores. The lowest F-score correlations are for Rouge-1 and Rouge-L (with INLINEFORM0 =0.454). Weak correlation of Rouge-1 shows that matching unigrams between the candidate summary and gold summaries is not accurate in quantifying the quality of the summary. On higher order n-grams, however, we can see that Rouge correlates better with pyramid. In fact, the highest overall INLINEFORM1 is obtained by Rouge-3. Rouge-L and its weighted version Rouge-W, both have weak correlations with pyramid. Skip-bigrams (Rouge-S) and its combination with unigrams (Rouge-SU) also show sub-optimal correlations. Note that INLINEFORM2 and INLINEFORM3 correlations are more reliable in our setup due to the small sample size.

These results confirm our initial hypothesis that Rouge is not accurate estimator of the quality of the summary in scientific summarization. We attribute this to the differences of scientific summarization with general domain summaries. When humans summarize a relatively long research paper, they might use different terminology and paraphrasing. Therefore, Rouge which only relies on term matching between a candidate and a gold summary, is not accurate in quantifying the quality of the candidate summary.
- claim3 cites [3]: Additionally, Rouge-2 and Rouge-3 are better correlated with manual judgments in the context of scientific summarization
  - [3] Revisiting Summarization Evaluation for Scientific Articles / Conclusions: We provided an analysis of existing evaluation metrics for scientific summarization with evaluation of all variants of Rouge. We showed that Rouge may not be the best metric for summarization evaluation; especially in summaries with high terminology variations and paraphrasing (e.g. scientific summaries). Furthermore, we showed that different variants of Rouge result in different correlation values with human judgments, indicating that not all Rouge scores are equally effective. Among all variants of Rouge, Rouge-2 and Rouge-3 are better correlated with manual judgments in the context of scientific summarization. We furthermore proposed an alternative and more effective approach for scientific summarization evaluation (Summarization Evaluation by Relevance Analysis - Sera). Results revealed that in general, the proposed evaluation metric achieves higher correlations with semi-manual pyramid evaluation scores in comparison with Rouge.

Our analysis on the effectiveness of evaluation measures for scientific summaries was performed using correlations with manual judgments. An alternative approach to follow would be to use statistical significance testing on the ability of the metrics to distinguish between the summarizers (similar to Rankel:2011). We studied the effectiveness of existing summarization evaluation metrics in the scientific text genre and proposed an alternative superior metric. Another extension of this work would be to evaluate automatic summarization evaluation in other genres of text (such as social media). Our proposed method only evaluates the content quality of the summary. Similar to most of existing summarization evaluation metrics, other qualities such as linguistic cohesion, coherence and readability are not captured by this method. Developing metrics that also incorporate these qualities is yet another future direction to follow.

## final-ans16 (answerable, expected: answer)

**Q:** Do they report results only on English data?  
**Paper:** Automated News Suggestions for Populating Wikipedia Entity Pages

- REF: Yes
- REF: Yes

**Answer** (answered): No [4].

- claim0 cites [4]: No
  - [4] Automated News Suggestions for Populating Wikipedia Entity P / Introduction: Wikipedia is the largest source of open and collaboratively curated knowledge in the world. Introduced in 2001, it has evolved into a reference work with around 5m pages for the English Wikipedia alone. In addition, entities and event pages are updated quickly via collaborative editing and all edits are encouraged to include source citations, creating a knowledge base which aims at being both timely as well as authoritative. As a result, it has become the preferred source of information consumption about entities and events. Moreso, this knowledge is harvested and utilized in building knowledge bases like YAGO BIBREF0 and DBpedia BIBREF1 , and used in applications like text categorization BIBREF2 , entity disambiguation BIBREF3 , entity ranking BIBREF4 and distant supervision BIBREF5 , BIBREF6 .

However, not all Wikipedia pages referring to entities (entity pages) are comprehensive: relevant information can either be missing or added with a delay. Consider the city of New Orleans and the state of Odisha which were severely affected by cyclones Hurricane Katrina and Odisha Cyclone, respectively. While Katrina finds extensive mention in the entity page for New Orleans, Odisha Cyclone which has 5 times more human casualties (cf. Figure FIGREF2 ) is not mentioned in the page for Odisha. Arguably Katrina and New Orleans are more popular entities, but Odisha Cyclone was also reported extensively in national and international news outlets. This highlights the lack of important facts in trunk and long-tail entity pages, even in the presence of relevant sources. In addition, previous studies have shown that there is an inherent delay or lag when facts are added to entity pages BIBREF7 .

## final-ans17 (answerable, expected: answer)

**Q:** What classifiers have been trained?  
**Paper:** Understanding Car-Speak: Replacing Humans in Dealerships

- REF: KNN
RF
SVM
MLP
- REF:  K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), Multi-layer Perceptron (MLP)

**Answer** (answered): K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Multi-layer Perceptron (MLP) [2].

- claim0 cites [2]: K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Multi-layer Perceptron (MLP)
  - [2] Understanding Car-Speak: Replacing Humans in Dealerships / Translating Car-Speak > Classification Experiments: We train a series of classifiers in order to classify car-speak. We train three classifiers on the review vectors that we prepared in Section SECREF8. The classifiers we use are K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Multi-layer Perceptron (MLP) BIBREF13.

In order to evaluate our classifiers, we perform 4-fold cross validation on a shuffled data set. Table TABREF10 shows the F1 micro and F1 macro scores for all the classifiers. The KNN classifier seem to perform the best across all four metrics. This is probably due to the multi-class nature of the data set.

## final-ans18 (answerable, expected: answer)

**Q:** What language(s) is/are represented in the dataset?  
**Paper:** Attention-based method for categorizing different types of online harassment language

- REF: english
- REF: english

**Answer** (answered): The dataset includes English tweets. [1]

- claim0 cites [1]: The dataset includes English tweets
  - [1] Attention-based method for categorizing different types of o / Dataset description: The dataset from Twitter that we are using in our work, consists of a train set, a validation set and a test set. It was published for the "First workshop on categorizing different types of online harassment languages in social media". The whole dataset is divided into two categories, which are harassment and non-harassment tweets. Moreover, considering the type of the harassment, the tweets are divided into three sub-categories which are indirect harassment, sexual and physical harassment. We can see in Table TABREF1 the class distribution of our dataset. One important issue here is that the categories of indirect and physical harassment seem to be more rare in the train set than in the validation and test sets. To tackle this issue, as we describe in the next section, we are performing data augmentation techniques. However, the dataset is imbalanced and this has a significant impact in our results.

## final-ans19 (answerable, expected: answer)

**Q:** What are the distinctive characteristics of how Arabic speakers use offensive language?  
**Paper:** Arabic Offensive Language on Twitter: Analysis and Experiments

- REF: Frequent use of direct animal name calling, using simile and metaphors, through indirect speech like sarcasm, wishing evil to others, name alteration, societal stratification, immoral behavior and sexually related uses.
- REF: Direct name calling, Simile and metaphor, Indirect speech, Wishing Evil, Name alteration, Societal stratification, Immoral behavior, Sexually related

**Answer** (answered): The distinctive characteristics of how Arabic speakers use offensive language include direct name calling, where animal names like كلب> (“klb” – “dog”), حمار> (“HmAr” – “donkey”), and بهيم> (“bhym” – “beast”) are commonly used as insults [3]. Additionally, religious hate speech appeared in only 15% of all hate speech tweets [3].

- claim0 cites [3]: The distinctive characteristics of how Arabic speakers use offensive language include direct name calling, where animal names like كلب> (“klb” – “dog”), حمار> (“HmAr” – “donkey”), and بهيم> (“bhym” – “beast”) are commonly used as insults
  - [3] Arabic Offensive Language on Twitter: Analysis and Experimen / Data Collection > Statistics and User Demographics: origin, and sport affiliation. Religious hate speech appeared in only 15% of all hate speech tweets.

Next, we analyzed all tweets labeled as offensive to better understand how Arabic speakers use offensive language. Here is a breakdown of usage:

Direct name calling: The most frequent attack is to call a person an animal name, and the most used animals were كلب> (“klb” – “dog”), حمار> (“HmAr” – “donkey”), and بهيم> (“bhym” – “beast”). The second most common was insulting mental abilities using words such as غبي> (“gby” – “stupid”) and عبيط> (“EbyT” –“idiot”). Some culture-specific differences should be considered. Not all animal names are used as insults. For example, animals such as أسد> (“Asd” – “lion”), صقر> (“Sqr” – “falcon”), and غزال> (“gzAl” – “gazelle”) are typically used for praise. For other insults, people use: some bird names such as دجاجة> (“djAjp” – “chicken”), بومة> (“bwmp” – “owl”), and غراب> (“grAb” – “crow”); insects such as ذبابة> (“*bAbp” – “fly”), صرصور> (“SrSwr” – “cockroach”), and حشرة> (“H$rp” – “insect”); microorganisms such as جرثومة> (“jrvwmp” – “microbe”) and طحالب> (“THAlb” – “algae”); inanimate objects such as جزمة> (“jzmp” – “shoes”) and سطل> (“sTl” – “bucket”) among other
- claim1 cites [3]: Additionally, religious hate speech appeared in only 15% of all hate speech tweets
  - [3] Arabic Offensive Language on Twitter: Analysis and Experimen / Data Collection > Statistics and User Demographics: origin, and sport affiliation. Religious hate speech appeared in only 15% of all hate speech tweets.

Next, we analyzed all tweets labeled as offensive to better understand how Arabic speakers use offensive language. Here is a breakdown of usage:

Direct name calling: The most frequent attack is to call a person an animal name, and the most used animals were كلب> (“klb” – “dog”), حمار> (“HmAr” – “donkey”), and بهيم> (“bhym” – “beast”). The second most common was insulting mental abilities using words such as غبي> (“gby” – “stupid”) and عبيط> (“EbyT” –“idiot”). Some culture-specific differences should be considered. Not all animal names are used as insults. For example, animals such as أسد> (“Asd” – “lion”), صقر> (“Sqr” – “falcon”), and غزال> (“gzAl” – “gazelle”) are typically used for praise. For other insults, people use: some bird names such as دجاجة> (“djAjp” – “chicken”), بومة> (“bwmp” – “owl”), and غراب> (“grAb” – “crow”); insects such as ذبابة> (“*bAbp” – “fly”), صرصور> (“SrSwr” – “cockroach”), and حشرة> (“H$rp” – “insect”); microorganisms such as جرثومة> (“jrvwmp” – “microbe”) and طحالب> (“THAlb” – “algae”); inanimate objects such as جزمة> (“jzmp” – “shoes”) and سطل> (“sTl” – “bucket”) among other

## final-ans20 (answerable, expected: answer)

**Q:** What background knowledge do they leverage?  
**Paper:** Robustly Leveraging Prior Knowledge in Text Classification

- REF: labeled features
- REF: labelled features, which are words whose presence strongly indicates a specific class or topic

**Answer** (answered): Words such as NBA, player, and basketball are strong indicators of the sports category [1], and words like terrible, boring, and messing indicate a negative polarity while words like perfect, exciting, and moving suggest a positive polarity in sentiment classification [1]. Additionally, labeled features such as words like amazing, exciting can be labeled features for class positive in sentiment classification [4].

- claim0 cites [1]: Words such as NBA, player, and basketball are strong indicators of the sports category , and words like terrible, boring, and messing indicate a negative polarity while words like perfect, exciting, and moving suggest a positive polarity in sentiment classification
  - [1] Robustly Leveraging Prior Knowledge in Text Classification / Introduction: We posses a wealth of prior knowledge about many natural language processing tasks. For example, in text categorization, we know that words such as NBA, player, and basketball are strong indicators of the sports category BIBREF0 , and words like terrible, boring, and messing indicate a negative polarity while words like perfect, exciting, and moving suggest a positive polarity in sentiment classification.

A key problem arisen here, is how to leverage such knowledge to guide the learning process, an interesting problem for both NLP and machine learning communities. Previous studies addressing the problem fall into several lines. First, to leverage prior knowledge to label data BIBREF1 , BIBREF2 . Second, to encode prior knowledge with a prior on parameters, which can be commonly seen in many Bayesian approaches BIBREF3 , BIBREF4 . Third, to formalise prior knowledge with additional variables and dependencies BIBREF5 . Last, to use prior knowledge to control the distributions over latent output variables BIBREF6 , BIBREF7 , BIBREF8 , which makes the output variables easily interpretable.

However, a crucial problem, which has rarely been addressed, is the bias in the prior knowledge that we supply to the learning model. Would the model be robust or sensitive to the prior knowledge? Or, which kind of knowledge is appropriate for the task? Let's see an example: we may be a baseball fan but unfamiliar with hockey so that we can provide a few number of feature words of baseball, but much less of hockey for a baseball-hockey classification task. Such prior knowledge may mislead the model with heavy bias to baseball. If the model cannot handle this situation appropriately, the performance may be undesirable.
- claim1 cites [4]: Additionally, labeled features such as words like amazing, exciting can be labeled features for class positive in sentiment classification
  - [4] Robustly Leveraging Prior Knowledge in Text Classification / Method: We address the robustness problem on top of GE-FL BIBREF0 , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification.

## final-ans21 (answerable, expected: answer)

**Q:** What do they mean by a person's industry?  
**Paper:** Predicting the Industry of Users on Social Media

- REF: the aggregate of enterprises in a particular field
- REF: the aggregate of enterprises in a particular field

**Answer** (answered): A person's industry refers to the aggregate of enterprises in a particular field, which can be predicted by identifying industry indicative text in social media. This concept is used to classify users based on their professional or business context [3].

- claim0 cites []: A person's industry refers to the aggregate of enterprises in a particular field, which can be predicted by identifying industry indicative text in social media
- claim1 cites [3]: This concept is used to classify users based on their professional or business context
  - [3] Predicting the Industry of Users on Social Media / Introduction: Over the past two decades, the emergence of social media has enabled the proliferation of traceable human behavior. The content posted by users can reflect who their friends are, what topics they are interested in, or which company they are working for. At the same time, users are listing a number of profile fields to define themselves to others. The utilization of such metadata has proven important in facilitating further developments of applications in advertising BIBREF0 , personalization BIBREF1 , and recommender systems BIBREF2 . However, profile information can be limited, depending on the platform, or it is often deliberately omitted BIBREF3 . To uncloak this information, a number of studies have utilized social media users' footprints to approximate their profiles.

This paper explores the potential of predicting a user's industry –the aggregate of enterprises in a particular field– by identifying industry indicative text in social media. The accurate prediction of users' industry can have a big impact on targeted advertising by minimizing wasted advertising BIBREF4 and improved personalized user experience. A number of studies in the social sciences have associated language use with social factors such as occupation, social class, education, and income BIBREF5 , BIBREF6 , BIBREF7 , BIBREF8 . An additional goal of this paper is to examine such findings, and in particular the link between language and occupational class, through a data-driven approach.

In addition, we explore how meaning changes depending on the occupational context. By leveraging word embeddings, we seek to quantify how, for example, cloud might mean a separate concept (e.g., condensed water vapor) in the text written by users that work in environmental jobs while it might be used differently by users in technology occupations (e.g., Internet-based computing).

Specifically, this paper makes four main contributions. First, we build a large, industry-annotated dataset that contains over 20,000 blog users. In addition to their posted text, we also link a number of user metadata including their gender, location, occupation, introduction and interests.

## final-ans22 (answerable, expected: answer)

**Q:** How they utilize LDA and Gibbs sampling to evaluate ISWC and WWW publications?  
**Paper:** Natural Language Processing via LDA Topic Model in Recommendation Systems

- REF: the LDA approaches to recommendation systems and given the importance of research, we have studied recent impressive articles on this subject and presented a taxonomy of recommendation systems based on LDA of the recent research, we evaluated ISWC and WWW conferences articles from DBLP website and used the Gibbs sampling algorithm as an evaluation parameter
- REF: discover the trends of the topics and find relationship between LDA topics and paper features and generate trust tags,  learn a LDA model with 100 topics; $\alpha =0.01$, $\beta = 0.01$ and using Gibbs sampling as a parameter estimation

**Answer** (answered): They utilized LDA and Gibbs sampling to evaluate ISWC and WWW publications by applying the LDA algorithm and Gibbs sampling on the conferences' publications from 2013-2017 [3]. They extracted abstracts and titles from each article and used MALLET to implement the inference and obtain the topic models [5].

- claim0 cites [3]: They utilized LDA and Gibbs sampling to evaluate ISWC and WWW publications by applying the LDA algorithm and Gibbs sampling on the conferences' publications from 2013-2017
  - [3] Natural Language Processing via LDA Topic Model in Recommend / Conclusion: In this paper, we presented a taxonomy of recommendation systems and applications based on LDA of the recent research, including app, travel, friend, location, scientific paper, and music recommendation. Furthermore, we applied LDA algorithm and Gibbs sampling on ISWC and WWW conference's publications from 2013-2017. Generally, recommendation systems can be an impressive interface between online users and websites in the Internet communities. Our study suggest that NLP methods based on LDA can discover hidden aspects to better understanding of the behaviors of the people to build smart recommendation systems in online communities.
- claim1 cites [5]: They extracted abstracts and titles from each article and used MALLET to implement the inference and obtain the topic models
  - [5] Natural Language Processing via LDA Topic Model in Recommend / Experiment: Semantic Discovery and researchers behavior analysis > Dataset: We extracted ISWC and WWW conferences publications from DBLP website by only considering conferences for which data was available for years 2013-2017. In total, It should be noted that in these experiments, we considered abstracts and titles from each article. In this paper, we used MALLET (http://mallet.cs.umass.edu/) to implement the inference and obtain the topic models. In addition, our full dataset is available at https://github.com/JeloH/Dataset_DBLP. The most important goal of this experiment is discover the trends of the topics and find relationship between LDA topics and paper features and generate trust tags.

## final-ans23 (answerable, expected: answer)

**Q:** Does this method help in sentiment classification task improvement?  
**Paper:** Shallow Syntax in Deep Water


**Answer** (answered): No [1]. The results show that shallow syntactic features are slightly harmful on average for sentiment classification, and mSynC performs similarly to ELMo-transformer [1].

- claim0 cites [1]: No
  - [1] Shallow Syntax in Deep Water / Experiments > Downstream Task Transfer > Sentiment analysis: We consider fine-grained (5-class) classification on Stanford Sentiment Treebank BIBREF24. The labels are negative, somewhat_negative, neutral, positive and somewhat_positive. Our model was based on the biattentive classification network BIBREF25. We used all phrase lengths in the dataset for training, but test results are reported only on full sentences, following prior work.

Results are shown in Table TABREF12. Consistent with previous findings, cwrs offer large improvements across all tasks. Though helpful to span-level task models without cwrs, shallow syntactic features offer little to no benefit to ELMo models. mSynC's performance is similar. This holds even for phrase-structure parsing, where (gold) chunks align with syntactic phrases, indicating that task-relevant signal learned from exposure to shallow syntax is already learned by ELMo. On sentiment classification, chunk features are slightly harmful on average (but variance is high); mSynC again performs similarly to ELMo-transformer. Overall, the performance differences across all tasks are small enough to infer that shallow syntax is not particularly helpful when using cwrs.
- claim1 cites [1]: The results show that shallow syntactic features are slightly harmful on average for sentiment classification, and mSynC performs similarly to ELMo-transformer
  - [1] Shallow Syntax in Deep Water / Experiments > Downstream Task Transfer > Sentiment analysis: We consider fine-grained (5-class) classification on Stanford Sentiment Treebank BIBREF24. The labels are negative, somewhat_negative, neutral, positive and somewhat_positive. Our model was based on the biattentive classification network BIBREF25. We used all phrase lengths in the dataset for training, but test results are reported only on full sentences, following prior work.

Results are shown in Table TABREF12. Consistent with previous findings, cwrs offer large improvements across all tasks. Though helpful to span-level task models without cwrs, shallow syntactic features offer little to no benefit to ELMo models. mSynC's performance is similar. This holds even for phrase-structure parsing, where (gold) chunks align with syntactic phrases, indicating that task-relevant signal learned from exposure to shallow syntax is already learned by ELMo. On sentiment classification, chunk features are slightly harmful on average (but variance is high); mSynC again performs similarly to ELMo-transformer. Overall, the performance differences across all tasks are small enough to infer that shallow syntax is not particularly helpful when using cwrs.

## final-una00 (unanswerable, expected: refuse)

**Q:** Is it possible to convert a cloze-style questions to a naturally-looking questions?  
**Paper:** Simple and Effective Semi-Supervised Question Answering


**Answer** (answered): Yes, it is possible to convert cloze-style questions to naturally-looking questions. The paper mentions that the cloze-style questions are constructed from the document structure and can be transformed into naturally-looking questions through the system's methodology [2][5].

- claim0 cites []: Yes, it is possible to convert cloze-style questions to naturally-looking questions
- claim1 cites [2, 5]: The paper mentions that the cloze-style questions are constructed from the document structure and can be transformed into naturally-looking questions through the system's methodology
  - [2] Simple and Effective Semi-Supervised Question Answering / Abstract: Recent success of deep learning models for the task of extractive Question Answering (QA) is hinged on the availability of large annotated corpora. However, large domain specific annotated corpora are limited and expensive to construct. In this work, we envision a system where the end user specifies a set of base documents and only a few labelled examples. Our system exploits the document structure to create cloze-style questions from these base documents; pre-trains a powerful neural network on the cloze style questions; and further fine-tunes the model on the labeled examples. We evaluate our proposed system across three diverse datasets from different domains, and find it to be highly effective with very little labeled data. We attain more than 50% F1 score on SQuAD and TriviaQA with less than a thousand labelled examples. We are also releasing a set of 3.2M cloze-style questions for practitioners to use while building QA systems.
  - [5] Simple and Effective Semi-Supervised Question Answering / Introduction: Deep learning systems have shown a lot of promise for extractive Question Answering (QA), with performance comparable to humans when large scale data is available. However, practitioners looking to build QA systems for specific applications may not have the resources to collect tens of thousands of questions on corpora of their choice. At the same time, state-of-the-art machine reading systems do not lend well to low-resource QA settings where the number of labeled question-answer pairs are limited (c.f. Table 2 ). Semi-supervised QA methods like BIBREF0 aim to improve this performance by leveraging unlabeled data which is easier to collect.

In this work, we present a semi-supervised QA system which requires the end user to specify a set of base documents and only a small set of question-answer pairs over a subset of these documents. Our proposed system consists of three stages. First, we construct cloze-style questions (predicting missing spans of text) from the unlabeled corpus; next, we use the generated clozes to pre-train a powerful neural network model for extractive QA BIBREF1 , BIBREF2 ; and finally, we fine-tune the model on the small set of provided QA pairs.

Our cloze construction process builds on a typical writing phenomenon and document structure: an introduction precedes and summarizes the main body of the article. Many large corpora follow such a structure, including Wikipedia, academic papers, and news articles. We hypothesize that we can benefit from the un-annotated corpora to better answer various questions – at least ones that are lexically similar to the content in base documents and directly require factual information.

## final-una01 (unanswerable, expected: refuse)

**Q:** How many GPUs are used for the training of SBERT?  
**Paper:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks


**Answer** (model_declined): I can't find this in the papers.


## final-una02 (unanswerable, expected: refuse)

**Q:** What is the average length of the title text?  
**Paper:** Shallow reading with Deep Learning: Predicting popularity of online content using only its title


**Answer** (model_declined): I can't find this in the papers.


## final-una03 (unanswerable, expected: refuse)

**Q:** How many layers does their model have?  
**Paper:** FacTweet: Profiling Fake News Twitter Accounts


**Answer** (answered): The model has one LSTM layer. [3]

- claim0 cites [3]: The model has one LSTM layer
  - [3] FacTweet: Profiling Fake News Twitter Accounts / Methodology: use canonical stylistic features, such as the count of question marks, exclamation marks, consecutive characters and letters, links, hashtags, users' mentions. In addition, we extract the uppercase ratio and the tweet length.

Words embeddings: We extract words embeddings of the words of the tweet using $Glove\-840B-300d$ BIBREF17 pretrained model. The tweet final representation is obtained by averaging its words embeddings.

Model. To account for chunk sequences we make use of a de facto standard approach and opt for a recurrent neural model using long short-term memory (LSTM) BIBREF18. In our model, the sequence consists of a sequence of tweets belonging to one chunk (Figure FIGREF11). The LSTM learns the hidden state $h\textsubscript {t}$ by capturing the previous timesteps (past tweets). The produced hidden state $h\textsubscript {t}$ at each time step is passed to the attention layer which computes a `context' vector $c\textsubscript {t}$ as the weighted mean of the state sequence $h$ by:

Where $T$ is the total number of timesteps in the input sequence and $\alpha \textsubscript {tj}$ is a weight computed at each time step $j$ for each state hj.

## final-una04 (unanswerable, expected: refuse)

**Q:** What is the performance of their method?  
**Paper:** Text Summarization using Abstract Meaning Representation


**Answer** (answered): Their method out-performs BIBREF0's method with best ROGUE-1 INLINEFORM0 scores using the first co-occurrence+first model for important sentences [2].

- claim0 cites [2]: Their method out-performs BIBREF0's method with best ROGUE-1 INLINEFORM0 scores using the first co-occurrence+first model for important sentences
  - [2] Text Summarization using Abstract Meaning Representation / Results on the Proxy report section: In table TABREF13 we report the results of using the pipeline with generation using the alignment based generation module defined in section SECREF12 , on the proxy report section of the AMR Bank. All of our methods out-perform BIBREF0 's method. We obtain best ROGUE-1 INLINEFORM0 scores using the first co-occurrence+first model for important sentences. This also out-perform our Lead-1-AMR baseline by 0.3 ROGUE-1 INLINEFORM1 points.

## final-una05 (unanswerable, expected: refuse)

**Q:** Do they report results only on English data?  
**Paper:** Open Event Extraction from Online Text using a Generative Adversarial Network


**Answer** (low_relevance): I can't find this in the papers.


## final-una06 (unanswerable, expected: refuse)

**Q:** What are the languages represented in the DSL datasets?  
**Paper:** Short Text Language Identification for Under Resourced Languages


**Answer** (answered): The DSL datasets represent 14 languages over 6 language groups [1].

- claim0 cites [1]: The DSL datasets represent 14 languages over 6 language groups
  - [1] Short Text Language Identification for Under Resourced Langu / Related Works: The focus of this section is on recently published datasets and LID research applicable to the South African context. An in depth survey of algorithms, features, datasets, shared tasks and evaluation methods may be found in BIBREF0.

The datasets for the DSL 2015 & DSL 2017 shared tasks BIBREF1 are often used in LID benchmarks and also available on Kaggle . The DSL datasets, like other LID datasets, consists of text sentences labelled by language. The 2017 dataset, for example, contains 14 languages over 6 language groups with 18000 training samples and 1000 testing samples per language.

The recently published JW300 parallel corpus BIBREF2 covers over 300 languages with around 100 thousand parallel sentences per language pair on average. In South Africa, a multilingual corpus of academic texts produced by university students with different mother tongues is being developed BIBREF3. The WiLI-2018 benchmark dataset BIBREF4 for monolingual written natural language identification includes around 1000 paragraphs of 235 languages. A possibly useful link can also be made BIBREF5 between Native Language Identification (NLI) (determining the native language of the author of a text) and Language Variety Identification (LVI) (classification of different varieties of a single language) which opens up more datasets. The Leipzig Corpora Collection BIBREF6, the Universal Declaration of Human Rights and Tatoeba are also often used sources of data.

The NCHLT text corpora BIBREF7 is likely a good starting point for a shared LID task dataset for the South African languages BIBREF8. The NCHLT text corpora contains enough data to have 3500 training samples and 600 testing samples of 300+ character sentences per language. Researchers have recently started applying existing algorithms for tasks like neural machine translation in earnest to such South African language datasets BIBREF9.

## final-una07 (unanswerable, expected: refuse)

**Q:** Do the errors of the model reflect linguistic similarity between different L1s?  
**Paper:** A Portuguese Native Language Identification Dataset


**Answer** (answered): No [4]. The errors of the model are attributed to topic bias rather than linguistic similarity between different L1s. The model could be learning topic associations instead of language-specific features [4].

- claim0 cites [4]: No
  - [4] A Portuguese Native Language Identification Dataset / A Baseline for Portuguese NLI: To demonstrate the usefulness of the dataset we present the first lexical baseline for Portuguese NLI using a sub-set of NLI-PT. To the best of our knowledge, no study has been published on Portuguese NLI and our work fills this gap.

In this experiment we included the five L1s in NLI-PT which contain the largest number of texts in this sub-set and run a simple linear SVM BIBREF21 classifier using a bag of words model to identify the L1 of each text. The languages included in this experiment were Chinese (355 texts), English (236 texts), German (214 texts), Italian (216 texts), and Spanish (271 texts).

We evaluated the model using stratified 10-fold cross-validation, achieving 70% accuracy. An important limitation of this experiment is that it does not account for topic bias, an important issue in NLI BIBREF22 . This is due to the fact that NLI-PT is not balanced by topic and the model could be learning topic associations instead. In future work we would like to carry out using syntactic features such as function words, syntactic relations and POS annotation.
- claim1 cites []: The errors of the model are attributed to topic bias rather than linguistic similarity between different L1s
- claim2 cites [4]: The model could be learning topic associations instead of language-specific features
  - [4] A Portuguese Native Language Identification Dataset / A Baseline for Portuguese NLI: To demonstrate the usefulness of the dataset we present the first lexical baseline for Portuguese NLI using a sub-set of NLI-PT. To the best of our knowledge, no study has been published on Portuguese NLI and our work fills this gap.

In this experiment we included the five L1s in NLI-PT which contain the largest number of texts in this sub-set and run a simple linear SVM BIBREF21 classifier using a bag of words model to identify the L1 of each text. The languages included in this experiment were Chinese (355 texts), English (236 texts), German (214 texts), Italian (216 texts), and Spanish (271 texts).

We evaluated the model using stratified 10-fold cross-validation, achieving 70% accuracy. An important limitation of this experiment is that it does not account for topic bias, an important issue in NLI BIBREF22 . This is due to the fact that NLI-PT is not balanced by topic and the model could be learning topic associations instead. In future work we would like to carry out using syntactic features such as function words, syntactic relations and POS annotation.

## final-una08 (unanswerable, expected: refuse)

**Q:** How large are the textual descriptions of entities?  
**Paper:** Knowledge Graph Representation with Jointly Structural and Textual Encoding


**Answer** (answered): The size of textual descriptions is not explicitly mentioned in the papers. However, it is noted that text descriptions may present an entity from various aspects, and different relations focus on fractional aspects of the description [3]. Additionally, the attention mechanism is introduced to select the most related information from text descriptions under different contexts [4].

- claim0 cites []: The size of textual descriptions is not explicitly mentioned in the papers
- claim1 cites [3]: However, it is noted that text descriptions may present an entity from various aspects, and different relations focus on fractional aspects of the description
  - [3] Knowledge Graph Representation with Jointly Structural and T / Introduction: -dimensional vector space BIBREF2 , BIBREF3 . The representations of entities and relations are obtained by minimizing a global loss function involving all entities and relations. Therefore, we can do reasoning over knowledge graphs through algebraic computations.

Although existing methods have good capability to learn knowledge graph embeddings, it remains challenging for entities with few or no facts BIBREF4 . To solve the issue of KB sparsity, many methods have been proposed to learn knowledge graph embeddings by utilizing related text information BIBREF5 , BIBREF6 , BIBREF7 . These methods learn joint embedding of entities, relations, and words (or phrases, sentences) into the same vector space. However, there are still three problems to be solved. (1) The combination methods of the structural and textual representations are not well studied in these methods, in which two kinds of representations are merely aligned on word level or separate loss function. (2) The text description may represent an entity from various aspects, and various relations only focus on fractional aspects of the description. A good encoder should select the information from text in accordance with certain contexts of relations. Figure 1 illustrates the fact that not all information provided in its description are useful to predict the linked entities given a specific relation. (3) Intuitively, entities with many facts depend more on well-trained structured representation while those with few or no facts might be largely determined by text descriptions. A good representation should learn the most valuable information by balancing both sides.
- claim2 cites [4]: Additionally, the attention mechanism is introduced to select the most related information from text descriptions under different contexts
  - [4] Knowledge Graph Representation with Jointly Structural and T / Attentive LSTM Encoder: While the LSTM encoder has richer capacity than NBOW, it produces the same representation for the entire text description regardless of its contexts. However, the text description may present an entity from various aspects, and various relations only focus on fractional aspects of the description. This phenomenon also occurs in structure embedding for an entity BIBREF8 , BIBREF19 .

Given a relation for an entity, not all of words/phrases in its text description are useful to model a specific fact. Some of them may be important for the given relation, but may be useless for other relations. Therefore, we introduce an attention mechanism BIBREF20 to utilize an attention-based encoder that constructs contextual text encodings according to different relations.

For each position $i$ of the text description, the attention for a given relation $r$ is defined as $\alpha _i(r)$ , which is

$$e_i(r) &= \mathbf {v}_a^T \tanh (\mathbf {W}_a {\mathbf {z}}_i + \mathbf {U}_a \mathbf {r}), \\ \alpha _i(r)&=\operatorname{\mathbf {softmax}}(e_i(r))\nonumber \\ &=\frac{\exp (e_i(r))}{\sum ^{n}_{j=1} \exp (e_j(r))},$$ (Eq. 12)

## final-una09 (unanswerable, expected: refuse)

**Q:** How does this approach compares to the state-of-the-art results on these tasks?  
**Paper:** Contextual Joint Factor Acoustic Embeddings


**Answer** (answered): The proposed approaches outperform the state-of-the-art results on these tasks. For phone classification, one of the proposed approaches achieves a classification accuracy of 74.1%, which is higher than the baseline VAE performance of 72.2% [1]. For speaker recognition, the CJFA encoder shows significant improvements, with an absolute accuracy improvement of 7.9% for Task b [2]. The CJFA encoder performs better than the VAE baseline and CJFS encoder on all tasks [3].

- claim0 cites []: The proposed approaches outperform the state-of-the-art results on these tasks
- claim1 cites [1]: For phone classification, one of the proposed approaches achieves a classification accuracy of 74.1%, which is higher than the baseline VAE performance of 72.2%
  - [1] Contextual Joint Factor Acoustic Embeddings / Abstract: Embedding acoustic information into fixed length representations is of interest for a whole range of applications in speech and audio technology. We propose two novel unsupervised approaches to generate acoustic embeddings by modelling of acoustic context. The first approach is a contextual joint factor synthesis encoder, where the encoder in an encoder/decoder framework is trained to extract joint factors from surrounding audio frames to best generate the target output. The second approach is a contextual joint factor analysis encoder, where the encoder is trained to analyse joint factors from the source signal that correlates best with the neighbouring audio. To evaluate the effectiveness of our approaches compared to prior work, we chose two tasks - phone classification and speaker recognition - and test on different TIMIT data sets. Experimental results show that one of our proposed approaches outperforms phone classification baselines, yielding a classification accuracy of 74.1%. When using additional out-of-domain data for training, an additional 2-3% improvements can be obtained, for both for phone classification and speaker recognition tasks.
- claim2 cites [2]: For speaker recognition, the CJFA encoder shows significant improvements, with an absolute accuracy improvement of 7.9% for Task b
  - [2] Contextual Joint Factor Acoustic Embeddings / Results and Discussion: It is not surprising to see CJFS perform poorly on phone classification as the target frame in not present in the input, therefore the embedding just does not have the phone segment information. However, as per speaker recognition results, speaker information is retained.

A variation of the window sizes to larger windows seems detrimental in almost all cases, aside from the more difficult Task b. This may be in part the effect of the amount of training data available, however it confirms that contextual models outperform the baseline VAE model configuration, generally, and in particular also with the same amount of input data for speaker recognition. It is also noticeable that the decline or variation as a function of window size is less pronounced for the CJFA case, implying increased stability. For phone classification the trade-off benefit for window size is less clear.

For phone classification, increasing the embedding $K$ is helpful, but performance remains stable at $K=150$. Hence in all of the rest of our experiments, the embedding dimension is set to 150 for all of the rest configurations. For speaker recognition the observed variations are small.

A further set of experiments investigated the use of out of domain data for improving classification in a completely unsupervised setting. The RM corpus BIBREF29 was used in this case to augment the TIMIT data for training the embeddings only. All other configurations an training settings are unchanged. Table TABREF18 shows improvement after using additional out-of-domain data for training, except for in the case of CJFS and for phone classification. The improvement on all tasks with the simple addition of unlabelled audio data is remarkable. This is also true for the baseline, but the benefit of the proposed methods seems unaffected. The CJFA encoder performs better in comparison of the other two approaches and a absolute accuracy improvement of 7.9% for speaker recognition Task b is observed. The classification tasks benefits from the additional data even though the labelled data remains the same.
- claim3 cites [3]: The CJFA encoder performs better than the VAE baseline and CJFS encoder on all tasks
  - [3] Contextual Joint Factor Acoustic Embeddings / Conclusion and Future Work: In this paper, two unsupervised acoustic embedding approaches to model the joint latent factors between the target window and neighbouring audio segments were proposed. Models are based on variational auto-encoders, which also constitute the baseline. In order to compare against the baseline models are assessed using phone classification and speaker recognition tasks, on TIMIT, and with additional RM data. Results show CJFA (contextual joint factor analysis) encoder performs significantly better in both phone classification and speaker recognition tasks compared with other two approaches. The CJFS (contextual joint factor synthesis) encoder performs close to CJFA in speaker recognition task, but poorer for phone classification. Overall a gain of up to 3% relative on phone classification accuracy is observed, relative improvements on speaker recognition show 3–6% gain. The proposed unsupervised approaches obtain embeddings and can be improved with unlabelled out-of-domain data, the classification tasks benefits even though the labelled data remains the same. Further work needs to expand experiments on larger data sets, phone recognition and more complex neural network architectures.

## final-off00 (off_topic, expected: refuse)

**Q:** What is the population of Canada?  
**Paper:** (none picked)


**Answer** (low_relevance): I can't find this in the papers.


## final-off01 (off_topic, expected: refuse)

**Q:** How do I change a flat car tire?  
**Paper:** (none picked)


**Answer** (low_relevance): I can't find this in the papers.


## final-off02 (off_topic, expected: refuse)

**Q:** Who directed the movie Inception?  
**Paper:** (none picked)


**Answer** (low_relevance): I can't find this in the papers.


## final-off03 (off_topic, expected: refuse)

**Q:** What's a good recipe for banana bread?  
**Paper:** (none picked)


**Answer** (low_relevance): I can't find this in the papers.


## final-amb00 (ambiguous, expected: clarify_or_attribute)

**Q:** What baselines do they compare against?  
**Paper:** (none picked)


**Answer** (needs_paper): This question doesn't say which paper it's about, and several papers could fit. Pick one, for example: Text Summarization using Abstract Meaning Representation; Multilingual Sentiment Analysis: An RNN-Based Framework for Limited Data; Distinguishing Antonyms and Synonyms in a Pattern-based Neural Network.


## final-amb01 (ambiguous, expected: clarify_or_attribute)

**Q:** How big is the dataset?  
**Paper:** (none picked)


**Answer** (needs_paper): This question doesn't say which paper it's about, and several papers could fit. Pick one, for example: ReviewQA: a relational aspect-based opinion reading dataset; Text Summarization using Abstract Meaning Representation; Named Entity Recognition for Nepali Language.


## final-amb02 (ambiguous, expected: clarify_or_attribute)

**Q:** Which evaluation metrics are used?  
**Paper:** (none picked)


**Answer** (needs_paper): This question doesn't say which paper it's about, and several papers could fit. Pick one, for example: Revisiting Summarization Evaluation for Scientific Articles; Improving Question Generation With to the Point Context.


## final-amb03 (ambiguous, expected: clarify_or_attribute)

**Q:** What language is the data in?  
**Paper:** (none picked)


**Answer** (needs_paper): This question doesn't say which paper it's about, and several papers could fit. Pick one, for example: Short Text Language Identification for Under Resourced Languages; A Portuguese Native Language Identification Dataset.


## final-mul00 (multi_paper, expected: answer_multi)

**Q:** Which languages do the offensive-language Twitter datasets in these papers cover, and how many tweets does each contain?  
**Paper:** (none picked)

- REF: Predicting the Type and Target of Offensive Posts (OLID): English, 14,100 tweets; Arabic Offensive Language on Twitter: Arabic, 10,000 tweets

**Answer** (answered): Arabic. The dataset contains 10,000 tweets [4], with 1,915 offensive tweets [2].

- claim0 cites []: Arabic
- claim1 cites [2, 4]: The dataset contains 10,000 tweets , with 1,915 offensive tweets
  - [2] Arabic Offensive Language on Twitter: Analysis and Experimen / Data Collection > Annotating Tweets > CLEAN (CLN):: ا زكي! تاني يا فاشل> (“tAny yA zky! tAny yA fA$l” – “again O Zaky! again O loser”). For other ambiguous cases, the annotator searched Twitter to find how actual users used expressions.

Table TABREF11 shows the distribution of the annotated tweets. There are 1,915 offensive tweets, including 225 vulgar tweet and 506 hate speech tweets, and 8,085 clean tweets. To validate the quality of annotation, a random sample of 100 tweets from the data, containing 50 offensive and 50 clean tweets, was given to additional three annotators. We calculated the Inter-Annotator Agreement between the annotators using Fleiss’s Kappa coefficient BIBREF23. The Kappa score was 0.92 indicating high quality annotation and agreement.
  - [4] Arabic Offensive Language on Twitter: Analysis and Experimen / Introduction: Disclaimer: Due to the nature of the paper, some examples contain highly offensive language and hate speech. They don't reflect the views of the authors in any way, and the point of the paper is to help fight such speech. Much recent interest has focused on the detection of offensive language and hate speech in online social media. Such language is often associated with undesirable online behaviors such as trolling, cyberbullying, online extremism, political polarization, and propaganda. Thus, offensive language detection is instrumental for a variety of application such as: quantifying polarization BIBREF0, BIBREF1, trolls and propaganda account detection BIBREF2, detecting the likelihood of hate crimes BIBREF3; and predicting conflict BIBREF4. In this paper, we describe our methodology for building a large dataset of Arabic offensive tweets. Given that roughly 1-2% of all Arabic tweets are offensive BIBREF5, targeted annotation is essential for efficiently building a large dataset. Since our methodology does not use a seed list of offensive words, it is not biased by topic, target, or dialect. Using our methodology, we tagged 10,000 Arabic tweet dataset for offensiveness, where offensive tweets account for roughly 19% of the tweets. Further, we labeled tweets as vulgar or hate speech. To date, this is the largest available dataset, which we plan to make publicly available along with annotation guidelines. We use this dataset to characterize Arabic offensive language to ascertain the topics, dialects, and users' gender that are most associated with the use of offensive language. Though we suspect that there are common features that span different languages and cultures, some characteristics of Arabic offensive language is language and culture specific. Thus, we conduct a thorough analysis of how Arabic users use offensive language. Next, we use the dataset to train strong Arabic offensive language classifiers using state-of-the-art representations and classification techniques. Specifically, we experiment with static and contextualized embeddings for representation along with a variety of classifiers such as a deep neural network classifier

## final-mul01 (multi_paper, expected: answer_multi)

**Q:** Which papers build on ELMo, and what does each one change about it?  
**Paper:** (none picked)

- REF: High Quality ELMo Embeddings for Seven Less-Resourced Languages: trains ELMo on larger corpora for Croatian, Estonian, Finnish, Latvian, Lithuanian, Slovenian and Swedish; Shallow Syntax in Deep Water: pretrains ELMo on predicted shallow syntactic (chunk) parses and adds shallow-syntax features on downstream tasks

**Answer** (answered): The paper "High Quality ELMo Embeddings for Seven Less-Resourced Languages" [1] builds on ELMo by computing ELMo embeddings for seven less-resourced languages on much larger corpora, improving the quality of the embeddings compared to previously available models [1]. The paper "High Quality ELMo Embeddings for Seven Less-Resourced Languages" [3] also builds on ELMo by describing high-quality models for contextual embeddings called ELMo, precomputed for seven morphologically rich, less-resourced languages, enhancing the applicability of ELMo to these languages [3].

- claim0 cites [1]: The paper "High Quality ELMo Embeddings for Seven Less-Resourced Languages"  builds on ELMo by computing ELMo embeddings for seven less-resourced languages on much larger corpora, improving the quality of the embeddings compared to previously available models
  - [1] High Quality ELMo Embeddings for Seven Less-Resourced Langua / ELMo > ELMoForManyLangs: Recently, ELMoForManyLangs BIBREF6 project released pre-trained ELMo models for a number of different languages BIBREF7. These models, however, were trained on a significantly smaller datasets. They used 20-million-words data randomly sampled from the raw text released by the CoNLL 2017 Shared Task - Automatically Annotated Raw Texts and Word Embeddings BIBREF8, which is a combination of Wikipedia dump and common crawl. The quality of these models is questionable. For example, we compared the Latvian model by ELMoForManyLangs with a model we trained on a complete (wikidump + common crawl) Latvian corpus, which has about 280 million tokens. The difference of each model on the word analogy task is shown in Figure FIGREF16 in Section SECREF5. As the results of the ELMoForManyLangs embeddings are significantly worse than using the full corpus, we can conclude that these embeddings are not of sufficient quality. For that reason, we computed ELMo embeddings for seven languages on much larger corpora. As this effort requires access to large amount of textual data and considerable computational resources, we made the precomputed models publicly available by depositing them to Clarin repository.
- claim1 cites [3]: The paper "High Quality ELMo Embeddings for Seven Less-Resourced Languages"  also builds on ELMo by describing high-quality models for contextual embeddings called ELMo, precomputed for seven morphologically rich, less-resourced languages, enhancing the applicability of ELMo to these languages
  - [3] High Quality ELMo Embeddings for Seven Less-Resourced Langua / Introduction: middle of all words' meanings. Consequently, rare meanings of words are poorly expressed with word2vec and the resulting vectors do not offer good semantic representations. For example, none of the 50 closest vectors of the word paper is related to science.

The idea of contextual embeddings is to generate a different vector for each context a word appears in and the context is typically defined sentence-wise. To a large extent, this solves the problems with word polysemy, i.e. the context of a sentence is typically enough to disambiguate different meanings of a word for humans and so it is for the learning algorithms. In this work, we describe high-quality models for contextual embeddings, called ELMo BIBREF3, precomputed for seven morphologically rich, less-resourced languages: Slovenian, Croatian, Finnish, Estonian, Latvian, Lithuanian, and Swedish. ELMo is one of the most successful approaches to contextual word embeddings. At time of its creation, ELMo has been shown to outperform previous word embeddings BIBREF3 like word2vec and GloVe on many NLP tasks, e.g., question answering, named entity extraction, sentiment analysis, textual entailment, semantic role labeling, and coreference resolution.

This report is split into further five sections. In section SECREF2, we describe the contextual embeddings ELMo. In Section SECREF3, we describe the datasets used and in Section SECREF4 we describe preprocessing and training of the embeddings. We describe the methodology for evaluation of created vectors and results in Section SECREF5. We present conclusion in Section SECREF6 where we also outline plans for further work.

## final-mul02 (multi_paper, expected: answer_multi)

**Q:** Which languages are covered by the stance detection datasets in these papers?  
**Paper:** (none picked)

- REF: UTCNN: Chinese (FBFans, Facebook) and English (CreateDebate forum); Stance Detection in Turkish Tweets: Turkish

**Answer** (answered): Turkish [1][2][5].

- claim0 cites [1, 2, 5]: Turkish
  - [1] Stance Detection in Turkish Tweets / Abstract: Stance detection is a classification problem in natural language processing where for a text and target pair, a class result from the set {Favor, Against, Neither} is expected. It is similar to the sentiment analysis problem but instead of the sentiment of the text author, the stance expressed for a particular target is investigated in stance detection. In this paper, we present a stance detection tweet data set for Turkish comprising stance annotations of these tweets for two popular sports clubs as targets. Additionally, we provide the evaluation results of SVM classifiers for each target on this data set, where the classifiers use unigram, bigram, and hashtag features. This study is significant as it presents one of the initial stance detection data sets proposed so far and the first one for Turkish language, to the best of our knowledge. The data set and the evaluation results of the corresponding SVM-based approaches will form plausible baselines for the comparison of future studies on stance detection.
  - [2] Stance Detection in Turkish Tweets / Conclusion: Stance detection is a considerably new research area in natural language processing and is considered within the scope of the well-studied topic of sentiment analysis. It is the detection of stance within text towards a target which may be explicitly specified in the text or not. In this study, we present a stance-annotated tweet data set in Turkish where the targets of the annotated stances are two popular sports clubs in Turkey. The corresponding annotations are made publicly-available for research purposes. To the best of our knowledge, this is the first stance detection data set for the Turkish language and also the first sports-related stance-annotated data set. Also presented in this study are SVM classifiers (one for each target) utilizing unigram and bigram features in addition to using the existence of hashtags as another feature. 10-fold cross validation results of these classifiers are presented which can be used as reference results by prospective systems. Both the annotated data set and the classifiers with evaluations are significant since they are the initial contributions to stance detection problem in Turkish tweets.
  - [5] Stance Detection in Turkish Tweets / Introduction: Stance Detection is described. Also provided are the results of the evaluations of 19 systems participating in two subtasks (one with training data set provided and the other without an annotated data set) of the shared task BIBREF0 .

In this paper, we present a tweet data set in Turkish annotated with stance information, where the corresponding annotations are made publicly available. The domain of the tweets comprises two popular football clubs which constitute the targets of the tweets included. We also provide the evaluation results of SVM classifiers (for each target) on this data set using unigram, bigram, and hashtag features.

To the best of our knowledge, the current study is the first one to target at stance detection in Turkish tweets. Together with the provided annotated data set and the corresponding evaluations with the aforementioned SVM classifiers which can be used as baseline systems, our study will hopefully help increase social media analysis studies on Turkish content.

The rest of the paper is organized as follows: In Section SECREF2 , we describe our tweet data set annotated with the target and stance information. Section SECREF3 includes the details of our SVM-based stance classifiers and their evaluation results with discussions. Section SECREF4 includes future research topics based on the current study, and finally Section SECREF5 concludes the paper with a summary.

## final-mul03 (multi_paper, expected: answer_multi)

**Q:** What new named-entity datasets are introduced in these papers, and what language or domain does each cover?  
**Paper:** (none picked)

- REF: Named Entity Recognition for Nepali Language: a Nepali news NER dataset (PER, LOC, ORG); Named Entities in Medical Case Reports: an English corpus of PubMed Central case reports annotated with cases, conditions, findings, factors and negation modifiers

**Answer** (answered): The new named-entity datasets introduced in these papers are the ILPRL dataset and the OurNepali dataset. The ILPRL dataset covers the Nepali language and is based on news articles [3]. The OurNepali dataset also covers the Nepali language and is based on news articles from 2015-2016 [4].

- claim0 cites []: The new named-entity datasets introduced in these papers are the ILPRL dataset and the OurNepali dataset
- claim1 cites [3]: The ILPRL dataset covers the Nepali language and is based on news articles
  - [3] Named Entity Recognition for Nepali Language / Dataset Statistics > ILPRL dataset: After much time, we received the dataset from Bal Krishna Bal, ILPRL, KU. This dataset follows standard CoNLL-2003 IOB formatBIBREF25 with POS tags. This dataset is prepared by ILPRL Lab, KU and KEIV Technologies. Few corrections like correcting the NER tags had to be made on the dataset. The statistics of both the dataset is presented in table TABREF23.

Table TABREF24 presents the total entities (PER, LOC, ORG and MISC) from both of the dataset used in our experiments. The dataset is divided into three parts with 64%, 16% and 20% of the total dataset into training set, development set and test set respectively.
- claim2 cites [4]: The OurNepali dataset also covers the Nepali language and is based on news articles from 2015-2016
  - [4] Named Entity Recognition for Nepali Language / Dataset Statistics > OurNepali dataset: Since, we there was no publicly available standard Nepali NER dataset and did not receive any dataset from the previous researchers, we had to create our own dataset. This dataset contains the sentences collected from daily newspaper of the year 2015-2016. This dataset has three major classes Person (PER), Location (LOC) and Organization (ORG). Pre-processing was performed on the text before creation of the dataset, for example all punctuations and numbers besides ',', '-', '|' and '.' were removed. Currently, the dataset is in standard CoNLL-2003 IO formatBIBREF25.

Since, this dataset is not lemmatized originally, we lemmatized only the post-positions like Ek, kO, l, mA, m, my, jF, sg, aEG which are just the few examples among 299 post positions in Nepali language. We obtained these post-positions from sanjaalcorps and added few more to match our dataset. We will be releasing this list in our github repository. We found out that lemmatizing the post-positions boosted the F1 score by almost 10%.

In order to label our dataset with POS-tags, we first created POS annotated dataset of 6946 sentences and 16225 unique words extracted from POS-tagged Nepali National Corpus and trained a BiLSTM model with 95.14% accuracy which was used to create POS-tags for our dataset.

The dataset released in our github repository contains each word in newline with space separated POS-tags and Entity-tags. The sentences are separated by empty newline. A sample sentence from the dataset is presented in table FIGREF13.
