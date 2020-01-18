# WSDM 2020 cup digsci citation intent recognization
 This is the solution for WSDM 2020 cup ciatation recogniation. Two phrase of recall by BM25 and rerank by bert is applied. The MAP@3 of recall on phrase-A is 0.27, on phrase-B is 0.29. After rerank by bert, the MAP@3 is 0.33 on phrase-A and 0.36 on phrase-B, respectively.
 The BM25 we adopted is from Anserini [https://github.com/castorini/anserini], and the bert we adopted is modified from paper "PASSAGE RE-RANKING WITH BERT". 

## 1. Prerequirements
##### 1.1 Ansirini is need to create index to boost retrive. 
##### 1.2 python 3.6, tensorflow 1.11.0


## 2. Steps to recall and rerank
##### 2.1 In "data_preprocess" file, preprocessing and generate key sentences is given. We did not use all descriptions, since descriptions + abstract is longer than 512. In fact, we found that the first sentence of descriptions + the referrence sentence + the sencence before the referrence sentence is better than other setences. 
##### 2.2 In "first_step_recall_by_BM25" file, BM25 is computed. Before that, you need construct data index by Anserini, since the candidate set is so large. 
##### 2.3 In "second_step_rerank_by_bert" file, after the BM25 recall top K (K = 10, 20, 50, 100, ...) papers, the bert need to rerank them. 
##### 2.3.1 "convert_msmarco_to_tfrecord.py" convert data to tfrecord format 
##### 2.3.2 "prepare_triple_train.py" prepare inputs for bert 
##### 2.3.4 "run_msmarco_ljf2.py" run train and eval 


## 3. Notations
##### 3.1 We treat this problem as a binary classification problem, the ground-truth is labeled pos, the random selected non ground-truth is labeled as neg. Then, we reimplement bert by pytorch, but it CAN NOT DO ANY IMPROMENTS. 
##### 3.2 We also tried CEDR: Contextualized Embeddings for Document Ranking, but it is NOT EFFECTIVE. 
##### 3.3 We also tried Cross-domain modeling of sentence-level evidence for document retrieval, but it is NOT SO EFFECTIVE, but is better than CEDR. 


## 4. Links to Other Solutions
##### 4.1 [rank: second, Team: SimpleBaseline, use data leak](https://github.com/steven95421/WSDM_SimpleBaseline). 
##### 4.2 [rank: 4th, Team: dlutycx , BM25 + bert](https://github.com/chengsyuan/WSDM-Adhoc-Document-Retrieval). 
##### 4.3 [Team: nlp_rabbit](https://github.com/supercoderhawk/wsdm-digg-2020).  
##### 4.4 [A fast BM25](https://github.com/shuiliwanwu/wsdm_cup2020). 
##### 4.5 [rank: 3th, Team: xiong](https://github.com/xiong666/wsdm2020_diggsci/tree/master/code).

## 5. References
##### 5.1 Nogueira R, Cho K. Passage Re-ranking with BERT[J]. arXiv preprint arXiv:1901.04085, 2019.
##### 5.2 Yang W, Zhang H, Lin J. Simple applications of bert for ad hoc document retrieval[J]. arXiv preprint arXiv:1903.10972, 2019. 
##### 5.3 Yilmaz Z A, Yang W, Zhang H, et al. Cross-domain modeling of sentence-level evidence for document retrieval[C]//Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019: 3481-3487.
##### 5.4 PASSAGE RANKING WITH WEAK SUPERVISION \<br>
##### 5.5 Qiao Y, Xiong C, Liu Z, et al. Understanding the Behaviors of BERT in Ranking[J]. arXiv preprint arXiv:1904.07531, 2019.
##### 5.6 CEDR: Contextualized Embeddings for Document Ranking 

