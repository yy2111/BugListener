# BugListener

This is a replication package for `BugListener: Identifying and Synthesizing Bug Reports from Collaborative Live Chats`. 
Our project is public at: <https://github.com/BugListener/BugListener2022>

## Content

- [Project Structure](#ProjectStructure)
- [Project Summary](#ProjectSummary)
- [Models](#Models)
- [Experiment](#Experiment)
- [Results](#Results)
- [Human Evaluation](#HumanEvaluation)

## Project Structure
- `data/`
	- `original_data/ :original dialogs data`
	- `*_bug.json/ :bug-report dialogs data`
	- `*_other.json/ :non bug-report dialogs data`

- `Src/`
	- `dataloader.py : dataset reader for BugListener`
	- `model.py : BugListener model`
	- `FocalLoss.py : focal loss function`
    - `train.py : a file for model training`

## Project Summary
In community-based software development, developers frequently rely on live-chatting to discuss emergent bugs/errors they encounter in daily development tasks. However, it remains a challenging task to accurately record such knowledge due to the noisy nature of interleaved dialogs in live chat data. In this paper, we first formulate the task of identifying and synthesizing bug reports from community live chats, and propose a novel approach, named BugListener, to address the challenges. Specifically, BugListener automates three sub-tasks: 1) Disentangle live chat logs; 2) Identify the bug-report dialog; 3) Synthesize the bug reports.

## Models
I.	The *Dialog Disentanglement* first uses the pipeline of data preprocessing, i.e., spell checking, low-frequency token replacement, acronym and emoji replacement, and broken utterance removal. Then we choose the SOTA model [irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master) to seperate the whole chat log into independent dialogs with reply-to relationships.

II.	The *Utterance Embedding Layer* aims to encode semantic information of words, as well as to learn the representation of utterances. We first encode each word in utterances into a semantic vector by utilizing the deep pre-trained BERT model. Then, we use TextCNN to learn the utterance representation.

III. The *Graph-based Context Embedding layer* aims to capture the graphical context of utterances in one dialog. we first construct a dialog graph. Then, we learn the contextual information of the dialog graph via a two-layer graph neural network. 

IV.	The *Dialog Embedding and Classification layer* aims to obtain the representation of an entire dialog and classify it as either a positive or a negative bug-report dialog. We First utilize the Sum-Pooling and the Max-Pooling layer to obtain the dialog embedding. Then, the label is predicted by feeding the dialog embedding into two Full-Connected (FC) layers followed by the Softmax function.

V.	In this layer, we synthesize the bug reports by utilizing the TextCNN model and Transfer Learning network to classify the sentences into three groups: observed behaviors (OB), expected behaviors (EB), and steps to reproduce the bug (SR).

## Experiments
We propose 3 RQs in our paper, which is related to our experiment:
- RQ1: How effective is BugListener in identifying bug-report dialogs from live chat data?
- RQ2: How effective is BugListener in synthesizing bug reports?
- RQ3 How does each individual component in BugListener contribute to the overall performance?

### Datasets
Our Experiment Dataset is as follows: (Part, Dial, Uttr, Sen are short for participating developers, dialog, utterance, and sentence, respectively. BR and NBR denote bug-report and non-bug-report dialogs. 𝑈𝑟 denotes sentences in reporter’s utterances, and 𝑈 ′ 𝑟 denotes the pruned 𝑈𝑟.)
### Baselines
For RQ1, we compare our BugListner with common baselines: i.e., Naive Bayesian (NB), Random Forest (RF), Gradient Boosting Decision Tree (GBDT), and FastText(FT); additional baselines: i.e., Casper, CNC, and DECA_PD.

For RQ2: we compare our BugListner with common baselines: i.e., Naive Bayesian (NB), Random Forest (RF), Gradient Boosting Decision Tree (GBDT), and FastText(FT); additional baselines: i.e.,BEE:

For RQ3, we compare BugListener with its two variants in bug report identification task: 1) BugListener w/o CNN, which removes the TextCNN. 2) BugListener w/o GNN, which removes the graph neural network. We compare BugListener with its variant without transfer learning (i.e., BugListener w/o TL) in bug report synthesis task.

## Results
RQ1:

Answering RQ1: when comparing with the best Precision-performer among the seven baselines, i.e., GBDT, BugListener can improve its average precision by 5.66%. Similarly, BugListener improves the best Recall-performer, i.e., FastText, by 7.56% for average recall, and improves the best F1 performer, i.e., CNC, by 10.37% for average F1. At the individual project level, BugListener can achieve the best performance in most of the six communities.

RQ2:

Answering RQ2: BugListener can achieve the highest performance in predicting OB, EB, and SR sentences. It outperforms the six baselines in terms of F1. For predicting OB sentences, it reaches the highest F1 (67.37%), improving the best baseline GBDT by 7.21%. For predicting EB sentences, it reaches the highest F1 (87.14%), improving the best baseline FastText by 7.38%. For predicting SR sentences, it reaches the highest F1 (65.03%), improving the best baseline FastText by 5.30%.

RQ3:
Answering RQ3: For BRI task: When compared with BugListener and BugListener w/o GNN, removing the GNN component will lead to a dramatic decrease of the average F1 (by 9.87%) across all the communities. When compared with BugListener and BugListener w/o CNN, removing the TextCNN component will lead to the average F1 declines by 8.21%. For BRS task. We can see that, without the transfer learning from large external bug reports dataset, the F1 will averagely decrease by 3.26%, 6.45%, 14.90% for OB, EB, and SR prediction, respectively.

## Human Evaluation
To further demonstrate the generalization and usefulness of our approach, we apply BugListener on recent live chats from five new communities: Webdriverio, Scala, Materialize, Webpack, and Pandas (note that these are different from our studied communities so that all data of these communities do not appear in our training/testing data). Then we ask human evaluators to assess the correctness, quality, and usefulness of the bug reports generated by BugListener.


For each dialog, the ground truth is obtained based on the majority vote from the three participants, and we use the average score of the three evaluations as the final score. 

Fig (a) shows the bar and pie chart depicting the correctness of BugListener. Among the 31 bug reports identified by BugListener, 24 (77%) of them are correct, while 7 (23%) of them are incorrect. The bar chart shows the correctness distributed among the five communities. The correctness ranges from 63% to 100%. The perceived correctness indicates that BugListener is likely generalized to other open source communities with a relatively good and stable performance. 

Fig (b) shows an asymmetric stacked bar chart depicting the perceived quality and usefulness of BugListener’s bug reports, in terms of description, observed behavior, expected behavior, and step to reproduce. We can see that, the high quality of bug report description is highly admitted, 85% of the responses agree that the bug report description is satisfactory (i.e., “somewhat satisfied” or “satisfied”). The high quality of OB, EB, and S2R are also moderately admitted (62%, 46%, and 58% on aggregated cases, respectively). In addition, the usefulness bar chart shows that 71% of participants agree that BugListener is useful.
