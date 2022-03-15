## 1️⃣Pre-training tasks([**Referenced from**](https://link.springer.com/content/pdf/10.1007/s11431-020-1647-3.pdf))

1. ### 😀Language modeling (LM)

   A statistical language model is a probability distribution over sequences of words. Given such a sequence, say of length *m*, it assigns a probability $P(w_1, ... , w_m) $ to the whole sequence.  

2. ### 😂Masked language modeling (MLM)

   MLM first masks out some tokens from the input sentences and then trains the model to predict the masked tokens by the rest of the tokens. (可以认为是DAE的一种)

   - 🏄‍♂️Enhanced masked language modeling (E-MLM): 

     there are multiple research proposing different enhanced versions of MLM to further improve on BERT.UniLM extends the task of mask prediction on three types of language modeling tasks: unidirectional, bidirectional, and sequence-to-sequence prediction. XLM performs MLM on a concatenation of parallel bilingual sentence pairs, called Translation Language Modeling (TLM). SpanBERT replaces MLM with Random Contiguous Words Masking and Span Boundary Objective (SBO) to integrate structure information into pre-training, which requires the system to predict masked spans based on span boundaries. Besides, StructBERT introduces the Span Order Recovery task to further incorporate language structures.Another way to enrich MLM is to incorporate external knowledge.

3. ### 😊Permuted language modeling (PLM)

   PLM is a language modeling task on a random permutation of input sequences. A permutation is randomly sampled from all possible permutations. Then some ofthe tokens in the permuted sequence are chosen as the target, and the model is trained to predict these targets, depending on the rest of the tokens and the natural positions of targets. 

4. ### 😆Denoising autoencoder (DAE)

   Denoising autoencoder (DAE) takes a partially corrupted input and aims to recover the original undistorted input.

   There are several ways to corrupt text:

   - 🏄‍♂️Token masking: Randomly sampling tokens from theinput and replacing them with [MASK] elements.
   - 🤾‍♂️Token deletion: Randomly deleting tokens from the input. Different from token masking, the model needs to decide the positions of missing inputs.
   - 🏋️‍♂️Text infilling: Like SpanBERT, a number of text spans are sampled and replaced with a single [MASK] token. Each span length is drawn from a Poisson distribution (λ = 3). The model needs to predict how many tokens are missing from a
     span.
   - 🚴‍♀️Sentence permutation: Dividing a document into sentences based on full stops and shuffling these sentences in
     random order.
   - 🚣‍♀️Document rotation: Selecting a token uniformly at random and rotating the document so that it begins with that to-
     ken. The model needs to identify the real start

5. ### 🙂Contrastive learning (CTL)

   Contrastive learning assumes some observed pairs of text that are more semantically similar than randomly sampled text. The idea behind CTL is "learning by comparison".

   There are some recently proposed CTL tasks:

   - 🏄‍♂️Deep InfoMax (DIM): Deep InfoMax is originally proposed for images, which improves the quality of the representation by maximizing the mutual information between an image representation and local regions of the image.
   - 🤾‍♂️Replaced token detection (RTD): Replaced token detection predicts whether a token is replaced given its surrounding context.
   - 🏋️‍♂️Next sentence prediction (NSP): NSP trains the model to distinguish whether two input sentences are continuous segments from the training corpus.
   - 🚴‍♀️Sentence order prediction (SOP): SOP uses two consecutive segments from the same document as positive examples, and the same two consecutive segments but with their order swapped as negative examples.

6. ### 😵Others

   Apart from the above tasks, there are many other auxiliary pre-training tasks designated to incorporate factual knowledge, improve cross-lingual tasks, multi-modal applications, or other specific tasks.

   - 🏄‍♂️Knowledge-enriched PTMs
   - 🤾‍♂️Multilingual and language-specific PTMs
   - 🏋️‍♂️Multi-modal PTMs
   - 🚴‍♀️Domain-specific and task-specific PTMs



## 2️⃣Pre-trained LMs

|                 模型名称                 |        参数量         |     语言     |         模型架构         |         训练目标         |                           训练语料                           | 链接                                                         | 其他                                               |
| :--------------------------------------: | :-------------------: | :----------: | :----------------------: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- | -------------------------------------------------- |
|                BERT-base                 |         110M          | 英/中/多语言 |   Transformer-encoder    |         MLM+NSP          |                BooksCorpus+Wikipedia  (16 GB)                | [github](https://github.com/google-research/bert)            |                                                    |
|                BERT-large                |         340M          |      英      |   Transformer-encoder    |         MLM+NSP          |                    BooksCorpus+Wikipedia                     | [github](https://github.com/google-research/bert)            |                                                    |
|             StructBERT-base              |         110M          |      /       |   Transformer-encoder    |       MLM+NSP+SOP        |                    BooksCorpus+Wikipedia                     | [github](https://github.com/alibaba/AliceMind/tree/main/StructBERT) |                                                    |
|             StructBERT-large             |      340M, 330M       |    英/中     |   Transformer-encoder    |       MLM+NSP+SOP        |                    BooksCorpus+Wikipedia                     | [github](https://github.com/alibaba/AliceMind/tree/main/StructBERT) |                                                    |
|                 SpanBERT                 |           /           |      /       |   Transformer-encoder    |         MLM(SBO)         |                    BooksCorpus+Wikipedia                     | [github](https://github.com/facebookresearch/SpanBERT)       |                                                    |
| ALBERT-base,large,  xlarge,<br />xxlarge |   12M,18M, 59M,233M   |    中/英     |   Transformer-encoder    |         MLM+SOP          |                    BooksCorpus+Wikipedia                     | [github](https://github.com/google-research/albert)          |                                                    |
|         BART-base,<br />large,m          |  140M,  400M,  610M   |  英/多语言   |       Transformer        |           DAE            |                    BooksCorpus+Wikipedia                     | [github](https://github.com/pytorch/fairseq/tree/main/examples/bart) |                                                    |
|            RoBERTa-base,large            |      125M,  355M      |      英      |   Transformer-encoder    |        MLM(动态)         | BooksCorpus+Wikipedia   + CC-NEWS(76 GB)+ OPENWEBTEXT(38 GB)+STORIES(31 GB) | [github](https://github.com/pytorch/fairseq/tree/main/examples/roberta) |                                                    |
|                   XLM                    |           /           |    多语言    |       Transformer        |         MLM，TLM         | Wikipedia+MultiUN+IIT Bombay+OPUS website：(EUbookshop, OpenSubtitles2018,  Tanzil, GlobalVoices) | [github](https://github.com/facebookresearch/XLM)            |                                                    |
|      ELECTRA-small,base<br />,large      |    14M,110M, 335M     |      英      | Generator+ Discriminator |           RTD            | BooksCorpus+Wikipedia, BooksCorpus+Wikipedia+ ClueWeb+CommonCrawl+ Gigaword | [github](https://github.com/google-research/electra)         | [中文版](https://github.com/ymcui/Chinese-ELECTRA) |
|                ERNIE-THU                 |           /           |      英      | Transformer-encoder+  KG |    MLM+NSP+   KG融合     |               BooksCorpus+Wikipedia  +Wikidata               | [github](https://github.com/thunlp/ERNIE)                    |                                                    |
|                ERNIE 3.0                 |          10B          |    中/英     |  Transformer-encoder+KG  | MLM  (Knowledge Masking) |     Chinese text corpora (4TB) 11 different categories.      | [github](https://github.com/PaddlePaddle/ERNIE)              | 未公开                                             |
|                   MASS                   |         120M          |     翻译     |       Transformer        |       Seq2Seq-MLM        |                 WMT16+WMT News Crawl dataset                 | [github](https://github.com/microsoft/MASS)                  |                                                    |
|                Wu Dao 2.0                |  1.75T(涵盖很多模型)  |  中/英/双语  |            \             |            \             |                      WuDaoCorpus 4.9 TB                      | [Official website](https://wudaoai.cn/home)                  | 可下载                                             |
|                CPM-2,MoE                 |       11B,198B        |    中/英     |  Transformer-encoder+KG  |         Span MLM         |              WuDaoCorpus (zh:2.3 TB; en:300 GB)              | [Official website](https://wudaoai.cn/model/detail/CPM%E7%B3%BB%E5%88%97) |                                                    |
|                 UniLM v2                 |         110M          |      英      |   Transformer-encoder    |         MLM+NSP          |    BooksCorpus+Wikipedia   + CC-NEWS+ OpenWebText+Stories    | [github](https://github.com/microsoft/unilm/tree/master/unilm) |                                                    |
|                    M6                    |         100B          | 中文-多模态  |            \             |            \             |              images(1.9 TB),     texts(292 GB)               | [github]()                                                   |                                                    |
|    T5-small,<br />base, large, 3B,11B    | 60M,220M, 770M,3B,11B |      英      |       Transformer        |         Span MLM         |                     Common Crawl(750 GB)                     | [github](https://github.com/google-research/text-to-text-transfer-transformer) |                                                    |
|                  CODEX                   |          12B          |     code     |   Transformer-decoder    |      基于GPT-3微调       |             Github Python files        (159 GB)              | [copilot](https://copilot.github.com/)                       |                                                    |
|             XLNet-base,large             |   similar with bert   |      英      |   Transformer-encoder    |           PLM            |   BooksCorpus+Wikipedia +Giga5+ClueWeb 2012-B,Common Crawl   | [github](https://github.com/zihangdai/xlnet)                 | [中文版](https://github.com/ymcui/Chinese-XLNet)   |
|                   GPT                    |         117M          |      英      |   Transformer-decoder    |            LM            |                         BooksCorpus                          | [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) |                                                    |
|                  GPT-2                   |         1.5B          |      英      |   Transformer-decoder    |            LM            |                     Common Crawl(40 GB)                      | [github](https://github.com/openai/gpt-2)                    |                                                    |
|                  GPT-3                   |         175B          |      英      |   Transformer-decoder    |            LM            | Common Crawl +  WebText datase+two internet-based books corpora+  English-language Wikipedia(570 GB-45 TB raw) | [Official website](https://beta.openai.com)                  | 付费                                               |



## **3️⃣部分大模型可用性调研**

1. BERT：参数量110-340M（Bert系列及其变种等小型PLM一般都已开源参数，可下载本地使用）
2. T5：参数量11B，模型大小约15GB，可下载本地使用
3. GPT-2：参数量1.5B，付费
4. GPT-3：参数量175B，付费，0.7-0.01RMB/1K TOKENS，付费方式（中国不在申请地区）
5. 华为盘古：参数量200B，未开放，在咨询
6. 百度ERNIE3.0：参数量10B，在咨询
7. RoBERTa：参数量125-355M，可下载使用
8. ALBERT：参数量125M
9. 悟道2.0-GLM（General Language Model）：参数10B，申请下载使用
10. 悟道2.0-CPM（Chinese Pretrained Models）：参数2.6,11,198B，申请下载使用
11. BART：参数量400M，可下载使用

![image-20220315121115658](https://raw.githubusercontent.com/BIT-Xu/pic/main/image-20220315121115658.png)

<div align = "center"><a href="https://arxiv.org/pdf/2107.13586.pdf?ref=https://githubhelp.com">Referenced from</a></div>



## 4️⃣Transformer预训练模型适用任务汇总([Referenced from PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html))

| Model                                                        | Sequence Classification | Token Classification | Question Answering | Text Generation | Multiple Choice |
| ------------------------------------------------------------ | ----------------------- | -------------------- | ------------------ | --------------- | --------------- |
| [ALBERT](https://arxiv.org/abs/1909.11942)                   | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| [BART](https://arxiv.org/abs/1910.13461)                     | ✅                       | ✅                    | ✅                  | ✅               | ❌               |
| [BERT](https://arxiv.org/abs/1810.04805)                     | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| [BigBird](https://arxiv.org/abs/2007.14062)                  | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| [Blenderbot](https://arxiv.org/pdf/2004.13637.pdf)           | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| [Blenderbot-Small](https://arxiv.org/pdf/2004.13637.pdf)     | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| [ConvBert](https://arxiv.org/abs/2008.02496)                 | ✅                       | ✅                    | ✅                  | ✅               | ✅               |
| [CTRL](https://arxiv.org/abs/1909.05858)                     | ✅                       | ❌                    | ❌                  | ❌               | ❌               |
| [DistilBert](https://arxiv.org/abs/1910.01108)               | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [ELECTRA](https://arxiv.org/abs/2003.10555)                  | ✅                       | ✅                    | ❌                  | ❌               | ✅               |
| [ERNIE](https://arxiv.org/abs/1904.09223)                    | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [ERNIE-DOC](https://arxiv.org/abs/2012.15688)                | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [ERNIE-GEN](https://arxiv.org/abs/2001.11314)                | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| [ERNIE-GRAM](https://arxiv.org/abs/2010.12148)               | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [GPT](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | ✅                       | ✅                    | ❌                  | ✅               | ❌               |
| [LayoutLM](https://arxiv.org/abs/1912.13318)                 | ✅                       | ✅                    | ❌                  | ❌               | ❌               |
| [LayoutLMV2](https://arxiv.org/abs/2012.14740)               | ❌                       | ✅                    | ❌                  | ❌               | ❌               |
| [LayoutXLM](https://arxiv.org/abs/2104.08836)                | ❌                       | ✅                    | ❌                  | ❌               | ❌               |
| [Mbart](https://arxiv.org/abs/2001.08210)                    | ✅                       | ❌                    | ✅                  | ❌               | ✅               |
| [MobileBert](https://arxiv.org/abs/2004.02984)               | ✅                       | ❌                    | ✅                  | ❌               | ❌               |
| [MPNet](https://arxiv.org/abs/2004.09297)                    | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| [NeZha](https://arxiv.org/abs/1909.00204)                    | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| [ReFormer](https://arxiv.org/abs/2001.04451)                 | ✅                       | ❌                    | ✅                  | ❌               | ❌               |
| [RoBERTa](https://arxiv.org/abs/1907.11692)                  | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [RoFormer](https://arxiv.org/abs/2104.09864)                 | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [SKEP](https://arxiv.org/abs/2005.05635)                     | ✅                       | ✅                    | ❌                  | ❌               | ❌               |
| [SqueezeBert](https://arxiv.org/abs/2006.11316)              | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| [T5](https://arxiv.org/abs/1910.10683)                       | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| [TinyBert](https://arxiv.org/abs/1909.10351)                 | ✅                       | ❌                    | ❌                  | ❌               | ❌               |
| [UnifiedTransformer](https://arxiv.org/abs/2006.16779)       | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| [XLNet](https://arxiv.org/abs/1906.08237)                    | ✅                       | ✅                    | ❌                  | ❌               | ❌               |


