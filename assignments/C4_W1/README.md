# Neural Machine Translation

- Accept [C4_W1](https://classroom.github.com/a/8KAZv250)

---

## Data

Dataset is getting large, so we can not push them in github, you need to download data from [google drive](https://drive.google.com/drive/folders/1CYia1fp1T7GbeAL0G4HfuG-CLn93emBK?usp=share_link)

- `model.pkl.gz`: put it under `./C4_W1/` folder.
- `opus-train.tfrecord-00000-of-00002` and `opus-train.tfrecord-00001-of-00002`: put these two under `./C4_W1/data/opus/medical/0.1.0/` folder.

Note: do **NOT** push these three files to github repo, otherwise you will get 0 on this assignment.

## Lab

- Basic Attention

  In this ungraded lab, you will see how Basic Attention works using Numpy. You will compute alignment scores and turn them to weights for the encoder output vectors.
  
- Scaled Dot-Product Attention

  In this ungraded lab, you will see how to implement the Scaled Dot-Product Attention (also known as Queries, Keys, Values Attention)

- BLEU Score

  In this ungraded lab, you will implement a popular metric for evaluating the quality of machine-translated text: the BLEU score proposed by Kishore Papineni, et al. In their 2002 paper “BLEU: a Method for Automatic Evaluation of Machine Translation“.

- Stack Semantics

  In this ungraded lab, you will see how stack semantics work in Trax. It will help in understanding how combinators like `tl.Select` and `tl.Residual` work and will help prepare you for this week's graded assignment.

## Assignment

- NMT with Attention
