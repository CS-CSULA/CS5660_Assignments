# NLP Projects

## Confident Machine Translation

### Motivation

- What problem did this project try to solve?

  - In real applications, we would like a system to abstain from making a prediction when its confidence is too low. Unfortunately, machine translation systems are not well calibrated when they are trained on small amounts of data, as it is typical in low resource language pairs. The aim of this project is to enable a machine translation system to give up translating whenever its generations have insufficient quality.

- Who cares? If you are successful, what difference will it make?

  - Whoever deploys MT systems in the real world cares about this, as the cost of not showing a translation is lower than the cost of showing a poor translation.

### Approaches

- How is the problem approached today, and what are the limits of current practice?

  - There is not much literature on this. Usually people use a threshold on the model logprobabilities with a threshold set on the validation set.

- What are the baselines? Is there any potential approaches students can start with, without major change from the baselines?

  - Students can start by using model log-probabilties (average per token, min, median, max) and plot sort of precision/recall curves (fraction of sentences translated VS BLEU).

  - Students can also explore training methods that yield better calibrated models, e.g., by regularizing more or by training ensembles.

### Metrics

- What are the common datasets and benchmarks?

  - What’s the availability of the datasets? Are they open-sourced? Is there any restriction regarding how students should use them?

    - Students can use data and models from [FLORES](https://github.com/facebookresearch/flores)

- What is the state-of-the-art approach, if applicable?

  - Model fine-tuning.

### Scope

- What are possible directions that students could explore? E.g. In terms of modeling, data, efficient computation.

  - students could include an ablation study to show the impact of adapter configuration and size on performance on the task
  - students could explore all of the domains explored in the paper or just the CS domain where domain and task adaptation were most beneficial
  - students could reproduce the results from the paper as a baseline or save time and take the results as presented
  - students could omit domain adaption or task adaption

- How much work in each direction would justify a good grade?

  - a team of 3 should be able to reproduce the main results of the paper related to Task-Adaptive Pretraining. The results related to Augmenting Training Data for Task-Adaptive Pretraining can be omitted.

- What is the ideal size for a team tackling this project?

  - 2 to 4 people

### Resources

- [FLORES](https://github.com/facebookresearch/flores)

## Transfer Learning for Machine Translation Quality Estimation

### Motivation

Machine translation quality estimation (QE) is the task of predicting the quality of a machine translated segment (generally a sentence) without access to a gold-standard (reference) translation. This is an important field of research especially given recent advances in machine translation: for medium to high resource languages, the level of fluency of translations tends to be very high, making translation seemingly high quality, while they may still contain meaning preservation errors. This makes it hard for models and even humans to identify mistakes in translations. Quality prediction models are built from a set of labelled instances for a given language pair and information from both the source and translated segments. This information varies from features such as language model scores for these sentences (Specia et al., 2015), to more complex word representations learnt with neural models (Kim et al., 2017; Ive et al., 2018; Kepler et al., 2019a).

### Limitation in current approaches

As with most supervised machine learning problems, an important bottleneck in QE is the need for labelled data. Most existing datasets predict a continuous score, e.g. in [0,100] and suffer from skewed distributions towards the high end of the quality spectrum (i.e. most translations are good). To alleviate the need for labelled data, state-of-the-art models for QE rely on pre-trained representations, such as those provided by BERT, XLM, or Laser (Kepler et al., 2019b). However, these models still require at least a few thousand instances for fine-tuning. This means that for each language pair (and possibly text domain), a new labelled dataset has to be collected to build language & domain-specific models. In this project the aim is to investigate ways to build QE models where labelled data only exists for other languages.

### Task definition

The task is to predict an adequacy-oriented quality score in [0,100] (the so-called Direct Assessment, or DA score), where 0 indicates a completely incorrect and disfluent translation, and 100 indicates a perfect translation for two language pairs, English-German and English-Chinese. This could be done in various ways, but ideally you should leverage the labels for the other languages and/or NMT model information.

### Datasets

The following open-source resources are available:

- Three sets of 7K training instances and 1K dev sets for different languages: [Estonian-English](https://github.com/facebookresearch/mlqe/blob/main/data/et-en.tar.gz), [Romanian-English](https://github.com/facebookresearch/mlqe/blob/main/data/ro-en.tar.gz), [English-Nepali](https://github.com/facebookresearch/mlqe/blob/main/data/ne-en.tar.gz) annotated with z-normalised DA scores. The score to predict is in the ‘z_mean’ column.

- Two sets of 1K development instances for [English-German](https://github.com/facebookresearch/mlqe/blob/main/data/en-de.tar.gz) and [English-Chinese](https://github.com/facebookresearch/mlqe/blob/main/data/en-zh.tar.gz) annotated with z-normalised DA scores. The score to predict is in the ‘z_mean’ column.

- Log probabilities for each translated word as given by the NMT models used to produce the translations in all the languages (the models themselves can also be made available) part the tar.gz files above.

- Two sets of 1K test instances for English-German and English-Chinese, where you will need to predict the z-normalised score for each [instance](https://github.com/facebookresearch/mlqe/blob/main/data/README-data.md).

### Baseline and state-of-the-art performance

The baseline will be a bidirectional recurrent neural net (RNN) model that encodes pre-trained word embeddings (multilingual BERT) and predicts DA scores as output. This will be based on the Predictor-Estimator architecture in the open source [openKiwi tool](https://github.com/Unbabel/OpenKiwi) (Kepler et al., 2019), but where the Predictor vectors are replaced by [multilingual BERT vectors](https://github.com/google-research/bert/blob/master/multilingual.md). Since the data to be used is new, we have not yet established state of the art performance on this data.

### Evaluation

Evaluation will be performed using Pearson Correlation and Root Mean Squared Error between the predictions and the gold-labels on a test set of 1,000 instances. Success in this task will indicate that QE models can be built for a wide range of languages without the need for labelled data in all languages. The test set will be kept blind with the evaluation run through Codalab.

### Possible directions

Possible directions include various flavors of transfer learning from the label data in other languages, including fine-tuning on the dev set, multi-task learning, upsampling, etc.; use of better pre-trained representations; ways to automatically collect labelled data for the two languages.

### Additional resources

Any additional corpus, pre-trained models and other resources, as well as existing software libraries can be used for the project. This will require use of GPUs (Google colab could be used).

### References

- Hyun Kim, Jong-Hyeok Lee, and Seung-Hoon Na. 2017. Predictor-estimator using multilevel task learning with stack propagation for neural quality estimation. In Proceedings of the Second Conference on Machine Translation (WMT), pages 562–568.

- Julia Ive, Frédéric Blain and Lucia Specia. 2018. DeepQuest: a framework for neural-based Quality Estimation. In Proceedings of COLING 2018, the 27th International Conference on Computational Linguistics.

- Fábio Kepler, Jonay Trénous, Marcos Treviso, Miguel Vera and André F. T. Martins. 2019a. OpenKiwi: An Open Source Framework for Quality Estimation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics - System Demonstrations, pages 117—122.

- Fábio Kepler, Jonay Trénous, Marcos Treviso, Miguel Vera, António Góis, M. Amin Farajian, António Lopes and André F. T. Martins. 2019b. Unbabel's Participation in the WMT19 Translation Quality Estimation Shared Task. In Proceedings of the Fourth Conference on Machine Translation.

## Transformer Adapters

### Motivation

- What problem did this project try to solve?

  - To achieve a SoTA result on a down-stream NLP task, a pre-trained Transformer language model is fine-tuned to the task. With standard fine-tuning, the pre-trained weights and a new top-layer are cotrained resulting a whole new set of weights for each new task. For large models which can be 100s of MB in size, storing and sharing a complete set of weights for each task can be a challenge for multi-task applications.

  - One possible solution is to train a single multi-task model. However, training a single multi-task model is challenging to train and requires simultaneous access to all tasks during training. Creating a single multi-task model by sequential fine-tuning runs into the problem of catastrophic forgetting (McCloskey and Cohen, 1989). Recently adapters ([Rebuffi et al., 2017; Houlsby et al., 2019](https://arxiv.org/abs/1705.08045)) have appeared as a parameter-efficient alternative to fine-tuning. On the [GLUE](https://gluebenchmark.com/) benchmark, adapters almost match ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)) the performance of fully fine-tuned BERT, but uses only 3% tasks pecific parameters, while fine-tuning uses 100% task-specific parameters.

- Who cares? If you are successful, what difference will it make?

  - Adapter-based tuning requires training much fewer parameters while attaining similar performance to fine-tuning. With full fine-tuning, avoiding over-fitting of the adaptation corpus can be challenging. This is much less of an issue with adapters. For multi-task learning, adapter training permits sequentially training while standard training does not. Finally, adapters are modular, composable ([Pfeiffer et al., 2020](https://arxiv.org/abs/2005.00247)) and easily share-able. In the future, an updated adapter (order of MB) could be deployed to update a task-specific model in the field instead of an entirely new model checkpoint (order of GB)

### Problem

- In [Gururangan et al. (2020)](https://arxiv.org/abs/2004.10964), the authors show that domain and task adaptation of the RoBERTa language model (LM) prior to task fine-tuning gives better results than task fine-tuning alone. The improvement is more prominent the lesser the overlap between task domain and the original corpus used to pre-train the language model. We would like to understand if it possible to achieve similar results with adapter-based methods.

- Other interesting problems:

  - Create adapters for a transformer not yet adapted (for which fine-tuned modes [already exist](https://huggingface.co/models)) and compare adapter performance to fine-tuning.

  - Create an adapter for a new task (for which fine-tuned model already exists) and compare adapter performance

  - Experiment with different adapter architectures and configurations for which there is already a published result and achieve a new state of the art results in size or performance.

  - In the ablation study in [Houlsby et al. (2019)](https://arxiv.org/abs/1902.00751), the authors show that adapters at lower layers have less impact than adapters at higher layers. Experiment with smaller adapters at lower layers and large adapters at higher layers in order to achieve a SoTA results with less parameters.

### Approaches

- How is the problem approached today, and what are the limits of current practice?

  - In [Gururangan et al. (2020)](https://arxiv.org/abs/2004.10964), the author’s use LM fine-tuning to adapt a pre-trained RoBERTa model to a domain and then a task. Fine-tuning requires generating a new set of weights and must be done serially.

- What are the baselines? Is there any potential approaches students can start with, without major change from the baselines?

- Students can use the [adapter-transformers](https://adapterhub.ml/) framework ([Pfeiffer et al., 2020](https://arxiv.org/abs/2007.07779)) to build and train domain and task specific adapters to compare with the previously published results ([Gururangan et al., 2020](https://arxiv.org/abs/2004.10964)) obtained through standard fine-tuning.

- The task and domain adapters can be trained sequentially or in-parallel. The adapters can be
composed or stacked

### Metrics

- What are the common datasets and benchmarks?

  - What’s the availability of the datasets? Are they open-sourced? Is there any restriction regarding how students should use them?

    - Students can use the same datasets as the original paper or potentially the [kaggle arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).

- What is the state-of-the-art approach, if applicable?

  - Model fine-tuning.

### Scope

- What are possible directions that students could explore? E.g. In terms of modeling, data, efficient computation.

  - students could include an ablation study to show the impact of adapter configuration and size on performance on the task
  - students could explore all of the domains explored in the paper or just the CS domain where domain and task adaptation were most beneficial
  - students could reproduce the results from the paper as a baseline or save time and take the results as presented
  - students could omit domain adaption or task adaption

- How much work in each direction would justify a good grade?

  - a team of 3 should be able to reproduce the main results of the paper related to Task-Adaptive Pretraining. The results related to Augmenting Training Data for Task-Adaptive Pretraining can be omitted.

- What is the ideal size for a team tackling this project?

  - 2 to 4 people

### Resources

- https://course.fast.ai/#using-a-gpu/
- https://adapterhub.ml/
- https://huggingface.co/transformers/

### References

1. Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith, “[Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)“, In ACL 2020.

2. Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly, “[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)”, In ICML 2019.

3. McCloskey, M. and Cohen, N. J., “Catastrophic interference in connectionist networks: The sequential learning problem”, In Psychology of learning and motivation. 1989.

4. Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, Iryna Gurevych, “[AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779)”, arXiv preprint.

5. Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych, “[AdapterFusion: NonDestructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247)”, arXiv preprint.

6. Sylvestre-Alvise Rebuffi, Hakan Bilen, Andrea Vedaldi, “[Learning multiple visual domains with residual adapters](https://arxiv.org/abs/1705.08045)”, In NeurIPS 2017.
