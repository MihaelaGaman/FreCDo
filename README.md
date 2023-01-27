# FreCDo
FreCDo: A Large Corpus for French Cross-Domain Dialect Identification

## 1. License Agreement

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

You are free to **share** (copy and redistribute the material in any medium or format) and **adapt** (remix, transform, and build upon the material) under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## 2. Citation

Please cite the corresponding work (see citation.bib file to obtain the citation in BibTex format) if you use this data set and software (or a modified version of it) in any scientific work:

**[1] Gaman, Mihaela, Adrian-Gabriel Chifu, William Domingues, and Radu Tudor Ionescu. "FreCDo: A Large Corpus for French Cross-Domain Dialect Identification." arXiv preprint arXiv:2212.07707 (2022). [(link to paper)](https://arxiv.org/abs/2212.07707).**


## 3. Description

#### Task Information

The task is to train a model on news samples collected from a set of publication sources and evaluate it on news samples collected from a different set of publication sources. Not only the sources are different, but also the topics. Therefore, the task is to build a model for a cross-domain 4-way classification by dialect task, in which a classification model is required to discriminate between the French (FH), Swiss (CH), Belgian (BE) and Canadian (CA) dialects across different news samples. 

The FreCDo data set contains French, Swiss, Belgian and Canadian samples of text collected from the news domain. The corpus is divided into training, validation and test, such that the publication sources and topics are distinct across splits. The training set contains 358,787 samples. The development set is composed of 18,002 samples. Another set of 36,733 samples are kept for testing. All samples are preprocessed in order to replace named entities with a special tag: \$NE\$. For more details, please refer to the paper [1].

#### Data Organization

The training data contains the following files:

	train.txt - training set
	train.labels - training labels
	dev.txt - development/validation set
	dev.labels - development/validation labels
	test.txt - test set
	test.labels - test labels
	
Each line in the *.txt files is tab-delimited in the format:

	dialect-label<tab>text-sample

Each line in the *.labels files is in the format:

	dialect-label
