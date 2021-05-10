# assess-multilingual-bert

This is the coder for paper: __On the Language Neutrality of Pre-trained Multilingual Representations__
by Jindřich Libovický, Rudolf Rosa and Alexander Fraser published in Findings of EMNLP 2020

The paper evaluates contextual multilingual representations on tasks that should more directly evaluate the language neutrality of the representations than the usual evaluation using zero-shot cross-lingual transfer.

The tasks are:

* Langauge identification: [`lang_id.py`](https://github.com/jlibovicky/assess-multilingual-bert/blob/master/lang_id.py)

* Crosslingual sentence retrieval: [`sentence_retrieval.py`](https://github.com/jlibovicky/assess-multilingual-bert/blob/master/sentence_retrieval.py)

* Word alignment: [`word_alignment.py`](https://github.com/jlibovicky/assess-multilingual-bert/blob/master/word_alignment.py)

* Machine translation quality estimation: [`qe_by_cosine.py`](https://github.com/jlibovicky/assess-multilingual-bert/blob/master/qe_by_cosine.py)

## Cite

```bibtex
@inproceedings{libovicky-etal-2020-language,
    title = "On the Language Neutrality of Pre-trained Multilingual Representations",
    author = "Libovick{\'y}, Jind{\v{r}}ich  and
      Rosa, Rudolf  and
      Fraser, Alexander",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.150",
    doi = "10.18653/v1/2020.findings-emnlp.150",
    pages = "1663--1674",    
}
```
