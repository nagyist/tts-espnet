# 🐁POWSM-CTC
<p align="left">
  <a href="https://arxiv.org/abs/tbd">
    <img src="https://img.shields.io/badge/arXiv-tbd-red.svg?logo=arxiv&logoColor=red"/>
  </a>
  <a href="https://huggingface.co/espnet/powsm">
    <img src="https://img.shields.io/badge/HuggingFace-POWSM_CTC-yellow.svg?logo=huggingface&logoColor=yellow"/>
  </a>
</p>

POWSM-CTC is a variant of [POWSM](https://huggingface.co/espnet/powsm),  the first phonetic foundation model that can perform four phone-related tasks:
Phone Recognition (PR), Automatic Speech Recognition (ASR), audio-guided grapheme-to-phoneme conversion (G2P), and audio-guided phoneme-to-grapheme
conversion (P2G).
Its multi-task encoder-CTC structure is based on [OWSM-CTC](https://aclanthology.org/2024.acl-long.549/), 
and trained on the same dataset as POWSM, [IPAPack++](https://huggingface.co/anyspeech).

POWSM-CTC is proposed together with our paper [PRiSM](tbd), the first open-source benchmark for phone recognition systems. 
Its decoding is much faster than encoder-decoder models, with similar or enhanced PR performance on unseen domain.
The checkpoint is available on [HuggingFace](https://huggingface.co/espnet/powsm-ctc)!


### Guidelines
1. Run stage 1-10 in `espnet/egs2/powsm/s2t1`
2. Train: stage 11
3. Eval: `espnet2.bin.s2t_inference_ctc` supports batched greedy decoding:

    ```
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = Speech2TextGreedySearch.from_pretrained(
        "espnet/powsm-ctc",
        device="cuda",
        use_flash_attn=True,
        lang_sym='<unk>',
        task_sym='<pr>',
    )

    res = s2t.batch_decode(
        ["audio1.wav", "audio2.wav"], # a list of audios (path or 1-D array/tensor)
        batch_size=16,
        context_len_in_secs=4,
    )   # res is a list of str

    ```
    See [owsm_ctc_v3.1 recipe](https://github.com/espnet/espnet/tree/master/egs2/owsm_ctc_v3.1/s2t1) for more usage of `Speech2TextGreedySearch`.


### Citations

```BibTex
@article{prism,
      title={PRiSM: Benchmarking Phone Realization in Speech Models},
      year={2026},
      url="tbd",
}

@inproceedings{zhu-etal-2025-zipa,
    title = "{ZIPA}: A family of efficient models for multilingual phone recognition",
    author = "Zhu, Jian  and  Samir, Farhan  and  Chodroff, Eleanor  and  Mortensen, David R.",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.961/",
}
```
