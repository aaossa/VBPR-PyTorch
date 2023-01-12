# Datasets

## [Tradesy Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#bartering_data)

> **Paper:** VBPR: visual bayesian personalized ranking from implicit feedback
>
> **Available at:** [arXiv](https://arxiv.org/abs/1510.01784)

```bibtex
@inproceedings{he2016vbpr,
  title={VBPR: visual bayesian personalized ranking from implicit feedback},
  author={He, Ruining and McAuley, Julian},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={30},
  number={1},
  year={2016}
}
```

You can download the files using `wget`:

```bash
wget --no-check-certificate --no-clobber https://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz -P ./data/Tradesy/
wget --no-check-certificate --no-clobber https://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b -P ./data/Tradesy/
wget --no-check-certificate --no-clobber https://jmcauley.ucsd.edu/data/tradesy/tradesy_item_urls.json.gz -P ./data/Tradesy/
```
