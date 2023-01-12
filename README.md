<img src="./vbpr.png" width="400png"></img>

# VBPR-PyTorch

Implementation of VBPR, a visual recommender model, from the paper ["VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback"](https://arxiv.org/abs/1510.01784).

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

## Notes

Notice that the original VBPR implementation (available at [He's website](https://sites.google.com/view/ruining-he/)) contains a different regularization than the one used in this implementation. The authors update the `gamma_items` embedding in a stronger way for positive than for negative items (dividing the lambda by 10). Notice that [Cornac](https://github.com/PreferredAI/cornac) applies this regularization in their implementation ([code](https://github.com/PreferredAI/cornac/blob/93058c04a15348de60b5190bda90a82dafa9d8b6/cornac/models/vbpr/recom_vbpr.py#L249)). In this implementation, it was not considered because it is hard to do without having to manually calculate the regularization as Cornac (I am prioritizing simplicity and portability of the implementation) or without having to use a custom loss that includes the regularization term (a future version might do this). In the future, I will implement the authors' regularization to check if the results change significantly, but it might not be significant (I assume this regularization was not included in the paper for a reason).

Relevant code:

```cpp
    // adjust latent factors
    for (int f = 0; f < K; f ++) {
        double w_uf = gamma_user[user_id][f];
        double h_if = gamma_item[pos_item_id][f];
        double h_jf = gamma_item[neg_item_id][f];

        gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
        gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
        gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 10.0 * h_jf);
    }
```
