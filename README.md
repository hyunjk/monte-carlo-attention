# Fast Monte-Carlo Approximation of the Attention Mechanism 


We introduce Monte-Carlo Attention (MCA), a randomized approximation method for reducing the computational cost of self-attention mechanisms in Transformer architectures. MCA exploits the fact that the importance of each token in an input sequence varies with respect to their attention scores; thus, some degree of error can be tolerable when encoding tokens with low attention. Using approximate matrix multiplication, MCA applies different error bounds to encode input tokens such that those with low attention scores are computed with relaxed precision, whereas errors of salient elements are minimized. MCA can operate in parallel with other attention optimization schemes and does not require model modification. We study the theoretical error bounds and demonstrate that MCA reduces attention complexity (in FLOPS) for various Transformer models by up to 11× in GLUE benchmarks without compromising model accuracy.


Paper link: [arXiv:2201.12854](https://arxiv.org/abs/2201.12854)

## Installation
Install `monte-carlo-attention` package via pip

```
pip install ./monte-carlo-attention
```

## Experiment

To reproduce experimental results, run:
```
python ./experiments/text_cls/exp.py
```