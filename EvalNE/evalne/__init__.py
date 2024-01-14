"""
EvalNE
======

EvalNE is a Python package for the evaluation of network embedding methods on a variety of downstream prediction tasks.
These tasks include link prediction, sign prediction, network reconstruction and node classification. Basic embedding
and graph visualization functions are also provided.

See https://evalne.readthedocs.io/en/latest/ for complete documentation.
"""

__author__ = "Alexandru Mara"
__version__ = "0.4.0"
__bibtex__ = """
@misc{Mara2019,
      author = {Alexandru Mara and Jefrey Lijffijt and Tijl De Bie},
      title = {EvalNE: A Framework for Evaluating Network Embeddings on Link Prediction},
      year = {2019},
      archivePrefix = {arXiv},
      eprint = {1901.09691}
}
"""

from src.EvalNE.evalne import evaluation
from src.EvalNE.evalne import methods
from src.EvalNE.evalne import utils
