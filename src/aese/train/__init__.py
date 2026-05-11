"""Training scripts for DAE variants.

Each script in this sub-package trains one :class:`~aese.nets.VanillaAutoEncoder`
variant with a different feature representation.  All scripts share the same
training loop structure; they differ only in feature extractor, network size,
and hyperparameters.

Scripts
-------
.. autosummary::

   aese.train.simpleAE_logmag_nc
"""
