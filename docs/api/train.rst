Training Scripts
================

Each script in ``libdse.train`` trains one :class:`~libdse.nets.VanillaAutoEncoder`
variant.  All scripts share the same training loop structure; they differ only
in the feature extractor, network size, and hyperparameters.

Run any script as a module from the repository root, for example::

    python -m libdse.train.simpleAE_logmag_nc

.. contents:: Scripts
   :local:
   :depth: 1

----

Log-magnitude — ``simpleAE_logmag_nc``
---------------------------------------

Production model.  Operates on log-magnitude STFT frames at 8 kHz, following
Nossier et al. (2020) architecture (d).

.. automodule:: libdse.train.simpleAE_logmag_nc
   :members:
   :undoc-members: False

----