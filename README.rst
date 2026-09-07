===========================================================
Mlfin.py - Advance Machine Learning application in Finance
===========================================================

.. image:: https://img.shields.io/pypi/v/mlfinpy.svg
        :target: https://pypi.python.org/pypi/mlfinpy
        :alt: PyPI Version


.. image:: https://img.shields.io/pypi/pyversions/mlfinpy.svg
        :target: https://pypi.python.org/pypi/mlfinpy
        :alt: Python Versions


.. image:: https://img.shields.io/badge/Platforms-linux--64,win--64,osx--64-orange.svg?style=flat-square
        :target: https://pypi.python.org/pypi/mlfinpy
        :alt: Platforms


.. image:: https://img.shields.io/badge/license-MIT-brightgreen.svg
        :target: https://pypi.python.org/pypi/mlfinpy
        :alt: MIT License


.. image:: https://img.shields.io/github/actions/workflow/status/baobach/mlfinpy/main.yml
        :target: https://github.com/baobach/mlfinpy
        :alt: Build Status


.. image:: https://codecov.io/github/baobach/mlfinpy/coverage.svg?branch=main
        :target: https://codecov.io/github/baobach/mlfinpy
        :alt: Coverage


.. image:: https://readthedocs.org/projects/mlfinpy/badge/?version=latest
        :target: https://mlfinpy.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


**MLfin.py** is an Advance Machine Learning toolbox for financial applications. The main ideas is using
proprietary works and code snippent by Dr. Marcos López de Prado to build a morden Pythonic package
that implements newest tech stacks from various libraries such as Numpy, Pandas, Numba, and Scikit-Learn.
This work inspired by the library `MlFinLab <https://github.com/hudson-and-thames/mlfinlab>`_ by
**Hudson and Thames**. Unfortunately, the library is closed-source and I believe in the power of open
source projects, it motivates me to build this package from ground up.

Leverage best practice in packaging Python library, morden documentation style and comprehensive examples,
**MLfin.py** will be the great tool for Quant Researchers, Algorithmic Traders, and Data Scientists as well as
Finance students to reproduce the complex data transformation, labeling, sampling and feature engineering
techniques with ease.

About this fork
================

This is a personal fork (`Xohn/mlfinpy <https://github.com/Xohn/mlfinpy>`_) of
`baobach/mlfinpy <https://github.com/baobach/mlfinpy>`_, maintained to support
a downstream AFML-based equity research project. All credit for the
original package design and implementation goes to Robert Bach (see `About
<https://mlfinpy.readthedocs.io/en/latest/About.html>`_) — this fork only adds
targeted bug fixes and a few backward-compatible extensions found while
integrating mlfinpy into a real, multi-asset daily-bar pipeline. If you don't
need these specific fixes, use the upstream project instead. Full changelog in
`docs/Roadmap.rst <docs/Roadmap.rst>`_.

Bug fixes
---------

- ``util.frac_diff.frac_diff_ffd``: fixed an index-misalignment bug that crashed
  (or silently misaligned dates) on any series with a leading NaN.
- ``structural_breaks.cusum.get_chu_stinchcombe_white_statistics``: fixed the
  :math:`S_{n,t}` denominator, which used the variance instead of its square
  root (the standard deviation) — this inflated every statistic by roughly
  :math:`1/\sigma_t` (commonly 80-140x for daily-return volatility), making the
  test almost always flag a "significant" break regardless of whether one
  occurred.
- ``cross_validation.cross_validation.PurgedKFold``: ``.split()`` crashed
  (``TypeError: '<' not supported between instances of 'int' and 'slice'``)
  whenever ``samples_info_sets`` had a non-unique index — e.g. pooling labeled
  events across multiple assets that can share the same event date.
- Relaxed the ``numpy``/``numba`` upper version bounds so the package actually
  installs with official wheels on Python 3.13 (the old bounds forced ``pip``
  onto an experimental, crash-prone MINGW numpy build on Windows).

Backward-compatible additions
------------------------------

All new arguments default to ``None``/``False`` and reproduce upstream's exact
existing behaviour, so this fork is a drop-in replacement.

- ``labeling.labeling.add_vertical_barrier(..., num_bars=None)``: advance
  ``num_bars`` bar *positions* ahead of each event instead of a calendar-time
  offset. On daily-bar data, the default calendar-time search silently
  shortens the intended holding period across weekends/holidays.
- ``structural_breaks.cusum.get_chu_stinchcombe_white_statistics(..., window=None)``:
  bound the reference-point search and the :math:`\sigma_t` estimate to the
  trailing ``window`` observations, instead of expanding from the series'
  start (which becomes almost trivially "significant" on long single-asset
  histories).
- ``structural_breaks.sadf.get_sadf(..., model="native")``: the plain
  Phillips-Wu-Yu (2011) / AFML Ch.17 base SADF spec (constant + lagged level +
  lagged diffs, no deterministic trend term) — none of the existing model
  choices provide it, they all add a trend regressor. Vectorised via
  cumulative sums (the other models' shared from-scratch-refit inner loop
  would be ~25x slower on a 1500-observation daily series).
- ``util.volatility.get_daily_vol(..., adjust=True, use_bars=False)``:
  ``adjust=`` is passed through to the EWM ``.std()`` call; ``use_bars=True``
  switches to plain ``close.pct_change()`` (Snippet 3.1's default calendar-day
  search steps one bar further back than intended whenever "1 day ago" lands
  exactly on an existing timestamp).

Installation
============
Installation can then be done via pip::

    pip install mlfinpy


For the sake of best practice, it is good to do this with a dependency manager. I suggest you
set yourself up with `poetry <https://github.com/sdispater/poetry>`_, then within a new poetry project
run:

.. code-block:: text

    poetry add mlfinpy

.. note::
    If any of these methods don't work, please `raise an issue
    <https://github.com/baobach/mlfinpy/issues>`_ with the ``packaging`` label on GitHub.



For developers
--------------

If you are planning on using Mlfinpy as a starting template for significant
modifications, it probably makes sense to clone the repository and to just use the
source code:

.. code-block:: text

    git clone https://github.com/baobach/mlfinpy

    # or, for this fork (bug fixes and additions, see "About this fork" above):
    git clone https://github.com/Xohn/mlfinpy

Alternatively, if you still want the convenience of a global ``from mlfinpy import x``,
you should try:

.. code-block:: text

    pip install -e git+https://github.com/baobach/mlfinpy.git

    # or, for this fork:
    pip install -e git+https://github.com/Xohn/mlfinpy.git

Work with HFT Data
==================
In reality, testing code snippets through the first 3 chapters of the book is challenging as it relies on HFT data to
create the new financial data structures. Sourcing the HFT data is very difficult and thus `TickData LLC`_ provides
the full history of S&P500 Emini futures tick data and available for purchase.

I am not affiliated with TickData in any way but would like to recommend others to make use of their service. The full
history costs $750 and is worth every penny. They have really done a great job at cleaning the data and providing
it in a user friendly manner.

.. _TickData LLC: https://www.tickdata.com/

Download Sources
----------------

TickData does offer about 20 days worth of raw tick data which can be sourced from their website `link`_.
For those of you interested in working with a two years of sample tick, volume, and dollar bars, it is provided for in
the `research repo`_. You should be able to work on a few implementations of the code with this set.

.. _link: https://s3-us-west-2.amazonaws.com/tick-data-s3/downloads/ES_Sample.zip
.. _research repo: https://github.com/hudson-and-thames/research/tree/master/Sample-Data

Searching for free tick data can be a challenging task. The following three sources may help:

1. `Dukascopy`_. Offers free historical tick data for some futures, though you do have to register.
2. Most crypto exchanges offer tick data but not historical (see `Binance API`_). So you'd have to run a script for a few days.
3. `Blog Post`_: How and why I got 75Gb of free foreign exchange “Tick” data.

.. _Dukascopy: https://www.dukascopy.com/swiss/english/marketwatch/historical/
.. _Binance API: https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
.. _Blog Post: https://towardsdatascience.com/how-and-why-i-got-75gb-of-free-foreign-exchange-tick-data-9ca78f5fa26c

Project principles and design decisions
=======================================

- It should be easy to swap out individual components of each module
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- The goal is creating a framework to build a robust and functional library for
  machine learning applications.
- Everything that has been implemented should be tested and formatted with lattest
  requirements.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this,
  I have deferred **all** formatting decisions to `Black
  <https://github.com/ambv/black>`_, `Flake8
  <https://github.com/PyCQA/flake8>`_, and `Isort
  <https://github.com/PyCQA/isort>`_.

Credits
=======

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
