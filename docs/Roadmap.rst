.. _roadmap:

#####################
Roadmap and Changelog
#####################

Xohn/mlfinpy fork (unreleased)
===============================

Changes specific to the `Xohn/mlfinpy <https://github.com/Xohn/mlfinpy>`_ fork,
made while integrating mlfinpy into a downstream AFML-based equity research
project. See "About this fork" in the README/index for a summary;
commit hashes below are on this fork's ``main`` branch.

Bug fixes:

- ``7ad4f60`` — relaxed ``numpy``/``numba`` upper version bounds so the package
  installs with official wheels on Python 3.13.
- ``f1688ce`` — fixed an index-misalignment bug in ``util.frac_diff.frac_diff_ffd``
  that broke on series with a leading NaN.
- ``da76295`` — fixed ``structural_breaks.cusum.get_chu_stinchcombe_white_statistics``
  using the variance instead of the standard deviation in the :math:`S_{n,t}`
  denominator (inflated every statistic ~80-140x).
- ``33c80e4`` — fixed ``cross_validation.cross_validation.PurgedKFold.split()``
  crashing on a non-unique ``samples_info_sets`` index (e.g. pooled multi-asset
  training data).

Backward-compatible additions (all default to upstream's exact behaviour):

- ``da76295`` — ``labeling.labeling.add_vertical_barrier(..., num_bars=None)``
  and ``structural_breaks.sadf.get_sadf(..., model="native")``.
- ``bac960f`` — ``structural_breaks.cusum.get_chu_stinchcombe_white_statistics(..., window=None)``.
- ``9a382d6`` — vectorised ``model="native"`` SADF via cumulative sums (~25x
  faster than the from-scratch-refit path shared by the other models).
- ``61e12ab`` — ``util.volatility.get_daily_vol(..., adjust=True, use_bars=False)``.

0.1.1
=====

Patch with additional method to label the data:

- Update Fixed-time horizon and Raw return labeling methods in the labeling module.
- Add tests for the new labeling methods.
- Update docs with detail articles and user guides.

To-do:
- Complete the user guide section.
- Add more tests to improve code coverage.
- Update ensemble module and cross-validation module.

0.1.0
=====

Initial release:

- Data Structure module
- Labeling module
- Sampling module
- Filters module
- Tests
- Docs
