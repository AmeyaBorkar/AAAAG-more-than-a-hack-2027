"""Architectures other than the baseline `team_net`: one module per design.

Add a design by adding a file here that registers it (the contract is in `aaaag/ml/model.py`):

    @register_architecture("my_design", MyConfig)
    class MyDesign(nn.Module): ...

and name it in a run config (`arch: my_design`, then a `model:` block with its fields). Every
module in this package is imported the first time an architecture is looked up, so no list needs
editing and a checkpoint of the design loads wherever the file exists.
"""
