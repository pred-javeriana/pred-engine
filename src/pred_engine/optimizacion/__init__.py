"""Walk-forward evaluation layer.

Runs the WFV engine across (SKU × model × cut) tasks, computes accuracy
metrics (MASE, RMSSE, bias), applies Diebold-Mariano and HLN statistical
tests, and determines the champion model through the cascade protocol.
"""
