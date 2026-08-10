import pandas as pd
import pytest

from src.stats import chi2_cramers_v


def test_chi2_cramers_v_tabela_independente():
    tabela = pd.DataFrame([[10, 10], [10, 10]], index=["x", "y"], columns=["a", "b"])
    chi2, p, v = chi2_cramers_v(tabela)
    assert chi2 == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_chi2_cramers_v_tabela_totalmente_associada():
    tabela = pd.DataFrame([[50, 0], [0, 50]], index=["x", "y"], columns=["a", "b"])
    chi2, p, v = chi2_cramers_v(tabela)
    assert chi2 > 0
    assert p < 0.001
    assert v > 0.9
