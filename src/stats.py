import numpy as np
from scipy.stats import chi2_contingency


def chi2_cramers_v(tabela):
    """Qui-quadrado de independência + V de Cramér para uma tabela de contingência 2D.

    Recebe um crosstab (pandas) e retorna (chi2, p_value, cramers_v).
    """
    chi2, p, dof, _ = chi2_contingency(tabela)
    total = tabela.values.sum()
    graus_menor_dim = min(tabela.shape) - 1
    v = np.sqrt(chi2 / (total * graus_menor_dim))
    return chi2, p, v
