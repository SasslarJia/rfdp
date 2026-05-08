#
import numpy as np


class PaperProblem(object):
    """Container for the paper-accurate local problem."""

    def __init__(self, query_ids: list, labeled_ids: list, unlabeled_ids: list,
                 a_ids: list, b_ids: list, w_uu: np.ndarray, w_ul: np.ndarray,
                 h_u: np.ndarray, d_u: np.ndarray, e_u: np.ndarray, p_uu: np.ndarray,
                 sub_ids: list, is_connected_to_l, perm_ids: list, w_sub: np.ndarray):
        self._query_ids = list(query_ids)
        self._labeled_ids = list(labeled_ids)
        self._unlabeled_ids = list(unlabeled_ids)
        self._a_ids = list(a_ids)
        self._b_ids = list(b_ids)
        self._w_uu = np.asarray(w_uu, dtype=float)
        self._w_ul = np.asarray(w_ul, dtype=float)
        self._h_u = np.asarray(h_u, dtype=float).reshape(-1)
        self._d_u = np.asarray(d_u, dtype=float).reshape(-1)
        self._e_u = np.asarray(e_u, dtype=float).reshape(-1)
        self._p_uu = np.asarray(p_uu, dtype=float)
        self._sub_ids = list(sub_ids)
        self._is_connected_to_l = np.asarray(is_connected_to_l, dtype=bool).reshape(-1)
        self._perm_ids = list(perm_ids)
        self._w_sub = np.asarray(w_sub, dtype=float)
        self._perm_pos = {tmp_id: idx for idx, tmp_id in enumerate(self._perm_ids)}
        self._sub_pos = {tmp_id: idx for idx, tmp_id in enumerate(self._sub_ids)}

    def empty_unlabeled(self) -> bool:
        return len(self._unlabeled_ids) == 0

    def embed_scores(self, unlabeled_scores, labeled_value: float = 0.0) -> np.ndarray:
        """Convert unlabeled scores back to the original subset order."""
        local_scores = np.full(len(self._sub_ids), labeled_value, dtype=float)
        for idx, tmp_id in enumerate(self._unlabeled_ids):
            local_scores[self._sub_pos[tmp_id]] = float(unlabeled_scores[idx])
        return local_scores

    @property
    def query_ids(self) -> list:
        return self._query_ids

    @property
    def labeled_ids(self) -> list:
        return self._labeled_ids

    @property
    def unlabeled_ids(self) -> list:
        return self._unlabeled_ids

    @property
    def a_ids(self) -> list:
        return self._a_ids

    @property
    def b_ids(self) -> list:
        return self._b_ids

    @property
    def w_uu(self) -> np.ndarray:
        return self._w_uu

    @property
    def w_ul(self) -> np.ndarray:
        return self._w_ul

    @property
    def h_u(self) -> np.ndarray:
        return self._h_u

    @property
    def d_u(self) -> np.ndarray:
        return self._d_u

    @property
    def e_u(self) -> np.ndarray:
        return self._e_u

    @property
    def p_uu(self) -> np.ndarray:
        return self._p_uu

    @property
    def sub_ids(self) -> list:
        return self._sub_ids

    @property
    def is_connected_to_l(self):
        return self._is_connected_to_l

    @property
    def perm_ids(self) -> list:
        return self._perm_ids

    @property
    def w_sub(self) -> np.ndarray:
        return self._w_sub

