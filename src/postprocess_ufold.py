import numpy as np

PAIR_OK = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def onehot_to_seq(onehot: np.ndarray) -> str:
    """
    onehot: (L,4) with channels ordered as A,C,G,U (若你顺序不同要改)
    PAD 行通常全0，会被转成 'N'
    """
    vocab = np.array(["A", "C", "G", "U"])
    s = []
    for row in onehot:
        if np.allclose(row, 0):
            s.append("N")
        else:
            s.append(vocab[int(row.argmax())])
    return "".join(s)


def build_M(seq: str, min_loop: int = 4) -> np.ndarray:
    """
    M[i,j]=1 if canonical+GU and |i-j|>=min_loop, else 0
    """
    seq = seq.upper().replace("T", "U")
    L = len(seq)
    M = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        bi = seq[i]
        if bi == "N":
            continue
        for j in range(i + min_loop, L):
            bj = seq[j]
            if bj == "N":
                continue
            if (bi, bj) in PAIR_OK:
                M[i, j] = 1.0
                M[j, i] = 1.0
    return M


def transform_Y(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    T(Y) = 0.5*(Y+Y^T) ∘ M
    """
    Ysym = 0.5 * (Y + Y.T)
    return Ysym * M


def max_weight_matching_decode(S: np.ndarray, offset: float = 0.0):
    """
    用最大权匹配实现 UFold 的约束(iv) 非重叠配对。
    S: (L,L) constrained score/prob matrix
    offset: only consider edges with score > offset
    return:
      A: (L,L) binary contact map
    """
    import networkx as nx

    L = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(L))

    for i in range(L):
        for j in range(i + 1, L):
            w = float(S[i, j])
            if w > offset:
                G.add_edge(i, j, weight=w)

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)

    A = np.zeros((L, L), dtype=np.int8)
    for u, v in matching:
        i, j = (u, v) if u < v else (v, u)
        A[i, j] = 1
        A[j, i] = 1
    return A