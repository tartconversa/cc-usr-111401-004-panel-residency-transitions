"""连续时间多状态核心计算（仅用标准库）。

模型为分段恒定强度的连续时间马尔可夫链。观察是面板式的：离散
时点上的允许状态集合（访谈中的“无法确定”即多状态集合，不是新
状态）。相邻观察之间的转移概率由生成矩阵的矩阵指数给出，对所
有允许的中间路径求和，而不是把相邻状态差当成一次精确转移。
首访之前不计暴露，末访之后不补写；每人条件于首个观察入组。
"""
import bisect
import math
import random

LOG_LOWER = math.log(1e-9)  # 对数强度下界（每天），触及视为边界解
LOG_UPPER = math.log(20.0)  # 对数强度上界（每天），触及视为边界解


def mat_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(A, B):
    rows, inner, cols = len(A), len(B), len(B[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        Ai, Oi = A[i], out[i]
        for k in range(inner):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(cols):
                Oi[j] += a * Bk[j]
    return out


def mat_scale(A, c):
    return [[x * c for x in row] for row in A]


def mat_norm_inf(A):
    return max(sum(abs(x) for x in row) for row in A)


def mat_expm(A):
    """矩阵指数：缩放平方加泰勒展开，适用于生成矩阵。"""
    n = len(A)
    norm = mat_norm_inf(A)
    if norm == 0.0:
        return mat_identity(n)
    shift = max(0, math.ceil(math.log2(norm / 0.5)))
    B = mat_scale(A, 1.0 / (1 << shift))
    result = mat_identity(n)
    term = mat_identity(n)
    for k in range(1, 400):
        term = mat_scale(mat_mul(term, B), 1.0 / k)
        result = [[result[i][j] + term[i][j] for j in range(n)] for i in range(n)]
        if mat_norm_inf(term) <= 1e-17 * mat_norm_inf(result):
            break
    for _ in range(shift):
        result = mat_mul(result, result)
    return result


def mat_vec(vector, M):
    n, m = len(M), len(M[0])
    out = [0.0] * m
    for i in range(n):
        v = vector[i]
        if v == 0.0:
            continue
        row = M[i]
        for j in range(m):
            out[j] += v * row[j]
    return out


def mat_inverse(A):
    """高斯-约当求逆；接近奇异时返回None。"""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        d = M[col][col]
        M[col] = [x / d for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                M[r] = [x - factor * y for x, y in zip(M[r], M[col])]
    return [row[n:] for row in M]


def sym_eigenvalues(A, sweeps=200, tol=1e-14):
    """雅可比法求对称矩阵特征值，返回升序列表。"""
    n = len(A)
    a = [list(row) for row in A]
    for _ in range(sweeps):
        off = max((abs(a[i][j]) for i in range(n) for j in range(i + 1, n)), default=0.0)
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


class PanelModel:
    """分段恒定强度生成矩阵与面板前向似然。"""

    def __init__(self, n_states, edges, cutpoints):
        if not edges:
            raise ValueError('允许转移集合为空')
        self.n_states = n_states
        self.edges = list(edges)
        self.cutpoints = [float(c) for c in cutpoints]
        self.n_segments = len(self.cutpoints)
        self.params = [(seg, edge)
                       for seg in range(self.n_segments)
                       for edge in range(len(self.edges))]
        self.index = {key: pos for pos, key in enumerate(self.params)}
        self.n_params = len(self.params)

    def segment_of(self, day):
        return min(max(bisect.bisect_right(self.cutpoints, day) - 1, 0), self.n_segments - 1)

    def segment_bounds(self, seg):
        start = self.cutpoints[seg] if seg > 0 else -math.inf
        end = self.cutpoints[seg + 1] if seg + 1 < self.n_segments else math.inf
        return start, end

    def generator(self, theta, seg):
        n = self.n_states
        Q = [[0.0] * n for _ in range(n)]
        for edge, (i, j) in enumerate(self.edges):
            Q[i][j] = math.exp(theta[self.index[(seg, edge)]])
        for i in range(n):
            Q[i][i] = -sum(Q[i])
        return Q

    def transition(self, theta, start, end, cache=None):
        """[start, end) 上的转移矩阵：跨分段时按时间序连乘各段矩阵指数，
        等价于对该区间内所有允许的中间路径求和。"""
        if end < start:
            raise ValueError('转移区间终点早于起点')
        if end == start:
            return mat_identity(self.n_states)
        if cache is None:
            cache = {}
        P = None
        for seg in range(self.segment_of(start), self.n_segments):
            lo, hi = self.segment_bounds(seg)
            lo = max(start, lo)
            hi = min(end, hi)
            if hi <= lo:
                continue
            length = hi - lo
            key = (seg, length)
            if key not in cache:
                cache[key] = mat_expm(mat_scale(self.generator(theta, seg), length))
            P = cache[key] if P is None else mat_mul(P, cache[key])
            if hi >= end:
                break
        return P if P is not None else mat_identity(self.n_states)

    def loglik(self, theta, histories, record=None):
        """面板似然：每人条件于首个观察，逐次观察前向递推并对允许集合求和。

        histories为[(day, 允许状态指标集合), ...]的列表（每人至少两次观察）。
        record非空时追加每一跨度的对数似然贡献与过滤后的状态分布。
        """
        cache = {}
        total = 0.0
        n = self.n_states
        for person_index, observations in enumerate(histories):
            _, first_states = observations[0]
            alpha = [1.0 if s in first_states else 0.0 for s in range(n)]
            for step in range(1, len(observations)):
                start = observations[step - 1][0]
                end, target = observations[step]
                P = self.transition(theta, start, end, cache)
                nxt = [0.0] * n
                for s in range(n):
                    a = alpha[s]
                    if a == 0.0:
                        continue
                    row = P[s]
                    for t in target:
                        nxt[t] += a * row[t]
                norm = sum(nxt)
                if norm <= 0.0:
                    return float('-inf')
                log_norm = math.log(norm)
                total += log_norm
                alpha = [x / norm for x in nxt]
                if record is not None:
                    record.append({'person_index': person_index, 'start': start, 'end': end,
                                   'log_likelihood': log_norm, 'filtered': list(alpha)})
        return total


def nelder_mead(objective, x0, step=0.5, max_iter=2000, tol=1e-8):
    """Nelder-Mead 单纯形优化。返回(最优点, 最优值, 是否收敛, 目标评估次数)。"""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += step
        simplex.append(point)
    values = [objective(p) for p in simplex]
    evaluations = n + 1
    converged = False
    iterations = 0
    while iterations < max_iter:
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        diameter = max(abs(simplex[i][k] - simplex[0][k])
                       for i in range(1, n + 1) for k in range(n))
        if abs(values[-1] - values[0]) <= tol * (1.0 + abs(values[0])) and diameter <= 1e-6:
            converged = True
            break
        iterations += 1
        centroid = [sum(simplex[i][k] for i in range(n)) / n for k in range(n)]
        worst = simplex[-1]
        reflected = [centroid[k] + (centroid[k] - worst[k]) for k in range(n)]
        fr = objective(reflected)
        evaluations += 1
        if fr < values[0]:
            expanded = [centroid[k] + 2.0 * (reflected[k] - centroid[k]) for k in range(n)]
            fe = objective(expanded)
            evaluations += 1
            if fe < fr:
                simplex[-1], values[-1] = expanded, fe
            else:
                simplex[-1], values[-1] = reflected, fr
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            if fr < values[-1]:
                contracted = [centroid[k] + 0.5 * (reflected[k] - centroid[k]) for k in range(n)]
            else:
                contracted = [centroid[k] - 0.5 * (centroid[k] - worst[k]) for k in range(n)]
            fc = objective(contracted)
            evaluations += 1
            if fc < min(fr, values[-1]):
                simplex[-1], values[-1] = contracted, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k])
                                  for k in range(n)]
                    values[i] = objective(simplex[i])
                    evaluations += 1
    order = sorted(range(n + 1), key=lambda i: values[i])
    return simplex[order[0]], values[order[0]], converged, evaluations


def clamped(theta):
    """把对数强度限制在数值安全范围，防止exp溢出。"""
    return [min(max(t, LOG_LOWER - 10.0), LOG_UPPER + 10.0) for t in theta]


def negative_loglik(model, histories, theta):
    """无惩罚的负对数似然，用于拟合目标与信息矩阵。"""
    ll = model.loglik(clamped(theta), histories)
    if ll == float('-inf'):
        return 1e100
    return -ll


def make_objective(model, histories):
    """带盒约束软惩罚的目标函数，使边界解可被识别而不是漂向无穷。"""
    def objective(theta):
        penalty = 0.0
        for t in theta:
            if t < LOG_LOWER:
                penalty += (LOG_LOWER - t) ** 2
            elif t > LOG_UPPER:
                penalty += (t - LOG_UPPER) ** 2
        return negative_loglik(model, histories, theta) + 1e4 * penalty
    return objective


def fit(model, histories, theta0, max_iter=2000, restarts=3, seed=20261114):
    """多起点Nelder-Mead拟合。收敛标志如实上报，不把未收敛当作精确估计。"""
    objective = make_objective(model, histories)
    rng = random.Random(seed)
    starts = [list(theta0)]
    for _ in range(max(1, restarts) - 1):
        starts.append([t + rng.uniform(-1.5, 1.5) for t in theta0])
    best = None
    evaluations = 0
    for start in starts:
        x, fx, converged, evals = nelder_mead(objective, start, max_iter=max_iter)
        evaluations += evals
        if best is None or fx < best[1]:
            best = (x, fx, converged)
    x, fx, converged = best
    return {'theta': x, 'objective': fx, 'converged': converged,
            'evaluations': evaluations, 'restarts': len(starts),
            'failed': fx >= 1e90}


def gradient(f, x, rel=1e-5):
    """中心差分梯度。"""
    out = []
    for i in range(len(x)):
        h = rel * (1.0 + abs(x[i]))
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        out.append((f(xp) - f(xm)) / (2.0 * h))
    return out


def hessian(f, x, rel=1e-4):
    """中心差分Hessian（对称），用于观察信息矩阵。"""
    n = len(x)
    h = [rel * (1.0 + abs(xi)) for xi in x]
    f0 = f(x)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xp, xm = list(x), list(x)
        xp[i] += h[i]
        xm[i] -= h[i]
        H[i][i] = (f(xp) - 2.0 * f0 + f(xm)) / (h[i] * h[i])
    for i in range(n):
        for j in range(i + 1, n):
            xpp, xpm, xmp, xmm = list(x), list(x), list(x), list(x)
            xpp[i] += h[i]; xpp[j] += h[j]
            xpm[i] += h[i]; xpm[j] -= h[j]
            xmp[i] -= h[i]; xmp[j] += h[j]
            xmm[i] -= h[i]; xmm[j] -= h[j]
            value = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
            H[i][j] = H[j][i] = value
    return H
