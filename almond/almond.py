#!/usr/bin/env python3
"""NASA almond: geometry, meshing, admissible vector-field perturbation, validity.

Geometry: Woo, Wang, Schuh & Sanders, IEEE Antennas Propag. Mag. 35(1) 1993.
d = 9.936 in = 0.252374 m; t in [-0.41667, 0.58333]; psi in [-pi, pi].

PERTURBATION MODEL (v2).  The deformation is the paper's own admissible class
(Section 3.1): a global smooth vector field on R^3,

    T(x) = x + eps * V(x),      V : R^3 -> R^3,

evaluated at vertex positions.  No surface normals enter.  V is built from
vector-valued random Fourier features with a squared-exponential spectrum, so
it is analytic, its Jacobian DV is available in closed form, and

    eps * sup |DV|_2 < 1   ==>   T injective on any convex set,

which is the injectivity certificate we impose (at 0.5, i.e. with a factor-two
margin) rather than discovering folds after the fact.  The triangle-triangle
test is retained purely as a safety check: under this model a reported
intersection is a BUG, not a sample to discard.

The earlier v1 model displaced each vertex along its own surface normal.  On a
body that tapers to a point at both ends that model folds near the apexes at
amplitudes far below anything interesting, which is a property of the model and
not of the almond; it is kept only in `legacy.py` for the record.

Everything here is pure geometry -- no solver -- so it runs anywhere and its
output (meshes, perturbed vertex sets, validity verdicts) is what the Julia
drivers consume.  Seeds are explicit; a given (seed, case, ell) always
reproduces the same field, and the amplitude levels merely rescale it.
"""
import numpy as np

D = 9.936 * 0.0254                      # 0.252374 m
A1Y, A1Z, T1 = 0.193333, 0.064444, 0.416667
A2Y, A2Z, T2, C2 = 4.83345, 1.61115, 2.08335, 0.96
TLO, THI = -0.41667, 0.58333
HALF_THICK_SEAM = A1Z * D               # z semi-axis at t = 0, 16.26 mm
X_TIP = D * THI                         # sharp (pointed) apex, +x
X_NOSE = D * TLO                        # blunt (rounded) apex, -x


# --------------------------------------------------------------------- geometry
def semi(t):
    """(y, z) semi-axes at station t, in metres. Vectorised."""
    t = np.asarray(t, float)
    s1 = np.sqrt(np.clip(1 - (t / T1) ** 2, 0, None))
    s2 = np.sqrt(np.clip(1 - (t / T2) ** 2, 0, None)) - C2
    a = np.where(t <= 0, A1Y * D * s1, A2Y * D * s2)
    b = np.where(t <= 0, A1Z * D * s1, A2Z * D * s2)
    return np.maximum(a, 0.0), np.maximum(b, 0.0)


def _perimeter(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def mesh(h, ngrade=4000):
    """Structured parametric surface mesh, graded to near-uniform edge length.

    t-stations are placed at equal ARC LENGTH along the profile rather than
    equal t, which is what keeps the blunt end from being under-resolved (the
    semi-axes vary fastest there). Each station carries a ring whose vertex
    count is its own perimeter divided by h, so the azimuthal spacing is also
    ~h everywhere. Both ends close to a single apex vertex.

    Returns (V, F) with F zero-based.
    """
    tg = np.linspace(TLO, THI, ngrade)
    ag, bg = semi(tg)
    pr = np.stack([D * tg, ag, bg], 1)                 # profile in (x, a, b)
    seg = np.linalg.norm(np.diff(pr, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(12, int(np.ceil(arc[-1] / h)))
    ts = np.interp(np.linspace(0, arc[-1], n + 1), arc, tg)

    V, rings = [], []
    for k, t in enumerate(ts):
        a, b = semi(t)
        a = float(a); b = float(b)
        per = float(_perimeter(a, b)) if (a > 0 and b > 0) else 0.0
        if k == 0 or k == len(ts) - 1 or per < 2 * h:
            rings.append([len(V)]); V.append([D * t, 0.0, 0.0]); continue
        m = max(8, int(np.ceil(per / h)))
        ps = np.linspace(0, 2 * np.pi, m, endpoint=False)
        # stagger alternate rings by half a step: kills the long diagonal
        if k % 2: ps = ps + np.pi / m
        rings.append(list(range(len(V), len(V) + m)))
        for p in ps:
            V.append([D * t, a * np.cos(p), b * np.sin(p)])
    V = np.asarray(V)

    F = []
    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        n0, n1 = len(r0), len(r1)
        if n0 == 1:
            for j in range(n1): F.append([r0[0], r1[j], r1[(j + 1) % n1]])
        elif n1 == 1:
            for i in range(n0): F.append([r0[i], r0[(i + 1) % n0], r1[0]])
        else:
            # walk both rings by angle, always emitting the shorter diagonal
            i = j = 0
            while i < n0 or j < n1:
                take0 = (j >= n1) or (i < n0 and (i + 1) / n0 <= (j + 1) / n1)
                if take0:
                    F.append([r0[i % n0], r0[(i + 1) % n0], r1[j % n1]]); i += 1
                else:
                    F.append([r0[i % n0], r1[(j + 1) % n1], r1[j % n1]]); j += 1
    return V, np.asarray(F, int)


def edge_consistency(F):
    """(directed edges, edges lacking an opposite twin, edges repeated).

    A closed, consistently oriented triangulation traverses every edge exactly
    twice, once in each direction. BEAST's `buffachristiansen` asserts this
    (`CompScienceMeshes.isoriented`) before it will build the dual space, so
    it is a hard requirement, not a nicety -- and a mesh can have a perfectly
    positive enclosed volume while still failing it, because the volume is a
    sum that a handful of reversed faces barely moves.
    """
    from collections import Counter
    d = Counter()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            d[(a, b)] += 1
    lone = sum(1 for (a, b), c in d.items() if d.get((b, a), 0) != c)
    dup = sum(1 for _, c in d.items() if c > 1)
    return len(d), lone, dup


def orient_consistently(V, F):
    """Propagate one orientation over the whole surface, then face it outward.

    Breadth-first over face adjacency, flipping any neighbour that traverses
    the shared edge in the same direction -- the same procedure as
    `CompScienceMeshes.orient`, done here so the SHIPPED mesh is already
    correct and Julia has nothing to repair. Afterwards the signed volume
    fixes the global sign.

    Returns (F, signed_volume). Raises if the surface is not closed.
    """
    F = np.array(F, dtype=int, copy=True)
    nf = len(F)
    edge = {}                       # undirected edge -> [face indices]
    for t, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge.setdefault((min(a, b), max(a, b)), []).append(t)
    bad = [k for k, v in edge.items() if len(v) != 2]
    if bad:
        raise ValueError(f"surface is not closed: {len(bad)} edges with "
                         f"{'/'.join(str(len(edge[k])) for k in bad[:5])} faces")

    def directed(t):
        f = F[t]
        return {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}

    seen = np.zeros(nf, bool)
    seen[0] = True
    stack = [0]
    while stack:
        t = stack.pop()
        dt = directed(t)
        f = F[t]
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(a, b), max(a, b))
            for u in edge[k]:
                if u == t or seen[u]:
                    continue
                # consistent iff u traverses the shared edge the other way
                if (a, b) in directed(u):
                    F[u] = F[u][[0, 2, 1]]
                seen[u] = True
                stack.append(u)
    if not seen.all():
        raise ValueError(f"surface is not connected: {(~seen).sum()} faces "
                         f"unreached")

    p, q, r = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = float(np.einsum('ij,ij->i', p, np.cross(q, r)).sum() / 6.0)
    if vol < 0:
        F = F[:, [0, 2, 1]]
        vol = -vol
    n, lone, dup = edge_consistency(F)
    if lone or dup:
        raise ValueError(f"orientation failed: {lone} lone, {dup} repeated")
    return F, vol


def quality(V, F):
    p, q, r = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
    e = np.stack([np.linalg.norm(r - q, axis=1),
                  np.linalg.norm(p - r, axis=1),
                  np.linalg.norm(q - p, axis=1)], 1)
    s = e.sum(1) / 2
    inr = np.divide(ar, s, out=np.zeros_like(ar), where=s > 0)
    asp = np.divide(e.max(1), 2 * inr, out=np.full_like(ar, np.inf), where=inr > 0)
    cl = lambda x: np.clip(x, -1, 1)
    A = np.degrees(np.arccos(cl((e[:, 1]**2 + e[:, 2]**2 - e[:, 0]**2) / (2*e[:, 1]*e[:, 2]))))
    B = np.degrees(np.arccos(cl((e[:, 0]**2 + e[:, 2]**2 - e[:, 1]**2) / (2*e[:, 0]*e[:, 2]))))
    ang = np.minimum(np.minimum(A, B), 180 - A - B)
    return dict(min_area=float(ar.min()), n_degenerate=int((ar <= 0).sum()),
                min_angle=float(ang.min()), max_aspect=float(asp.max()),
                edge_min=float(e.min()), edge_max=float(e.max()))


def vertex_normals(V, F):
    """Area-weighted outward vertex normals; +/-x at the apexes.

    NOT used by the perturbation model any more (v2 displaces along a vector
    field, not along normals). Retained because the incident-field and RCS
    post-processing want an orientation.
    """
    p, q, r = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(q - p, r - p)                    # magnitude = 2*area
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.divide(N, ln, out=np.zeros_like(N), where=ln > 1e-14)
    rad = V.copy(); rad[:, 0] = 0.0
    rl = np.linalg.norm(rad, axis=1, keepdims=True)
    flip = (np.sum(N * np.divide(rad, rl, out=np.zeros_like(rad), where=rl > 1e-12),
                   axis=1) < 0) & (rl[:, 0] > 1e-12)
    N[flip] *= -1
    apex = rl[:, 0] <= 1e-12
    N[apex] = np.where((V[apex, 0] > 0)[:, None], [1.0, 0, 0], [-1.0, 0, 0])
    return N


# -------------------------------------------------- admissible vector field
def smoothstep(s):
    """C^1 Hermite step on [0,1]; returns (chi, dchi/ds)."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s), 6 * s * (1 - s)


class Cutoff:
    """C^1 cutoff chi that vanishes identically within `dead`*d of the tip.

        chi(x) = H( (x_tip - dead*d - x_1) / (ramp*d) ),   H = Hermite step,

    so chi == 0 on the whole cap x_1 >= x_tip - dead*d, rises smoothly over
    the next ramp*d, and is 1 on the rest of the body. `None` (case U) means
    the constant 1 with zero gradient.

    The two lengths do different jobs and must be set separately. `dead` is
    the physics: it is what "the tip does not move" means, and the paper fixes
    it at d/10. `ramp` is the Lipschitz budget: |grad chi| <= 3/(2 ramp*d), so
    a short ramp makes the CUTOFF, rather than the random field, the thing
    that limits the admissible amplitude. With ramp = dead = 1/10 the cutoff
    contributes 59.4 m^-1 against the field's own 25-40 m^-1, i.e. case T
    would be capped at roughly half case U's amplitude for a reason that has
    nothing to do with the tip. `ramp` is therefore chosen so that the two
    cases can share one amplitude ladder.
    """

    def __init__(self, dead=0.10, ramp=0.25):
        self.dead, self.ramp = dead, ramp
        self.x0 = X_TIP - dead * D
        self.L = ramp * D

    def __call__(self, X):
        s = (self.x0 - X[:, 0]) / self.L
        chi, dH = smoothstep(s)
        g = np.zeros_like(X)
        inside = (s > 0) & (s < 1)
        g[inside, 0] = -dH[inside] / self.L          # d chi / d x_1
        return chi, g


class VectorField:
    """Vector-valued random Fourier features, squared-exponential spectrum.

        V(x) = c * sum_k a_k cos(w_k . x + b_k),
        w_k ~ N(0, ell^-2 I),  b_k ~ U[0,2pi),  a_k ~ N(0, I_3),
        c   = sqrt(2/M),

    so E[V_i(x) V_j(y)] = delta_ij exp(-|x-y|^2 / (2 ell^2)): three i.i.d.
    squared-exponential components, isotropic, defined on all of R^3 (no
    parametric seam, no pole artefact).

    Jacobian, exactly:
        dV_i/dx_j = -c * sum_k a_ki w_kj sin(w_k . x + b_k).

    Optionally multiplied by a C^1 cutoff chi (case T); the product rule
    contribution V (x) grad chi(x)^T is included in the Jacobian, so the
    injectivity bound below is a bound for the field that is actually applied.

    `scale` is set by `normalise_on`, which fixes max |V| = 1 over a given
    vertex set. Amplitude levels then multiply this normalised field, so a
    level eps IS the maximum vertex displacement, in metres, for every sample.
    """

    def __init__(self, ell, seed, nfeat=512, cutoff=None):
        rng = np.random.default_rng(seed)
        self.ell, self.seed, self.M = ell, seed, nfeat
        self.W = rng.normal(0.0, 1.0 / ell, size=(nfeat, 3))
        self.b = rng.uniform(0.0, 2 * np.pi, size=nfeat)
        self.A = rng.normal(0.0, 1.0, size=(nfeat, 3))
        self.c = np.sqrt(2.0 / nfeat)
        self.cutoff = cutoff
        self.scale = 1.0

    # -- raw (pre-cutoff, pre-scale) ---------------------------------------
    def _raw(self, X):
        th = X @ self.W.T + self.b
        return self.c * (np.cos(th) @ self.A)

    def _raw_jac(self, X, chunk=4096):
        out = np.empty((len(X), 3, 3))
        for i in range(0, len(X), chunk):
            th = X[i:i + chunk] @ self.W.T + self.b
            s = -self.c * np.sin(th)
            out[i:i + chunk] = np.einsum('nk,ki,kj->nij', s, self.A, self.W,
                                         optimize=True)
        return out

    # -- with cutoff and scale ---------------------------------------------
    def __call__(self, X):
        v = self._raw(X)
        if self.cutoff is not None:
            chi, _ = self.cutoff(X)
            v = v * chi[:, None]
        return self.scale * v

    def jac(self, X):
        J = self._raw_jac(X)
        if self.cutoff is not None:
            v = self._raw(X)
            chi, g = self.cutoff(X)
            J = chi[:, None, None] * J + v[:, :, None] * g[:, None, :]
        return self.scale * J

    def normalise_on(self, V):
        """Fix max |field| = 1 over the vertex set V."""
        self.scale = 1.0
        m = float(np.linalg.norm(self(V), axis=1).max())
        self.scale = 1.0 / m if m > 0 else 1.0
        return m


def sup_jacobian(field, V, pad=None, ngrid=48, refine=3, report=False):
    """sup over the bounding box of |D(field)|_2 (largest singular value).

    Injectivity of T = Id + eps*field needs the sup over a CONVEX set
    containing the surface, so the axis-aligned bounding box of the nominal
    mesh (padded) is the right domain -- not the surface itself. The coarse
    tensor grid is followed by `refine` local re-griddings around the running
    argmax, each shrinking the window by 4x, which is what turns a grid
    maximum into a usable estimate of the supremum.

    Returns (sup_box, sup_surface). The second is reported only for contrast;
    the certificate uses the first.
    """
    lo, hi = V.min(0), V.max(0)
    if pad is None:
        pad = 0.05 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad

    def grid_max(lo, hi, n):
        axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        s = np.linalg.svd(field.jac(G), compute_uv=False)[:, 0]
        i = int(np.argmax(s))
        return float(s[i]), G[i]

    best, xstar = grid_max(lo, hi, ngrid)
    span = (hi - lo) / (ngrid - 1)
    for _ in range(refine):
        lo2 = np.maximum(lo, xstar - span)
        hi2 = np.minimum(hi, xstar + span)
        b2, x2 = grid_max(lo2, hi2, 17)
        if b2 > best: best, xstar = b2, x2
        span = (hi2 - lo2) / 16 * 2
    s_surf = float(np.linalg.svd(field.jac(V), compute_uv=False)[:, 0].max())
    if report:
        return best, s_surf, xstar
    return best, s_surf


def make_field(ell, seed, case, V_nominal, nfeat=512, dead=0.10, ramp=0.25):
    """Build, cut off and normalise the field for one sample."""
    cut = Cutoff(dead, ramp) if case == "T" else None
    fld = VectorField(ell, seed, nfeat=nfeat, cutoff=cut)
    fld.normalise_on(V_nominal)
    return fld


def perturb(V, field, eps):
    """T(x) = x + eps*field(x). Max vertex displacement is exactly eps."""
    disp = eps * field(V)
    return V + disp, np.linalg.norm(disp, axis=1)


# ---------------------------------------------------------- self-intersection
def _bvh(V, F, leaf=24):
    c = V[F].mean(1)
    nodes = []

    def build(ids):
        pts = V[F[ids]].reshape(-1, 3)
        lo, hi = pts.min(0), pts.max(0)
        me = len(nodes)
        nodes.append([lo, hi, None, None, ids if len(ids) <= leaf else None])
        if len(ids) > leaf:
            ax = int(np.argmax(hi - lo))
            o = ids[np.argsort(c[ids, ax])]
            nodes[me][2] = build(o[:len(o) // 2])
            nodes[me][3] = build(o[len(o) // 2:])
        return me

    build(np.arange(len(F)))
    return nodes


def tri_tri(P, Q, tol=1e-9):
    """Separating-axis triangle-triangle overlap. P, Q: (n,3,3) -> bool (n,).

    The 11 candidate axes are the two face normals and the nine edge-edge
    cross products. Each axis is NORMALISED before the interval test, so `tol`
    is a length in metres (1 nm here): triangles that come closer than tol
    without crossing are reported as separated. Testing unnormalised axes --
    as the first version did -- makes the tolerance meaningless, because the
    cross products span several orders of magnitude over a graded mesh.

    An axis whose length is negligible against the edges that generated it is
    skipped rather than used, again relatively: parallel edges produce a near
    zero cross product that carries no information.

    Exactly coplanar triangles are not fully resolved by this axis set. That
    case does not arise for a randomly perturbed smooth surface, and the
    brute-force cross-check in `verify_tri_tri` covers the point.
    """
    ea = [P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]]
    eb = [Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 1], Q[:, 0] - Q[:, 2]]
    la = [np.linalg.norm(u, axis=1) for u in ea]
    lb = [np.linalg.norm(u, axis=1) for u in eb]
    axes = [(np.cross(ea[0], -ea[2]), la[0] * la[2]),
            (np.cross(eb[0], -eb[2]), lb[0] * lb[2])]
    axes += [(np.cross(u, v), lu * lv)
             for u, lu in zip(ea, la) for v, lv in zip(eb, lb)]
    sep = np.zeros(len(P), bool)
    for n, ref in axes:
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-6 * np.maximum(ref, 1e-300)
        if not ok.any():
            continue
        nu = n / np.where(ln > 0, ln, 1.0)[:, None]
        pa = np.einsum('nij,nj->ni', P, nu)
        pb = np.einsum('nij,nj->ni', Q, nu)
        sep |= ok & ((pa.min(1) > pb.max(1) + tol) | (pb.min(1) > pa.max(1) + tol))
    return ~sep


def candidate_pairs(V, F):
    """Broad phase: every face pair whose bounding spheres overlap, minus the
    pairs that share a vertex.

    A KD-tree on triangle centroids with a global search radius 2*max(r) is
    both simpler and an order of magnitude faster than the recursive BVH it
    replaces (0.06 s vs 0.88 s on the lambda/12 mesh), and -- unlike a
    traversal with an early exit -- it costs the same whether the geometry is
    valid or not, so a whole dry run is affordable.

    Pairs sharing at least one vertex are removed BY INDEX, before any
    geometry is evaluated: two faces of a closed mesh that share an edge or a
    vertex always meet along that shared feature, so they carry no
    information. A genuine fold spans several elements and is always caught
    through some non-adjacent pair of the same fold.
    """
    from scipy.spatial import cKDTree
    T = V[F]
    c = T.mean(1)
    r = np.linalg.norm(T - c[:, None, :], axis=2).max(1)
    tree = cKDTree(c)
    pairs = np.asarray(sorted(tree.query_pairs(2.0 * r.max())), dtype=int)
    if not len(pairs):
        return pairs
    I, J = pairs[:, 0], pairs[:, 1]
    keep = np.linalg.norm(c[I] - c[J], axis=1) <= r[I] + r[J]
    I, J = I[keep], J[keep]
    shared = (F[I][:, :, None] == F[J][:, None, :]).any(2).any(1)
    return np.stack([I[~shared], J[~shared]], 1)


def self_intersects(V, F, max_report=8, tol=1e-9, chunk=200000):
    """Face pairs that genuinely overlap. Returns at most `max_report`."""
    P = candidate_pairs(V, F)
    hits = []
    for s in range(0, len(P), chunk):
        a, b = P[s:s + chunk, 0], P[s:s + chunk, 1]
        hit = tri_tri(V[F[a]], V[F[b]], tol=tol)
        for x, y in zip(a[hit], b[hit]):
            hits.append((int(x), int(y)))
            if len(hits) >= max_report:
                return hits
    return hits


def self_intersects_bvh(V, F, max_report=8, leaf=24, tol=1e-9):
    """Original BVH traversal, kept as an independent cross-check.

    Two triangles of a closed surface mesh that share an edge or a single
    vertex always "intersect" along that shared feature; those pairs carry no
    information and are excluded by INDEX, before any geometry is evaluated,
    so a fold cannot hide behind a shared vertex either -- a genuinely folded
    neighbour pair also overlaps away from the shared feature, but it will be
    caught through some other, non-adjacent pair of the fold. The residual
    blind spot is a fold entirely contained in one vertex star; at these
    amplitudes it is smaller than the tip cap's own elements.

    Returns a list of (i, j) face indices, at most `max_report`.
    """
    nodes = _bvh(V, F, leaf=leaf)
    tris = V[F]
    hits = []

    def overlap(i, j):
        return bool(np.all(nodes[i][0] <= nodes[j][1] + tol) and
                    np.all(nodes[j][0] <= nodes[i][1] + tol))

    def leafpair(ia, ib):
        a = np.asarray(nodes[ia][4]); b = np.asarray(nodes[ib][4])
        A_, B_ = np.meshgrid(a, b, indexing="ij")
        A_, B_ = A_.ravel(), B_.ravel()
        keep = A_ < B_                        # each unordered pair once
        A_, B_ = A_[keep], B_[keep]
        if not len(A_):
            return
        fa, fb = F[A_], F[B_]
        shared = (fa[:, :, None] == fb[:, None, :]).any(2).any(1)
        A_, B_ = A_[~shared], B_[~shared]
        if not len(A_):
            return
        hit = tri_tri(tris[A_], tris[B_], tol=tol)
        for a_, b_ in zip(A_[hit], B_[hit]):
            hits.append((int(a_), int(b_)))
            if len(hits) >= max_report:
                return

    def rec(i, j):
        if len(hits) >= max_report or not overlap(i, j):
            return
        li, lj = nodes[i][4] is not None, nodes[j][4] is not None
        if li and lj:
            leafpair(i, j)
        elif li:
            rec(i, nodes[j][2]); rec(i, nodes[j][3])
        elif lj:
            rec(nodes[i][2], j); rec(nodes[i][3], j)
        else:
            rec(nodes[i][2], nodes[j][2]); rec(nodes[i][2], nodes[j][3])
            rec(nodes[i][3], nodes[j][2]); rec(nodes[i][3], nodes[j][3])

    rec(0, 0)
    return hits


def verify_tri_tri(V, F, max_report=8, tol=1e-9):
    """Brute-force O(n^2) reference for `self_intersects`. Small meshes only."""
    n = len(F)
    I, J = np.triu_indices(n, k=1)
    shared = (F[I][:, :, None] == F[J][:, None, :]).any(2).any(1)
    I, J = I[~shared], J[~shared]
    hits = []
    for s in range(0, len(I), 200000):
        a, b = I[s:s + 200000], J[s:s + 200000]
        hit = tri_tri(V[F[a]], V[F[b]], tol=tol)
        for x, y in zip(a[hit], b[hit]):
            hits.append((int(x), int(y)))
            if len(hits) >= max_report:
                return hits
    return hits
