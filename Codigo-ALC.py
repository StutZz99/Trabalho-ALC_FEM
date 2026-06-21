import numpy as np
from scipy.sparse import diags
from scipy.linalg import cholesky_banded, cho_solve_banded
from typing import Callable
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})

"""
Montagem das matrizes de massa (M) e rigidez (A) para a discretizacao por Elementos Finitos em malha uniforme 1D, com condicoes de contorno nulas nos extremos.

Referencia: Larson & Bengzon (2013), Secs. 1.5.1, 2.4.1.
"""

def constroi_M(n: int, L: float = 1.0, rho: float = 1.0):

    h = L / n
    N = n - 1
    diag_principal = np.full(N, 2 * rho * h / 3, dtype=float)
    diag_secundaria = np.full(N - 1, rho * h / 6, dtype=float)
    M = diags([diag_secundaria, diag_principal, diag_secundaria], [-1, 0, 1], shape=(N, N), format='csr', dtype=float)

    return M

def constroi_A(n: int, L: float = 1.0):

    h = L / n
    N = n - 1
    diag_principal = np.full(N, 2 / h, dtype=float)
    diag_secundaria = np.full(N - 1, -1 / h, dtype=float)
    A = diags([diag_secundaria, diag_principal, diag_secundaria], [-1, 0, 1], shape=(N, N), format='csr', dtype=float)

    return A

def nos(n: int, L: float = 1.0) -> np.ndarray:

    h = L / n

    return np.linspace(h, L - h, n - 1)

"""
Solver MEF para a equacao da onda elastica em barra 1D:

Referencias: Larson & Bengzon (2013), Secs. 5.7.3 - 5.7.5, Alg. 15.; Trefethen & Bau (1997), Lec. 23.
"""

def solver_onda_CN(n: int, m: int, T: float, E: float, rho: float, u0: Callable[[np.ndarray], np.ndarray], v0: Callable[[np.ndarray], np.ndarray], L: float = 1.0) -> dict:

    c2 = E / rho
    c = np.sqrt(c2)
    h = L / n
    k = T / m

    print(f"Solver MEF da Equacao da Onda 1D (Crank-Nicolson)")
    print(f"Parametros fisicos: L={L}, E={E}, rho={rho}, c={c}")
    print(f"Malha espacial: n={n} elementos, h={h:.4f}, N={n-1} nos interiores")
    print(f"Integracao temporal: m={m} passos, k={k}, T={T}")

    x_nos = nos(n, L)
    M = constroi_M(n, L, rho)
    A = constroi_A(n, L)

    # Condicoes iniciais
    U = u0(x_nos).astype(float)
    V = v0(x_nos).astype(float)
    print(f"Condicao inicial: u0 = {np.max(np.abs(U))}, v0 = {np.max(np.abs(V))}")

    # Fatoracao de Cholesky de A_chapeu
    A_chapeu = M + (c2 * k**2 / 4.0) * A
    ab = np.zeros((2, A_chapeu.shape[0]))
    ab[0, :] = A_chapeu.diagonal(0)
    ab[1, :-1] = A_chapeu.diagonal(-1)
    cho = cholesky_banded(ab, lower=True)

    # Inicializacao
    deslocamentos = [U.copy()]
    tempos = [0.0]

    for l in range(1, m + 1):

        t_atual = l * k
        r_l = (M @ U + k * (M @ V) - (c2 * k**2 / 4.0) * (A @ U))
        U_novo = cho_solve_banded((cho, True), r_l)
        V = (2.0 / k) * (U_novo - U) - V
        U = U_novo
        deslocamentos.append(U.copy())
        tempos.append(t_atual)

    return {'nos': x_nos, 'tempos': np.array(tempos), 'deslocamentos': np.array(deslocamentos), 'h': h, 'k': k, 'c': c}

def solucao_exata(x: np.ndarray, t: float, L: float = 1.0, c: float = 1.0, n_terms: int = 200) -> np.ndarray:

    u = np.zeros_like(x, dtype=float)

    for k in range(1, n_terms + 1):

        Ak = (8.0 / (k * np.pi)**2) * np.sin(k * np.pi / 2.0)
        u += Ak * np.sin(k * np.pi * x / L) * np.cos(k * np.pi * c * t / L)

    return u

"""
Gera as figuras do artigo:
    -> Fig. 1: Estrutura das matrizes M e A para n=12.
    -> Fig. 2: Numero de condicionamento k(M) e k(A) em funcao de h.
    -> Fig. 3: Momentos da solução MEF (Crank-Nicolson)
"""

# Pasta de saida
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'Figuras')
os.makedirs(FIG_DIR, exist_ok=True)

# Parametros
L = 1.0
E = 1.0
rho = 1.0
c = np.sqrt(E / rho)
T = 4.0
n = 80
m = 2000

# Condicao inicial
def u0(x):
    
    return np.where(x <= 0.5, 2.0 * x, 2.0 * (1.0 - x))

def v0(x):
    
    return np.zeros_like(x)

momentos = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Simulacao
resultado = solver_onda_CN(n, m, T, E, rho, u0, v0, L=L)
x_nos = resultado['nos']
deslocamento = resultado['deslocamentos']
k = resultado['k']

# Selecionar frames
frames = {}
for i in momentos:
    idx = int(round(i / k))
    idx = min(idx, len(deslocamento) - 1)
    frames[i] = deslocamento[idx]

# Figura 1
n_vis = 12
M_vis = constroi_M(n_vis, rho=1.0).toarray()
A_vis = constroi_A(n_vis).toarray()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

im1 = ax1.imshow(M_vis, cmap='Blues', aspect='equal')
ax1.set_title(f'Matriz de massa $M$ ($n={n_vis}$)', fontsize=11)
ax1.set_xlabel('coluna $j$', fontsize=10)
ax1.set_ylabel('linha $i$', fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.046)

im2 = ax2.imshow(A_vis, cmap='Reds', aspect='equal')
ax2.set_title(f'Matriz de rigidez $A$ ($n={n_vis}$)', fontsize=11)
ax2.set_xlabel('coluna $j$', fontsize=10)
plt.colorbar(im2, ax=ax2, fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_matrizes.png'), bbox_inches='tight', dpi=150)

# Figura 2
ns = [5, 10, 20, 50, 100, 200]
hs = []
cond_M = []
cond_A = []

for i in ns:
    h_local = L / i
    M = constroi_M(i, L, rho=1.0)
    A = constroi_A(i, L)
    eM = np.linalg.eigvalsh(M.toarray())
    eA = np.linalg.eigvalsh(A.toarray())
    kM = eM[-1] / eM[0]
    kA = eA[-1] / eA[0]
    hs.append(h_local)
    cond_M.append(kM)
    cond_A.append(kA)

hs = np.array(hs)

fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.loglog(hs, cond_M, 'gs-', ms=7, lw=1.8, label=r'$\kappa(M)$')
ax.loglog(hs, cond_A, 'r^-', ms=7, lw=1.8, label=r'$\kappa(A)$')
ref = cond_A[0] * (hs / hs[0])**(-2)
ax.loglog(hs, ref, 'k--', lw=1.2, alpha=0.55, label=r'$O(h^{-2})$')
ax.axhline(3.0, color='green', ls=':', lw=1.0, alpha=0.6)
ax.set_xlabel('h (passo de malha)', fontsize=12)
ax.set_ylabel(r'Número de condicionamento $\kappa$', fontsize=12)
ax.set_title(r'$\kappa(M)$ e $\kappa(A)$ vs. refinamento de malha', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_condicionamento.png'), bbox_inches='tight', dpi=150)

# Figura 3
fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey=True)
axes = axes.flatten()
x_ideal = np.linspace(0.0, L, 500)

for idx, t in enumerate(momentos):
    ax = axes[idx]
    u_calc = frames[t]
    u_ref = solucao_exata(x_ideal, t, L=L, c=c)
    ax.plot(x_ideal, u_ref, 'k--', lw=1.1, alpha=0.75, label='Analítica')
    ax.plot(x_nos, u_calc, 'b-', lw=1.6, label='MEF (CN)')
    ax.set_title(f'$t = {t:.1f}$ s', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('$x$ [m]', fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel('$u_h$ [m]', fontsize=9)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

axes[-1].axis('off')
fig.suptitle(rf'Propagação de onda elástica em barra 1D — MEF (n={n}, m={m})', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_momentos.png'), bbox_inches='tight', dpi=150)