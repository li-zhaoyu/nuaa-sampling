"""NUAA-MU：Mamba–Unfolding 学习多陪集重构器（对应 计算光采样.md §5C/§6）。

faithful 组件：
- 事件 Token 嵌入（含归一化 Δt）；
- Bi-Mamba（ExtBiMamba：前/后向选择性 SSM，显式注入归一化事件间隔，纯 PyTorch 扫描）；
- K 步展开数据一致性解码（用测量矩阵 A=Φ_C 的 matvec/A^H，FISTA 风格），层间学习去噪 + 软阈值。

为 CPU 可训练，训练时常用固定/采样的 R=L*steps；推理时可通过
``forward_chunk`` 接收滚动事件缓冲的可变长度分块。
复数用 torch.complex64。
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 选择性状态空间（Mamba-like，对角 A，内容相关 Δ,B,C，注入归一化 Δt）
# --------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        # 对角、实部为负的稳定参数化：A = -softplus(A_raw)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1)   # -> (B_l, C_l, dt_content)
        self.dt_bias = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x, dt_norm):
        """x:(B,T,d), dt_norm:(B,T) 归一化事件间隔（log1p）。返回 (B,T,d)。"""
        Bsz, T, d = x.shape
        A = -torch.nn.functional.softplus(self.A_log)        # (d, n) 负实，稳定
        proj = self.x_proj(x)                                # (B,T,2n+1)
        Bm, Cm, dt_c = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        # 内容相关步长 × 归一化真实间隔（§6：禁止 ps 原值直代）
        dt = torch.nn.functional.softplus(dt_c.squeeze(-1) + self.dt_bias.mean()) * (1.0 + dt_norm)  # (B,T)
        h = x.new_zeros(Bsz, d, self.d_state)
        ys = []
        for t in range(T):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)       # (B,1,1)
            Abar = torch.exp(dt_t * A.unsqueeze(0))           # (B,d,n)
            Bbar = dt_t * Bm[:, t].unsqueeze(1)               # (B,1,n)
            h = Abar * h + Bbar * x[:, t].unsqueeze(-1)       # (B,d,n)
            y = torch.einsum("bdn,bn->bd", h, Cm[:, t])       # (B,d)
            ys.append(y)
        y = torch.stack(ys, dim=1)                            # (B,T,d)
        return y + x * self.D


class BiMamba(nn.Module):
    """ExtBiMamba：前/后向独立 SSM，投影后相加（参《Mamba in Speech》）。"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.fwd = SelectiveSSM(d_model, d_state)
        self.bwd = SelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x, dt_norm):
        xn = self.norm(x)
        yf = self.fwd(xn, dt_norm)
        yb = torch.flip(self.bwd(torch.flip(xn, [1]), torch.flip(dt_norm, [1])), [1])
        return x + self.out(self.act(yf + yb))


# --------------------------------------------------------------------------
# NUAA-MU 主体：编码器 + K 步展开数据一致性解码
# --------------------------------------------------------------------------
class NUAAMU(nn.Module):
    def __init__(self, d_in: int, nC: int, d_model: int = 64,
                 n_layers: int = 2, d_state: int = 16, K_unroll: int = 4,
                 K_sparse: int = 3, cand_dim: int = 1):
        super().__init__()
        self.nC, self.K = nC, K_unroll
        self.K_sparse = K_sparse
        self.cand_dim = cand_dim
        self.embed = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([BiMamba(d_model, d_state) for _ in range(n_layers)])
        self.ctx_proj = nn.Linear(d_model, d_model)
        # 每个展开步的学习步长 + 去噪 MLP（输入 [ReX, ImX, ctx]）
        self.alpha = nn.Parameter(torch.ones(K_unroll) * 0.5)
        self.thr = nn.Parameter(torch.ones(K_unroll) * 1e-2)
        self.alpha_final = nn.Parameter(torch.tensor(0.3))
        self.prior_beta = nn.Parameter(torch.tensor(2.0))
        self.prior_head = nn.Sequential(
            nn.Linear(3 + d_model + cand_dim, 96), nn.SiLU(), nn.Linear(96, 2))
        self.noise_head = nn.Sequential(
            nn.Linear(2 + d_model, 64), nn.SiLU(), nn.Linear(64, 1))
        self.event_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.burst_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 3))
        self.denoise = nn.ModuleList([
            nn.Sequential(nn.Linear(2 + d_model, 64), nn.SiLU(), nn.Linear(64, 2))
            for _ in range(K_unroll)])

    @staticmethod
    def _soft_threshold(xr, xi, thr):
        mag = torch.sqrt(xr ** 2 + xi ** 2) + 1e-9
        scale = torch.clamp(1.0 - thr / mag, min=0.0)
        return xr * scale, xi * scale

    @staticmethod
    def _snr_gate_from_residual(r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """由残差/观测功率估计 SNR，高 SNR → gate→0（减弱软阈值）。"""
        noise = r.abs().pow(2).mean(dim=-1)
        sig = y.abs().pow(2).mean(dim=-1) + 1e-12
        snr_db = 10.0 * torch.log10(sig / (noise + 1e-12))
        return torch.sigmoid(-(snr_db - 5.0) / 2.5)          # (B,)

    def _refine_support_ls(self, X: torch.Tensor, A: torch.Tensor,
                           Y: torch.Tensor, k: int,
                           prior_score: torch.Tensor | None = None) -> torch.Tensor:
        """完整 K 步先验加权 SOMP + 末步 LS（高 SNR 逼近/超越纯 SOMP）。"""
        Bsz, nC = X.shape
        col_norm = torch.linalg.norm(A, dim=1) + 1e-12          # (B,nC)
        An = A / col_norm.unsqueeze(1)
        prior = prior_score if prior_score is not None else X.abs()
        pmax = prior.amax(dim=1, keepdim=True) + 1e-12
        pw = 1.0 + torch.relu(self.prior_beta) * (prior / pmax)
        resid = Y.clone()
        support = torch.full((Bsz, k), -1, dtype=torch.long, device=X.device)
        for step in range(k):
            corr = torch.abs(torch.bmm(An.conj().transpose(1, 2),
                                       resid.unsqueeze(-1)).squeeze(-1)) * pw
            for s in range(step):
                corr[torch.arange(Bsz), support[:, s]] = -1e9
            j = corr.argmax(dim=1)
            support[:, step] = j
            idx_exp = j.unsqueeze(1).unsqueeze(2).expand(-1, A.shape[1], step + 1)
            sup_cols = support[:, : step + 1].unsqueeze(1).expand(-1, A.shape[1], -1)
            As = torch.gather(A, 2, sup_cols)
            coef = torch.linalg.lstsq(As, Y.unsqueeze(-1)).solution.squeeze(-1)
            resid = Y - torch.bmm(As, coef.unsqueeze(-1)).squeeze(-1)
        idx = support
        idx_exp = idx.unsqueeze(1).expand(-1, A.shape[1], -1)
        As = torch.gather(A, 2, idx_exp)
        coef = torch.linalg.lstsq(As, Y.unsqueeze(-1)).solution.squeeze(-1)
        X_out = torch.zeros(Bsz, nC, dtype=torch.complex64, device=X.device)
        X_out.scatter_(1, idx, coef.to(torch.complex64))
        return X_out

    @staticmethod
    def _null_predicted_jammer(A: torch.Tensor, Y: torch.Tensor,
                               jammer_logits: torch.Tensor,
                               prob_thr: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """用 jammer head 的最高置信列做批量子空间投影；无置信干扰则保持原样。"""
        jam_prob = torch.sigmoid(jammer_logits)
        pmax, idx = jam_prob.max(dim=1)
        active = pmax > prob_thr
        if not bool(active.any()):
            return A, Y
        A_out = A.clone()
        Y_out = Y.clone()
        rows = torch.arange(A.shape[0], device=A.device)
        u = A[rows, :, idx]                                      # (B,R)
        den = (u.conj() * u).sum(dim=1, keepdim=True) + 1e-9
        coeff_y = (u.conj() * Y).sum(dim=1, keepdim=True) / den
        Y_proj = Y - coeff_y * u
        coeff_a = torch.bmm(u.conj().unsqueeze(1), A).squeeze(1) / den
        A_proj = A - u.unsqueeze(2) * coeff_a.unsqueeze(1)
        Y_out[active] = Y_proj[active]
        A_out[active] = A_proj[active]
        return A_out, Y_out

    def forward(self, tok, dt_norm, A, Y, refine: bool = True,
                return_aux: bool = False, cand_feat: torch.Tensor | None = None,
                use_burst: bool = False, jam_prob_thr: float = 0.5):
        """tok:(B,R,d_in) dt_norm:(B,R) A:(B,R,nC) complex Y:(B,R) complex。

        refine=True 时末步硬数据一致性 + 先验加权支撑 LS（高 SNR 超越纯 SOMP）。
        返回 Xhat:(B,nC) complex。
        """
        Bsz = tok.shape[0]
        h = self.embed(tok)
        for layer in self.layers:
            h = layer(h, dt_norm)
        ctx = torch.tanh(self.ctx_proj(h.mean(dim=1)))       # (B,d_model)

        anomaly = tok[..., 3] if tok.shape[-1] >= 8 else torch.zeros_like(dt_norm)
        event_logits = self.event_head(h).squeeze(-1)
        burst_raw = self.burst_head(h)
        burst_logits = burst_raw[..., 0]
        burst_complex = torch.complex(burst_raw[..., 1], burst_raw[..., 2])
        if use_burst:
            burst_prob = torch.sigmoid(burst_logits)
            # Event-domain burst is modeled as a separate component; subtract it
            # before sparse spectral reconstruction, while exposing it as an aux output.
            Y_base = Y - burst_prob.to(Y.dtype) * burst_complex
        else:
            burst_prob = torch.zeros_like(dt_norm)
            burst_complex = torch.zeros_like(Y)
            Y_base = Y
        robust_weight = torch.exp(-0.15 * anomaly).clamp_min(0.02)
        event_weight = (torch.sigmoid(event_logits) * robust_weight).clamp_min(0.02)
        sqrt_w = torch.sqrt(event_weight).to(A.dtype)
        A_w = A * sqrt_w.unsqueeze(-1)
        Y_w = Y_base * sqrt_w

        Ah = A_w.conj().transpose(1, 2)                        # (B,nC,R)
        matched = torch.bmm(Ah, Y_w.unsqueeze(-1)).squeeze(-1)  # (B,nC)
        mf_abs = torch.log1p(matched.abs()).unsqueeze(-1)
        if cand_feat is None:
            cand_feat = torch.zeros(Bsz, self.nC, self.cand_dim,
                                    dtype=tok.dtype, device=tok.device)
        cand_feat = torch.cat([
            matched.real.unsqueeze(-1),
            matched.imag.unsqueeze(-1),
            mf_abs,
            cand_feat,
            ctx.unsqueeze(1).expand(-1, self.nC, -1),
        ], dim=-1)
        prior_logits = self.prior_head(cand_feat)
        support_logits = prior_logits[..., 0]
        jammer_logits = prior_logits[..., 1]
        useful_prior = torch.sigmoid(support_logits) * (1.0 - torch.sigmoid(jammer_logits))
        y_stat = torch.stack([
            torch.log1p(Y_base.abs().pow(2).mean(dim=1)),
            dt_norm.mean(dim=1),
        ], dim=1)
        noise_logvar = self.noise_head(torch.cat([ctx, y_stat], dim=1)).squeeze(-1)

        # Small learned prior initialization lets capacity express support/jammer beliefs
        # before the physics unroll refines the amplitudes.
        X = (0.1 * useful_prior).to(torch.complex64) * matched
        snr_gate = None
        for k in range(self.K):
            r = torch.bmm(A, X.unsqueeze(-1)).squeeze(-1) - Y_base     # (B,R)
            if snr_gate is None:
                snr_gate = self._snr_gate_from_residual(r, Y_base)       # (B,)
            r_w = torch.bmm(A_w, X.unsqueeze(-1)).squeeze(-1) - Y_w
            grad = torch.bmm(Ah, r_w.unsqueeze(-1)).squeeze(-1)        # (B,nC)
            X = X - self.alpha[k] * grad                               # 数据一致性梯度步
            feat = torch.cat([X.real.unsqueeze(-1), X.imag.unsqueeze(-1),
                              ctx.unsqueeze(1).expand(-1, self.nC, -1)], dim=-1)
            d = self.denoise[k](feat)                                  # (B,nC,2)
            xr = X.real + d[..., 0]
            xi = X.imag + d[..., 1]
            thr = torch.relu(self.thr[k]) * snr_gate.unsqueeze(-1)
            xr, xi = self._soft_threshold(xr, xi, thr)
            X = torch.complex(xr, xi)
        if refine:
            A_ref, Y_ref = self._null_predicted_jammer(
                A_w, Y_w, jammer_logits, prob_thr=jam_prob_thr)
            Ah_ref = A_ref.conj().transpose(1, 2)
            r = torch.bmm(A_ref, X.unsqueeze(-1)).squeeze(-1) - Y_ref
            grad = torch.bmm(Ah_ref, r.unsqueeze(-1)).squeeze(-1)
            X = X - torch.relu(self.alpha_final) * grad                # 末步纯 DC，无去噪
            X = self._refine_support_ls(X, A_ref, Y_ref, self.K_sparse, useful_prior)
        if return_aux:
            return X, dict(
                support_logits=support_logits,
                jammer_logits=jammer_logits,
                noise_logvar=noise_logvar,
                useful_prior=useful_prior,
                event_weight=event_weight,
                burst_logits=burst_logits,
                burst_complex=burst_complex,
                burst_prob=burst_prob,
                event_logits=event_logits,
            )
        return X

    def forward_chunk(self, tok, dt_norm, A, Y, state: dict | None = None,
                      refine: bool = True, return_aux: bool = False,
                      cand_feat: torch.Tensor | None = None,
                      use_burst: bool = False):
        """Streaming-friendly wrapper over ``forward``.

        The current CPU implementation reruns the bidirectional encoder on each
        finite observation window. ``state`` carries lightweight
        summaries for the outer streaming controller; a future causal Mamba
        implementation can replace this without changing the call site.
        """
        out = self.forward(
            tok, dt_norm, A, Y, refine=refine, return_aux=return_aux,
            cand_feat=cand_feat, use_burst=use_burst)
        if return_aux:
            xhat, aux = out
            new_state = {
                "n_events": int(tok.shape[1]),
                "support_prior": aux.get("useful_prior").detach(),
                "event_weight": aux.get("event_weight").detach(),
            }
            if state:
                new_state["prev_n_events"] = state.get("n_events", 0)
            return xhat, aux, new_state
        new_state = {"n_events": int(tok.shape[1])}
        if state:
            new_state["prev_n_events"] = state.get("n_events", 0)
        return out, new_state


def si_complex_nmse(Xhat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """尺度不变复 NMSE（按样本对齐复标度），返回标量均值。"""
    num = (Xhat.conj() * X).sum(dim=1)
    den = (Xhat.conj() * Xhat).sum(dim=1) + 1e-9
    alpha = (num / den).unsqueeze(1)
    err = (X - alpha * Xhat).abs().pow(2).sum(dim=1)
    ref = X.abs().pow(2).sum(dim=1) + 1e-12
    return (err / ref).mean()
