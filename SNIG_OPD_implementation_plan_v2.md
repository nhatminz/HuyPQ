# Implementation plan: SNIG-OPD (v2 — đã review với repo `nhatminz/BellmanOPD`)

## 0. Các vấn đề tìm thấy ở v1 và cách sửa

Review dựa trên code hiện tại: `selectors/pgt_selector.py`, `selectors/cmt_selector.py`, `selectors/rac_selector.py`, `trainer.py`, `opd_core.py`, `diagnostics.py`, `selector_logging.py`, `tensorboard_logging.py`, `resume.py`, `configs/qwen3_b200_base.yaml`.

| # | Mức độ | Vấn đề ở v1 | Sửa ở v2 |
|---|---|---|---|
| P1 | **Nghiêm trọng** | Gibbs với nhiệt độ cố định `τ=1` trên raw `S` không bất biến theo scale. `g_t = Var_p[r]` có đơn vị nats² và thường nằm trong [0, 10]; trên ~10⁵ token global, `softmax(S/1)` sụp về vài token (mô phỏng gain lognormal: ESS ≈ 5·10⁻⁶·N, KL ≈ 12 nats). Scale của `g` còn trôi theo training nên cùng một `τ` cho allocation khác nhau ở mỗi step. CMT đã tránh đúng lỗi này bằng `kl_constrained_allocation`. | Dùng lại `kl_constrained_allocation(S, ε)` — nghiệm vẫn là Gibbs `ν ∝ exp(βS)`, nhưng `β` được giải mỗi step để `KL(ν‖uniform)=ε`. `τ = 1/β` trở thành output được log, không phải hyperparameter. Config: `snig_allocation_kl`. |
| P2 | **Nghiêm trọng (claim sai)** | "w có mean 1 nên tổng supervision mass bằng pure OPD" và "global Gibbs allocation" không đúng với trainer. `_train_opd_step` chia loss cho `global_weight_mass` **của từng PPO mini-batch** (`trainer.py` ~L1624–1766). Với config base (64 trajectories, `ppo_mini_batch_size=16`) mỗi rollout có 4 optimizer steps, mỗi step là weighted mean riêng: allocation giữa các mini-batch bị renormalize mất, chỉ allocation *bên trong* mini-batch được giữ. `mean(w)=1` vì vậy không ảnh hưởng loss. | Giữ nguyên normalization của CMT (để so sánh công bằng), nhưng sửa claim: Gibbs weights là global, còn supervision mass được chuẩn hoá theo từng PPO mini-batch. Log `global_weight_mass` từng mini-batch (đã có trong `minibatches`) để thấy hệ số renormalize. Không thêm chế độ normalization mới trong v1. |
| P3 | **Nghiêm trọng (khoa học)** | Scale của hai số hạng trong `S_t = g_t + λu_t^succ` lệch nhau cỡ `O(T)`. Mẫu số `M_t(M_t+R_t) ≈ T²(1+ḡ)` khi `γ=1, c≈1`, còn tử `R_{t+1}-g_tM_{t+1} ≈ T·Δg`, nên `u ≈ Δg / (T(1+ḡ))`. Ví dụ T=4000, gain nhảy từ 0.5 lên 2.0 ở giữa: `u ≈ 1.3·10⁻⁴` so với `g=0.5`. Với `λ=1` và `max_new_tokens=7168`, SNIG gần như trùng "PGT + Gibbs allocation"; mọi khác biệt so với PGT/CMT sẽ đến từ allocator, không phải successor term. | (a) Thêm arm bắt buộc `snig` với `λ=0` (cùng allocator) vào comparison, gọi là `PGT-Gibbs`. (b) Log `successor_share = mean|λu| / mean|g|` mỗi step. (c) Trước long run, chạy step-1 offline và chọn `λ` sao cho `successor_share` nằm trong khoảng mong muốn; ghi quyết định vào `RESEARCH_DECISION`. Không đổi định nghĩa toán học trong v1. |
| P4 | Trung bình | Công thức tử số dạng tích `M_t R_{t+1} - R_t M_{t+1}` trừ hai số cỡ `gT²`, và `bellman_parallel_scan` luôn cast về float32. Test "constant-gain cho u=0" với tolerance chặt sẽ fail vì sai số tích luỹ trong R (float32, T=7168: residual ~0.15 trong `R-gM`). | Dùng đồng nhất thức (đã kiểm tra): `M_t R_{t+1} − R_t M_{t+1} = R_{t+1} − g_t M_{t+1}`, tức đúng `successor_excess` của CMT. Test dùng tolerance tương đối (xem §4). |
| P5 | Nhỏ | `eps` cho mẫu số là thừa: trên valid positions `M_t ≥ 1` và `R_t ≥ 0`, nên `M_t(M_t+R_t) ≥ 1`. Đưa `snig_eps` vào resume keys chỉ tạo mismatch giả. | Bỏ `snig_eps` khỏi config/resume. Mask invalid positions về 0 trước khi chia. Dùng `log1p`. |
| P6 | Nhỏ | Khi `δ_t ≠ 0` thì `p°(y_t) < q°(y_t)`, nên `c_t = 1`. Mẫu số của `u` luôn được đánh giá ở nhánh accepted; v1 không nêu điều này. | Ghi vào docstring và thêm test `δ≠0 ⇒ c=1`. |
| P7 | Trung bình | Scalar diagnostics (`allocation_kl`, `τ`) sẽ bị rơi mất: `selector_summary` chỉ tóm tắt tensor token-shaped. Kiểm tra repo: `allocation_kl_achieved` / `allocation_inverse_temperature` của CMT được set nhưng **không bao giờ** tới metrics.json/CSV/TensorBoard. | Truyền scalar tường minh vào `train_metrics["selector"]`, cột CSV, tag TensorBoard, và payload `token_score_stats`. |
| P8 | Trung bình | Thiếu touchpoints trong trainer/logging. `TokenScoreStatsLogger.__init__` và `selector_summary` sẽ **raise ValueError** với method lạ, nên run crash ngay khi khởi tạo. Ngược lại, `SelectedTokenLogger` bị disable với soft-weight methods, nên mục "selected-token fields" ở v1 là việc không cần. Resume keys là tuple tường minh, không hỗ trợ wildcard `selector.snig_*`. | Liệt kê đầy đủ ở §3.2–3.4. |
| P9 | Nhỏ | `snig_full_vocab_diagnostics` có trong config nhưng không được nối: `compute_full_vocab_metrics` chỉ bật cho `method == "cmt"`. | Bỏ khỏi v1. |
| P10 | Test | (i) Finite-difference test phải định nghĩa trên functional kỳ vọng với successor values tất định theo từng action, không phải trên one-sample score. (ii) Rollout hash của SNIG và OPD chỉ trùng ở rollout đầu tiên, cùng checkpoint và seed. (iii) Các test theo `τ` phải đổi sang `ε`. | Viết lại §4. |
| P11 | Ghi chú | Response bị cắt ở `max_new_tokens` được coi là terminal (`R_{T+1}=M_{T+1}=0`), giống CMT. | Ghi trong docstring; `response_clip_ratio` đã được log. |

---

## 1. Phạm vi

Thêm method mới với slug `snig`. Giữ nguyên `cmt` làm baseline để không làm hỏng checkpoint/log cũ.

Giữ nguyên student rollout, joint student–teacher scoring, Top-K union, OPD loss, và normalization theo từng PPO mini-batch.

SNIG chỉ thay selector score. Allocator dùng lại `kl_constrained_allocation` của CMT. Không thêm model forward pass mới.

Luồng:

```
rollout → PGT local gain/support → shared-continuation recursion → SNIG score
        → global KL-constrained Gibbs weights → weighted OPD loss (per-PPO-minibatch mean)
```

## 2. Mathematical contract

Tại mỗi prefix hợp lệ, dùng lại output của `PGTSelector` trên `U = TopK(student) ∪ TopK(teacher)`:

- `p_U, q_U`: distribution đã conditionalize trên U; `m_p, m_q`: raw mass trên U.
- `p° = m_p p_U`, `q° = m_q q_U`; `r(a) = log q_U(a) − log p_U(a)`; `r̄ = E_{p_U}[r]`.
- Local gain: `g_t = Var_{p_U}[r]` (clamp ≥ 0 như CMT).

Với `y_t` sample từ full student policy (tempered theo `rollout.temperature`, `top_p=1`):

```
c_t = 1[y_t ∈ U] · min(1, q°(y_t)/p°(y_t))
δ_t = 1[y_t ∈ U, p°(y_t) < q°(y_t)] · (r(y_t) − r̄)
```

Tại `p° = q°` derivative bằng 0 (convention của CMT). Hệ quả: `δ_t ≠ 0 ⇒ c_t = 1`.

Suffix recurrences (bằng `bellman_parallel_scan`, padding là hard boundary):

```
R_t = g_t + γ c_t R_{t+1}
M_t = 1   + γ c_t M_{t+1}
Φ_t = log1p(R_t / M_t)
```

Khi perturb distribution tại t (giữ `g_t` và descendant values cố định, path `p_ε ∝ p_U exp(ε r)` với `m_p` cố định), successor utility là:

```
u_t^succ = γ δ_t · (R_{t+1} − g_t M_{t+1}) / [M_t (M_t + R_t)]
S_t      = g_t + λ u_t^succ
```

Tử số dùng dạng rút gọn, tương đương đại số với `M_t R_{t+1} − R_t M_{t+1}` và chính là `successor_excess` của CMT. Mẫu số ≥ 1 trên valid positions nên không cần eps. Ở terminal, `R_{t+1}=M_{t+1}=0` nên `u=0`. Suffix có gain hằng số cho `R = gM` và `u = 0` với mọi `c`.

**Quan hệ với CMT (ghi vào docstring).** `u_t^succ = sequential_gain_CMT / [M_t(M_t+R_t)]` khi cùng `γ, λ`. SNIG là CMT với successor term được chuẩn hoá theo suffix mass, đổi length bias `O(T)` lấy scale `O(1/T)` (xem P3).

**Estimator.** `c_t`, `δ_t` là one-sample estimator dưới full student sampling (`top_p=1`). `R_t, M_t, u_t^succ` là self-normalized plug-in estimates; không claim unbiased.

### KL-constrained Gibbs allocation

Trên toàn bộ N valid token của global rollout batch:

```
w, β, KL_achieved = kl_constrained_allocation(S, ε)     # w = N·ν, ν ∝ exp(β S)
```

Tính chất: `w > 0` (trừ trường hợp `ε ≥ log N` thì dồn vào argmax), `mean(w) = 1`, bất biến khi cộng hằng số hoặc nhân `S` với số dương. `β` và `KL_achieved` là outputs.

Log mỗi step:

- `allocation_kl_epsilon = ε`
- `allocation_kl_achieved = Σ ν log(Nν)`
- `allocation_inverse_temperature = β`; `allocation_temperature = 1/β` (`inf` khi β=0, 0 khi β=inf; ghi JSON bằng null)
- `effective_sample_size = (Σw)² / Σw²` (đã có sẵn trong `selector_summary`)
- `successor_share = mean|λu| / max(mean|g|, 1e-12)`

**Ghi chú normalization (sửa claim v1).** Weights được tính global, nhưng `_train_opd_step` chuẩn hoá loss theo `global_weight_mass` của từng PPO mini-batch. SNIG vì vậy chỉ đổi phân bổ *trong* mỗi mini-batch; tổng supervision mỗi optimizer step giống pure OPD. Đây là cùng protocol với CMT.

## 3. Thay đổi code

### 3.1 Core selector — `b200_experiment/selectors/snig_selector.py`

`SNIGSelector(gamma: float = 1.0, successor_lambda: float = 1.0)`.
Validate `0 ≤ γ ≤ 1` và `λ ≥ 0`. Không có `eps`.

`compute_scores(pgt_support: PGTOutput, sampled_token_ids, valid_mask) -> PGTOutput`, chạy trong `@torch.no_grad()`:

1. Tính `g, original_p/q, in_support, sampled_cond_r, sampled_r, mean_r, transition_weight (=c), marginal_flux (=δ)` **y hệt CMT**. Nên tách phần này thành helper dùng chung `_shared_kernel_samples(pgt_support, sampled_ids, valid)` trong `cmt_selector.py`. CMT gọi lại helper này, và test CMT hiện có phải pass không đổi.
2. `R, M, _ = bellman_parallel_scan(g, c, valid, gamma)`; shift sang `R_next, M_next` bằng mask `valid[:, 1:]` như CMT.
3. `excess_next = R_next − g·M_next`
4. `denom = where(valid, M·(M+R), 1)`
5. `u = where(valid, γ·δ·excess_next / denom, 0)`
6. `Phi = where(valid, log1p(R / M.clamp_min(1)), 0)`
7. `score = where(valid, g + λu, 0)`

Assert mọi tensor detached và finite trên valid. Trả `PGTOutput(score, diagnostics, <candidate support của PGT không đổi>)`.

Diagnostics (token-shaped, 0 ngoài valid): `gain, s_PGT, support_common_mass, transition_weight, kernel_derivative (=δ), teacher_deficit, R, M, R_next, M_next, successor_excess, Phi, successor_utility, score, s_SNIG` và các field support của PGT (`student_union_mass, teacher_union_mass, teacher_tail_mass, support_width`).
Scalars: `gamma, successor_lambda, score_definition="snig_pgt_gain_plus_normalized_log_potential_successor_derivative"`.

Cập nhật `selectors/__init__.py` để export `SNIGSelector`.

### 3.2 Trainer — `b200_experiment/trainer.py`

Tất cả vị trí đang hard-code method (tham chiếu dòng theo commit hiện tại):

- `METHOD_LABELS` (~L137): thêm `"snig": "SNIG-OPD"`.
- Method whitelist và error message (~L2738).
- Block kiểm tra `top_p`/temperature (~L2745): áp dụng cho `method in {"cmt", "snig"}`, dùng cùng warning.
- `metadata["distributed"]` (~L3028): thêm `"global_snig_kl_allocation": method == "snig"`.
- Khởi tạo `snig_selector = SNIGSelector(gamma=selector_cfg.get("snig_gamma", 1.0), successor_lambda=selector_cfg.get("snig_successor_lambda", 1.0))`.
- `use_joint_scoring` (~L3310) và nhánh tính `pgt_raw` (~L3398): `{"pgt", "cmt", "snig"}`. `compute_full_vocab_metrics` giữ nguyên chỉ cho `cmt`.
- Tính `snig_raw` ngay sau `pgt_raw`, không chạy CMT selector. Thêm timing `snig_score_time`.
- Hàm mới `_globalize_snig_output(local, valid_mask, epsilon, lam, distributed)`:
  - gather tất cả key token-shaped ở §3.1 bằng `_gather_selector_diagnostics`;
  - `w, β, kl = kl_constrained_allocation(gathered["s_SNIG"], epsilon)`;
  - scatter `w[start:end]` về local;
  - tính `successor_share` trên vector global;
  - trả `SelectorOutput(w, diagnostics)`, `gathered` (có `w`), `start`, `end`, và thêm dict scalar `allocation`.
- Nhánh `elif method == "snig"` trong chuỗi selector (~L3630), tương tự CMT. `bellman_scan_time = snig_score_time`.
- Chọn token: `snig` rơi vào nhánh soft-weight (`selected = valid`, `token_allocation = primary.scores`), không cần sửa vì điều kiện là `method in {"ta","pgt"}`.
- Metadata train (~L4000–4025):
  - `all_response_tokens_supervised`: thêm `snig`;
  - `token_allocation_policy["snig"] = "kl_constrained_global_successor_normalized_information_geometric_gibbs"`;
  - `opd_candidate_support` và `opd_support_geometry`: `{"pgt","cmt","snig"}`.
- Timing metrics (~L4099): `"snig_score_time_sec"`.
- **Scalar allocation (P7):** gán `train_metrics["selector"].update(allocation)` sau `selector_summary(...)` (~L4146), với các key `allocation_kl_epsilon, allocation_kl_achieved, allocation_inverse_temperature, allocation_temperature, successor_share`.
- `_append_train_metrics_csv` (~L605 và ~L683, **cả hai** tuple score): thêm `kernel_derivative, Phi, successor_utility, s_SNIG, R_next, M_next`; thêm 5 scalar allocation vào tuple `common`.
- Cleanup `del` (~L4170): thêm `snig_raw, global_snig_diagnostics`.

Không sửa `opd_core.py`.

### 3.3 Config, launcher, resume

`configs/qwen3_b200_snig.yaml`:

```yaml
_base_: qwen3_b200_base.yaml

experiment:
  method: snig
  output_dir: outputs/snig_opd

selector:
  joint_cross_scoring: true
  pgt_vocab_chunk_tokens: 2048
  # Gibbs allocation qua KL budget: KL(nu || uniform) <= epsilon; temperature
  # được giải mỗi step. Mặc định bằng cmt_allocation_kl để so sánh công bằng.
  snig_allocation_kl: 0.5
  snig_gamma: 1.0
  # Xem P3: hiệu chỉnh bằng successor_share ở step 1 trước long run.
  snig_successor_lambda: 1.0
```

Launcher và scripts:

- `scripts/common_b200.sh`: thêm `SNIG_CONFIG`, run/output paths, overrides `SNIG_ALLOCATION_KL`, `SNIG_GAMMA`, `SNIG_SUCCESSOR_LAMBDA`.
- `scripts/train_snig_b200.sh`: copy từ `train_cmt_b200.sh`, đổi namespace env sang `SNIG_*`.
- `scripts/train_all_b200.sh`, `smoke_test_b200.sh`, `smoke_test_fsdp_2gpu.sh`, `smoke_test_fsdp_multigpu.sh`: thêm `snig` vào method choices, disabled mặc định.

`b200_experiment/resume.py`: thêm tường minh `"selector.snig_allocation_kl"`, `"selector.snig_gamma"`, `"selector.snig_successor_lambda"` vào tuple frozen keys. `rollout.top_p` và `rollout.temperature` đã có sẵn.

`rollout.top_p = 1` là cấu hình chuẩn.

### 3.4 Logging và evaluation registry

- `diagnostics.py::selector_summary`:
  - thêm nhánh `snig` với các key §3.1 cộng `w` (nếu thiếu, hàm raise);
  - thêm `snig` vào điều kiện tính `effective_sample_size`.
- `selector_logging.py::TokenScoreStatsLogger` (bắt buộc, nếu thiếu thì raise lúc init). Ranges đề xuất:
  - `score`, `s_SNIG`, `gain`: (0, 10)
  - `successor_utility`: (−1e-2, 1e-2) — scale `O(1/T)`, dùng range của CMT sẽ dồn hết vào 1 bin
  - `kernel_derivative`: (−20, 20)
  - `transition_weight`, `support_common_mass`, `teacher_deficit`: (0, 1)
  - `R`: (0, 1e4); `M`: (0, 8192); `Phi`: (0, 5); `w`: (0, 20)

  Thêm vào payload một khối `"allocation": {epsilon, kl_achieved, inverse_temperature, temperature, successor_share}` (truyền từ trainer). Raw samples dùng cùng chỉ số `linspace` cho mọi field, nên có thể reconstruct từng mẫu `S = g + λu`, và `w` từ `β` với `logZ` suy ra qua mean(w)=1.
- `SelectedTokenLogger`: không cần sửa (disabled cho soft-weight methods).
- `tensorboard_logging.py`:
  - `SNIG_TAGS`: `snig/local_pgt_mean`, `snig/successor_utility_mean`, `snig/successor_utility_abs_q95`, `snig/Phi_mean`, `snig/score_mean`, `snig/score_std`, `snig/weight_std`, `snig/weight_max`, `snig/transition_weight_mean`;
  - scalar tags: `snig/allocation_kl`, `snig/inverse_temperature`, `snig/successor_share`, `snig/effective_token_fraction`.
- `plotting.py`, `cli.py`, `evaluation.py`, `checkpoint_evaluation.py`: label `SNIG-OPD`, slug `snig_opd`, argument `--snig-output`.
- `scripts/eval_snig_b200.sh`; nối vào `eval_all_b200.sh`, `eval_checkpoint_b200.sh`, `reeval_*`, `plot_results.sh`, `plot_training_progress.sh`.

Không cần plot chuyên biệt ở v1.

## 4. Tests bắt buộc

### `tests/test_snig_selector.py`

**Recursion.** `R, M` khớp `bellman_reference_scan` (rtol 1e-5 với T ≤ 64; với T=4096 dùng rtol 1e-4).

**Terminal.** Token cuối có `u=0` và `S=g`. Token trước padding cũng vậy (padding là hard boundary; mọi diagnostics ngoài valid bằng 0).

**Chống length bias.** Suffix có gain hằng số cho `u=0`, kể cả khi `c<1`. Assert `|u| ≤ 1e-6·max(g,1)` với T ≤ 512 (float32). Thêm một case T=4096 kiểm tra `|λu| / g < 1e-6`.

**Đồng nhất thức.** Trên dữ liệu ngẫu nhiên (float64 reference), `M_t R_{t+1} − R_t M_{t+1} = R_{t+1} − g_t M_{t+1}`.

**Nhánh accepted.** Mọi vị trí có `δ≠0` đều có `c=1`.

**Finite difference (P10-i).** Categorical nhỏ (|U|=3, không tie `p°=q°`, `m_p, m_q < 1`), successor `(R(sa), M(sa))` tất định theo từng action. Kiểm tra:
- `d/dε log1p(R_ε/M_ε)` với `R_ε = g + γΣ_a min(m_p p_ε(a), q°(a)) R(sa)` (tương tự cho M) và `p_ε ∝ p_U e^{εr}`, bằng FD trung tâm (h=1e-6, float64);
- khớp công thức giải tích `γ(Σ dk·R·M − R·Σ dk·M)/(M(M+R))`;
- khớp kỳ vọng liệt kê `E_{Y~p}[selector u | Y]` sau khi chia cho plug-in denominator của nhánh tương ứng. Test này chỉ khớp khi successor values không phụ thuộc action — ghi rõ đây là kiểm tra estimator, không phải unbiasedness.

**Kernel estimator.** `c ∈ [0,1]`; `Σ_{a∈U} p°(a)·c(a) = Σ_a min(p°_a, q°_a)`, và action ngoài U cho `c=0`.

**Quan hệ với CMT.** Cùng input: `λ=0` cho `score == gain` chính xác. Với `λ>0`: `successor_utility · M(M+R) == CMT.sequential_gain · (λ_snig/λ_cmt)`.

**Cơ bản.** Mọi output finite, detached, candidate support giữ nguyên (`candidate_ids`, log-probs, `support_mask` là cùng tensor).

**Không regression CMT.** Refactor helper dùng chung không đổi output: toàn bộ `tests/test_cmt_selector.py` pass.

### Allocation và globalization

- Với `ε ∈ (0, log N)`: `w > 0`, `mean(w) = 1`, ordering của `w` trùng ordering của `S`, `KL_achieved ≈ ε`.
- Cộng hằng số hoặc nhân `S` với `c>0` không đổi `w`.
- `ε → 0` cho uniform; tăng `ε` làm ESS giảm đơn điệu.
- Scalar allocation có mặt trong `train_metrics["selector"]`, CSV row và TensorBoard scalars (P7).
- Distributed gather/scatter (2 process, gloo) cho `w` giống single-process trên cùng global scores; mở rộng `test_distributed.py` / `test_fsdp_distributed.py`.

### Protocol

- Ở rollout đầu tiên (cùng checkpoint, seed, `rollout.backend=hf` hoặc vLLM deterministic): `rollout_hash` của `snig` và `opd` trùng nhau. Assertion "selector không mutate rollout" trong trainer vẫn giữ.
- Weighted OPD loss với `w ≡ 1` bằng loss OPD. Với cùng `w`, loss của `snig` và `cmt` bằng nhau (chỉ selector khác).
- Test mô tả P2: hai mini-batch có tổng weight khác nhau vẫn cho mỗi optimizer step một weighted mean chuẩn hoá riêng.

### Launcher, resume, logging

- `test_training_launchers.py`: `snig` được nhận, disabled mặc định.
- `test_resume.py`: đổi `snig_allocation_kl`, `snig_gamma` hoặc `snig_successor_lambda` bị từ chối khi resume.
- `test_selector_logging.py`, `test_tensorboard_logging.py`: `TokenScoreStatsLogger("snig")` và `selector_summary("snig")` không raise; payload có khối `allocation`.

## 5. Thứ tự implement

1. Tách helper shared-kernel khỏi CMT; chạy test CMT.
2. `SNIGSelector` và unit tests toán học.
3. Nối vào trainer, KL-Gibbs globalization và plumbing scalar metrics; chạy CPU tests.
4. Config, launcher, logging registries và resume safeguards.
5. Smoke test 1-GPU, rồi 2-GPU/FSDP; kiểm tra global allocation invariance.
6. **Calibration (P3).** Chạy 1 rollout step và đọc `successor_share` cùng histogram `successor_utility`. Nếu share < ~1%, ghi nhận SNIG ≈ PGT-Gibbs và quyết định λ (hoặc γ<1) trước khi chạy dài. Ghi quyết định vào `RESEARCH_DECISION.md`.
7. Short comparison với cùng seed, rollout batch và checkpoint schedule: **OPD / PGT / CMT / PGT-Gibbs (`snig`, λ=0) / SNIG**.

## 6. Điều kiện hoàn thành

- `pytest` pass, gồm finite-difference, CMT regression và distributed allocation tests.
- Không có thêm model forward so với CMT (so sánh `total_scoring_time_sec` và số lần gọi scoring).
- Global Gibbs weights có mean 1 và `KL_achieved ≈ snig_allocation_kl`. Normalization loss vẫn là weighted mean theo từng PPO mini-batch, như CMT, và được ghi đúng trong docstring và metadata.
- Resume từ SNIG checkpoint từ chối thay đổi `snig_allocation_kl`, `snig_gamma`, `snig_successor_lambda`.
- Mỗi logged step có đủ `g`, `u`, `λ`, `β`, `KL`, ESS, `successor_share`, để reconstruct `S_t = g_t + λu_t^succ` và `w` trên raw samples.
- Comparison có arm PGT-Gibbs (λ=0), để tách tác động của successor term khỏi tác động của allocator.
