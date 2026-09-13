# 可微切比雪夫-NUFFT混合层：方法设计文档

## 1. 问题定位

第一篇论文（finufft-PINN）处理**周期边界条件**下的非均匀数据：
- 域：$[-\pi, \pi]^d$，函数周期延拓
- 求导：Type1 NUFFT → 频域乘 $(ik)^m$ → Type2 NUFFT
- 局限：无法处理Dirichlet/Neumann/Robin等非周期边界

第二篇论文目标：**非周期边界 + 非均匀数据**，核心创新是切比雪夫与NUFFT的混合。

## 2. 数学基础

### 2.1 切比雪夫变换（离散余弦变换DCT实现）

切比雪夫多项式 $T_n(x) = \cos(n \arccos x)$，$x \in [-1, 1]$。

函数 $f(x)$ 的切比雪夫展开：
$$f(x) = \sum_{n=0}^{N} \!\!\!\!\prime\prime\, a_n T_n(x)$$
其中 $\sum\!\!\!\prime\prime$ 表示首末项减半。

**切比雪夫点**（Gauss-Lobatto）：$x_j = \cos(\pi j / N),\; j=0,\dots,N$

在切比雪夫点上，离散切比雪夫变换（DCT）与FFT等价：
$$a_n = \frac{2}{N} \sum_{j=0}^{N} \!\!\!\!\prime\prime\, f(x_j) \cos(n \pi j / N)$$

这正是Type-II DCT（scipy.fft.dct with type=2, norm='ortho'的变体）。

### 2.2 切比雪夫求导矩阵

切比雪夫系数空间中的求导：若 $f(x) = \sum a_n T_n(x)$，则
$$f'(x) = \sum b_n T_n(x), \quad b_n = \frac{2}{c_n} \sum_{\substack{k=n+1 \\ k+n \text{ odd}}}^{N} k a_k$$
其中 $c_0 = 2, c_n = 1$ (n≥1)。

物理空间求导矩阵 $D$（$N+1$ 阶方阵）：
$$D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j}, \quad i \neq j$$
$$D_{ii} = -\frac{x_i}{2(1-x_i^2)}, \quad 1 \leq i \leq N-1$$
$$D_{00} = \frac{2N^2+1}{6}, \quad D_{NN} = -\frac{2N^2+1}{6}$$

### 2.3 可微切比雪夫变换的实现方案

**方案A：DCT-based（推荐）**
- 前向：`scipy.fft.dct`（Type II）→ 切比雪夫系数
- 求导：系数空间递推（矩阵乘法，可微）
- 反向：`scipy.fft.idct`（Type III）→ 物理空间值
- 可微性：DCT本身是线性变换，可用矩阵表示或用torch.fft实现

**方案B：求导矩阵法**
- 构造切比雪夫求导矩阵 $D$（常数矩阵）
- $f' = D \cdot f$（矩阵乘法，天然可微）
- 局限：$O(N^2)$，但对PINN的配点数通常N≤128，可接受

**混合策略**：
- 边界层（切比雪夫区域）：用求导矩阵法，简单且边界条件处理直接
- 内部区域（NUFFT区域）：用finufft谱求导，处理非均匀采样

## 3. 混合层架构设计

### 3.1 域分解与坐标映射

将计算域 $\Omega = [a, b]$（1D）或 $[a,b]\times[c,d]$（2D）分为：
- **边界层** $\Omega_B$：靠近边界的区域，用切比雪夫变换处理
- **内部区域** $\Omega_I$：远离边界的区域，用NUFFT处理非均匀采样

**1D情形**：
- 左边界层：$[a, a+\delta]$，映射到切比雪夫域 $[-1, 1]$
- 右边界层：$[b-\delta, b]$，映射到切比雪夫域 $[-1, 1]$
- 内部：$[a+\delta, b-\delta]$，映射到周期域 $[-\pi, \pi]$（用于NUFFT）

**坐标映射函数**：
- 切比雪夫映射：$x_{cheb} = 2\frac{x - x_{left}}{x_{right} - x_{left}} - 1$
- NUFFT映射：$x_{fourier} = 2\pi \frac{x - x_{left}}{x_{right} - x_{left}} - \pi$

### 3.2 重叠区域与一致性约束

边界层与内部区域之间设**重叠区** $\Omega_O$（宽度 $\epsilon$）：
- 在重叠区内，两种变换都计算导数
- 一致性损失：$\mathcal{L}_{overlap} = \|u'_{cheb} - u'_{nufft}\|_{\Omega_O}^2$
- 或者用加权平均：$u' = w(x) u'_{cheb} + (1-w(x)) u'_{nufft}$，$w$ 在重叠区平滑过渡

### 3.3 完整求导管线

```
输入: u(x_j) at non-uniform points x_j, j=1..M
  │
  ├─ 区域分类: 每个点标记为 {left_bdry, right_bdry, interior, overlap}
  │
  ├─ 边界层求导 (Chebyshev):
  │   ├─ 插值到切比雪夫点 (RBF或多项式插值)
  │   ├─ 切比雪夫求导矩阵 D 作用
  │   └─ 插值回原始非均匀点
  │
  ├─ 内部求导 (NUFFT):
  │   ├─ Type1 NUFFT: u(x_j) → û_k
  │   ├─ 频域乘 (ik)^m
  │   └─ Type2 NUFFT: → u^(m)(x_j)
  │
  └─ 融合: 加权组合 / 一致性约束
输出: u^(m)(x_j) at all non-uniform points
```

## 4. 伴随算子推导

### 4.1 切比雪夫求导矩阵的伴随

切比雪夫求导矩阵 $D$ 是实矩阵，但**不是对称矩阵**。其伴随（转置）$D^T$ 用于反向传播。

若前向：$g = D f$，则反向：$\bar{f} = D^T \bar{g}$。

这天然可微，无需特殊处理。

### 4.2 DCT-based切比雪夫变换的伴随

DCT是正交变换（orthonormal），其伴随是其逆（IDCT）。
- 前向：$a = \text{DCT}(f)$
- 反向：$\bar{f} = \text{IDCT}(\bar{a})$

系数空间求导矩阵 $C$（将 $a_n$ 映射到 $b_n$）的伴随是 $C^T$。

### 4.3 混合变换的伴随

完整混合算子 $\mathcal{D}_{hybrid} = W_{cheb} \mathcal{D}_{cheb} P_{cheb} + W_{nufft} \mathcal{D}_{nufft} P_{nufft}$，其中：
- $P$：区域投影/插值算子
- $W$：区域权重（窗函数）

伴随：$\mathcal{D}_{hybrid}^* = P_{cheb}^T \mathcal{D}_{cheb}^* W_{cheb} + P_{nufft}^T \mathcal{D}_{nufft}^* W_{nufft}$

各部分的伴随：
- 插值矩阵 $P$ 的伴随：$P^T$（转置）
- 切比雪夫求导的伴随：$D^T$ 或 IDCT→$C^T$→DCT
- NUFFT的伴随：Type1↔Type2（第一篇论文已验证）
- 窗函数 $W$：逐元素乘法，伴随是自身

## 5. 边界条件的可微处理

### 5.1 Dirichlet边界

**硬约束**：网络输出时强制 $u(a) = g_a, u(b) = g_b$：
$$u_{NN}(x) = (x-a)(b-x) \cdot \text{MLP}(x) + \frac{b-x}{b-a}g_a + \frac{x-a}{b-a}g_b$$

**软约束**：边界损失 $\mathcal{L}_{bc} = \|u(a)-g_a\|^2 + \|u(b)-g_b\|^2$

切比雪夫求导矩阵法天然支持硬约束：边界点函数值固定，只优化内部点。

### 5.2 Neumann边界

$u'(a) = h_a$：在切比雪夫求导中，边界导数值由求导矩阵第一行给出，可直接约束。

### 5.3 Robin边界

$\alpha u(a) + \beta u'(a) = r_a$：组合Dirichlet和Neumann的处理。

## 6. 误差分析框架

### 6.1 切比雪夫谱精度

对解析函数，切比雪夫谱方法误差呈指数衰减：$\|f - f_N\| \leq C \rho^{-N}$。

### 6.2 NUFFT网格化误差

finufft的精度由 $\epsilon$ 参数控制，可达到 $10^{-14}$（双精度极限）。

### 6.3 混合误差

总误差 = 切比雪夫区域误差 + NUFFT区域误差 + 重叠区一致性误差 + 插值误差。

关键参数：边界层宽度 $\delta$、重叠区宽度 $\epsilon$、切比雪夫点数 $N_c$、NUFFT模数 $N_f$。

## 7. 实现路线图

### Phase 1: 基础组件（1-2天）
- [ ] 可微切比雪夫变换层（DCT-based + 求导矩阵法两种实现）
- [ ] gradcheck验证
- [ ] 切比雪夫求导精度测试（vs 解析解，机器精度）

### Phase 2: 混合层（2-3天）
- [ ] 域分解与坐标映射
- [ ] 区域分类与插值
- [ ] 混合求导算子（1D先，再2D）
- [ ] 伴随算子验证（gradcheck + adjoint identity test）

### Phase 3: PINN集成（2-3天）
- [ ] 边界条件硬约束/软约束
- [ ] 1D Poisson方程测试
- [ ] 2D Poisson方程测试
- [ ] 与标准切比雪夫谱方法、AD-PINN、finufft-PINN对比

### Phase 4: 扩展实验（3-5天）
- [ ] 热传导方程（Neumann/Robin）
- [ ] 波动方程（混合边界）
- [ ] 参数反演
- [ ] 消融实验

### Phase 5: 论文撰写（持续）
- [ ] 引言（基于文献调研）
- [ ] 方法论（理论推导）
- [ ] 数值实验
- [ ] 结论

## 8. 关键技术决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 切比雪夫实现 | 求导矩阵法为主，DCT为辅 | 求导矩阵简单、边界条件直接、N≤128时O(N²)可接受 |
| 域分解 | 左右边界层+内部，重叠区 | 经典谱元素法思路，保证边界精度 |
| 重叠区处理 | 加权窗函数平滑过渡 | 避免不连续性，比硬一致性约束更稳定 |
| 边界条件 | 硬约束（网络输出变换）+ 切比雪夫边界点固定 | 硬约束精度更高，避免边界损失权重调参 |
| 非均匀点处理 | 边界层用RBF插值到切比雪夫点，内部直接NUFFT | 边界层点少，插值开销可忽略；内部NUFFT原生支持非均匀 |
