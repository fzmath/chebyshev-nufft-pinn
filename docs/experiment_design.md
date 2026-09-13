# 数值实验设计文档

## 总体设置

### 计算环境
- Python 3.11, PyTorch 2.6.0+cu118, finufft 2.5.1, numpy 1.26.0, scipy 1.15.2
- CPU: Intel Core i7-8850H @2.60GHz 6核12线程, 32GB RAM
- GPU: NVIDIA Quadro P1000 4GB（可选，主要用CPU）

### 网络结构
- MLP: 输入层(2或3) → 隐藏层(64, Tanh) × 3 → 输出层(1)
- 初始化: Xavier uniform
- 优化器: Adam, lr=1e-3, warmup 500步, cosine annealing
- 梯度裁剪: max_norm=1.0
- 训练步数: 5000-10000（根据问题调整）
- 随机种子: 3个种子(42, 123, 456)，报告均值±标准差

### 评估指标
- 相对L2误差: $\|u_{pred} - u_{exact}\|_2 / \|u_{exact}\|_2$
- 相对L∞误差
- PDE残差范数
- 训练时间（秒）
- 每步平均时间

### 对比方法
1. **AD-PINN**: 标准PINN，自动微分求导
2. **RBF-FD PINN (DT-PINN)**: 径向基函数有限差分，非均匀点上局部求导
3. **finufft-PINN**: 纯NUFFT层（周期边界，在非周期问题上作为对照）
4. **Cheb-PINN**: 纯切比雪夫层（均匀Gauss-Lobatto点）
5. **FC-PINO**: Fourier Continuation PINO（均匀网格）
6. **Hybrid-PINN (Ours)**: 切比雪夫-NUFFT混合层

### 混合层默认参数
- delta = 0.15 * (b-a) （边界层宽度）
- overlap = 0.05 * (b-a) （重叠区宽度）
- N_cheb = 32 （切比雪夫模数）
- N_fourier = 16 （NUFFT傅里叶模数）
- eps_nufft = 1e-6
- window_scale = overlap / 4

---

## 实验1: 可微层验证（单元测试级别）

### 1.1 gradcheck验证
- 对DiffChebyshevDerivative和DiffHybridDerivative分别做gradcheck
- double precision, eps=1e-6, atol=1e-4, rtol=1e-3
- 报告：通过/失败

### 1.2 伴随恒等式
- $\langle D u, v \rangle = \langle u, D^* v \rangle$
- 随机u,v，N=16, M=100
- 报告：相对误差（应<1e-10）

### 1.3 谱收敛性
- 测试函数: $u(x) = \exp(\sin(\pi x))$ on $[-1,1]$
- N从8到128，画L∞误差vs N（半对数图）
- 预期：指数收敛到机器精度

---

## 实验2: 1D Poisson方程（Dirichlet边界）

### PDE
$$-u''(x) = f(x), \quad x \in (0,1)$$
$$u(0) = 0, \quad u(1) = 0$$

### 精确解
$$u(x) = \sin(\pi x) + 0.5\sin(3\pi x)$$
$$f(x) = \pi^2 \sin(\pi x) + 4.5\pi^2 \sin(3\pi x)$$

### 设置
- 配点: M=2000（Halton采样）+ 边界点
- 训练步数: 5000
- 硬约束: $u_{NN}(x) = x(1-x) \cdot \text{MLP}(x)$
- 对比: 所有6种方法

### 报告
- 收敛曲线（L2误差vs训练步数）
- 最终L2/L∞误差表
- 训练时间对比
- 不同M（500, 1000, 2000, 5000）的扩展性

---

## 实验3: 2D Poisson方程（Dirichlet边界）

### PDE
$$-\Delta u(x,y) = f(x,y), \quad (x,y) \in (0,1)^2$$
$$u|_{\partial\Omega} = 0$$

### 精确解
$$u(x,y) = \sin(\pi x)\sin(\pi y) + 0.3\sin(2\pi x)\sin(2\pi y)$$

### 设置
- 配点: M=5000（Halton采样）
- 训练步数: 8000
- 硬约束: $u_{NN}(x,y) = x(1-x)y(1-y) \cdot \text{MLP}(x,y)$
- 2D混合层: 逐方向1D混合 + 求和

### 报告
- 解的热力图（预测、误差）
- L2/L∞误差表
- 不同N_cheb/N_fourier的消融

---

## 实验4: 热传导方程（Neumann边界）

### PDE
$$u_t = \nu u_{xx}, \quad x \in (0,1), \quad t \in (0,1]$$
$$u_x(0,t) = 0, \quad u_x(1,t) = 0 \quad \text{(Neumann)}$$
$$u(x,0) = \cos(\pi x)$$

### 精确解
$$u(x,t) = \exp(-\nu \pi^2 t) \cos(\pi x)$$

### 设置
- 配点: M=3000（时空域Halton采样）
- 训练步数: 8000
- ν = 0.1
- Neumann边界处理: 软约束（边界损失）或硬约束（网络输出变换）

### 报告
- 不同时刻的解曲线
- L2误差vs时间
- 与AD-PINN对比（AD对Neumann的处理通常较弱）

---

## 实验5: 热传导方程（Robin边界）

### PDE
$$u_t = \nu u_{xx}, \quad x \in (0,1)$$
$$-u_x(0,t) + u(0,t) = 0, \quad u_x(1,t) + u(1,t) = 0 \quad \text{(Robin)}$$
$$u(x,0) = 1 - x^2$$

### 设置
- 配点: M=3000
- 训练步数: 8000
- 用有限差分数值解作为参考解（精细网格）

### 报告
- 解的时空热力图
- 边界条件满足度
- 与AD-PINN对比

---

## 实验6: 波动方程（混合边界）

### PDE
$$u_{tt} = c^2 u_{xx}, \quad x \in (0,1), \quad t \in (0,1]$$
$$u(0,t) = 0 \quad \text{(Dirichlet)}$$
$$u_x(1,t) = 0 \quad \text{(Neumann)}$$
$$u(x,0) = \sin(\pi x / 2), \quad u_t(x,0) = 0$$

### 精确解
$$u(x,t) = \sin(\pi x / 2) \cos(c \pi t / 2)$$

### 设置
- 配点: M=4000
- 训练步数: 10000
- c = 1.0
- 混合边界: 左端硬约束Dirichlet，右端软约束Neumann

### 报告
- 不同时刻的波形
- 能量守恒验证
- 与AD-PINN对比（二阶时间导数AD开销大）

---

## 实验7: 参数反演

### 问题
在热传导方程中反演扩散系数ν：
$$u_t = \nu u_{xx}$$
- 给定稀疏观测数据（M_obs=50个时空点，含1%噪声）
- ν初始猜测: 0.05（真实值0.1）
- 同时训练网络和ν

### 设置
- 配点: M=3000
- 观测点: 50个，噪声σ=0.01
- 训练步数: 10000
- ν为可学习参数

### 报告
- ν的收敛曲线
- 最终ν相对误差
- 不同噪声水平（0%, 1%, 5%, 10%）的鲁棒性
- 与AD-PINN反演对比

---

## 实验8: 消融实验

### 8.1 边界层宽度δ
- δ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}
- 固定其他参数
- 1D Poisson，报告L2误差
- 预期: δ太小则边界处理不足，δ太大则NUFFT区域太小

### 8.2 重叠区宽度ε
- ε ∈ {0, 0.025, 0.05, 0.1, 0.15}
- 1D Poisson，报告L2误差和训练稳定性
- 预期: ε=0可能有不连续，ε适中最优

### 8.3 切比雪夫模数N_cheb
- N_cheb ∈ {8, 16, 32, 64, 128}
- 固定N_fourier=16
- 报告误差和计算时间
- 预期: 指数收敛后平台

### 8.4 傅里叶模数N_fourier
- N_fourier ∈ {4, 8, 16, 32, 64}
- 固定N_cheb=32
- 报告误差
- 预期: 内部区域精度提升，但非周期函数在NUFFT区域有Gibbs现象

### 8.5 纯切比雪夫 vs 纯NUFFT vs 混合
- 纯切比雪夫: δ=0.5（全边界层）
- 纯NUFFT: δ=0.01（几乎全内部）
- 混合: δ=0.15
- 在1D/2D Poisson上对比
- 预期: 混合在非周期边界+非均匀点上最优

### 8.6 均匀 vs 非均匀配点
- 均匀网格 vs Halton随机 vs 带噪声的随机
- 混合层 vs AD-PINN
- 预期: 混合层在非均匀点上优势更明显

### 8.7 窗函数类型
- Sigmoid窗 vs 线性窗 vs 余弦窗
- 报告误差和训练稳定性

---

## 实验9: 计算复杂度分析

### 测量
- 不同M（配点数）下，每步训练时间
- 不同N_cheb, N_fourier下的前向/反向时间
- 与AD-PINN的时间对比（AD时间随导数阶数增长）

### 报告
- 时间复杂度表
- 前向vs反向时间分解
- 内存使用

---

## 结果呈现规范

### 表格
- 所有对比表用booktabs
- 均值±标准差（3个种子）
- 最优值加粗

### 图片
- 收敛曲线: 对数y轴，清晰图例
- 解的热力图: colorbar，统一colormap
- 消融实验: 折线图或热力图
- 所有图保存为PNG（300dpi）和PDF

### 统计检验
- 对关键对比做配对t检验（3个种子）
- 报告p值

---

## 实验执行顺序

1. 实验1（层验证）— 混合层实现后立即做
2. 实验2（1D Poisson）— 核心验证
3. 实验8.5（纯vs混合消融）— 证明混合必要性
4. 实验3（2D Poisson）— 维度扩展
5. 实验4-6（时变问题）— 普适性
6. 实验7（反演）— 应用价值
7. 实验8.1-8.4, 8.6, 8.7（消融）— 深入分析
8. 实验9（复杂度）— 效率分析
