# 文献综述：可微切比雪夫–NUFFT 混合层用于非周期边界 PINN

> 调研目标：为投稿 *Journal of Computational Physics* 的论文《可微切比雪夫-NUFFT 混合层（Differentiable Chebyshev-NUFFT Hybrid Layer）用于带 Dirichlet/Neumann/Robin 边界条件、支持非均匀空间采样的物理信息神经网络》做 Related Work 定位。
> 检索时间：2026-09-12。重点覆盖 2020–2026 年文献；每条给出标题、作者、年份、出处、核心贡献、与本工作的差异。链接均为 arXiv 摘要页或出版商页面（DOI 可解析）。

---

## 一、方向 1：可微谱方法 / 可微傅里叶算子用于深度学习

本方向关注：用可微的谱/傅里叶算子层替代 PINN 中昂贵的自动微分（autograd），以及神经算子（neural operator）中的谱卷积。核心结论是：**"可微傅里叶层"在算子学习中已成熟（FNO/PINO 族），但几乎全部建立在均匀网格 + 周期延拓之上；把 NUFFT 嵌入 PyTorch autograd 并直接用于 PINN 残差的工作仍很少。**

1. **Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. (2021).** *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. https://arxiv.org/abs/2010.08895
   - **核心贡献**：在傅里叶空间参数化积分核，FFT 卷积 + 逐点非线性激活，实现零样本超分辨率；Burgers/Darcy/NS 上比传统求解器快三个数量级。
   - **与本工作差异**：FNO 是"算子学习"框架（学解映射），输入必须是均匀矩形网格，FFT 天然假设周期；它不解决非周期 BC，也不在非均匀配置点上做谱导数。

2. **Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021).** *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* *Machine Learning: Science and Technology* 2, 025015. https://arxiv.org/abs/1910.03193
   - **核心贡献**：分支网 + 主干网结构逼近非线性算子，算子泛逼近定理；PINN 与 DeepONet 的经典对照基线。
   - **与本工作差异**：DeepONet 是 MLP 架构，空间导数仍靠 autograd，无谱层；非周期 BC 由软约束或 trial function 处理，无切比雪夫/NUFFT 成分。

3. **Wang, S., Wang, H., Perdikaris, P. (2021).** *Learning the solution operator of parametric PDEs with physics-informed DeepONets.* *Science Advances* 7(20), eabi8605. https://www.science.org/doi/10.1126/sciadv.abi8605
   - **核心贡献**：把物理残差直接写进 DeepONet 训练，无配对数据也可学算子；揭示 PINN 谱偏置（spectral bias）。
   - **与本工作差异**：纯 MLP 主干、autograd 求导；不提供谱精度的空间导数算子层。

4. **Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., Anandkumar, A. (2024).** *Physics-Informed Neural Operator for Learning Partial Differential Equations (PINO).* *ACM/J. Data Science* 1(3), Art. 9. https://dl.acm.org/doi/10.1145/3648506
   - **核心贡献**：数据驱动预训练 + 物理残差微调；在傅里叶空间精确计算卷积梯度，谱微分替代 autograd。
   - **与本工作差异**：PIN O 的谱微分假设周期解，非周期问题靠 padding/延拓，存在 ill-conditioning（见 FC-PINO）；仍需均匀网格，不支持任意非均匀采样点。

5. **Maust, H., Li, Z., Wang, Y., Leibovici, D., Bruno, O., Hou, T., Anandkumar, A. (2022).** *Fourier Continuation for Exact Derivative Computation in Physics-Informed Neural Operators (FC-PINO).* NeurIPS 2022（扩展版 arXiv v3 2024/2025，Ganeshram 等加入 FC-Legendre/FC-Gram）. https://arxiv.org/abs/2211.15960
   - **核心贡献**：把 Oscar Bruno 的 Fourier Continuation 引入 PINO，在延拓域上把非周期函数光滑延拓为周期函数，从而在傅里叶空间精确求导；方程损失比 padded PINO 低数个数量级。
   - **与本工作差异（最接近的对照之一）**：FC 是"为傅里叶周期化服务的光滑延拓"，本质仍是 FFT-on-uniform-grid 范式，处理的是算子学习（PINO）而非单实例 PINN；**它不使用 NUFFT，也不直接在任意非均匀配置点输出谱导数**。本工作以切比雪夫基天然承载非周期 BC，省去 FC 求解延拓线性系统的代价。

6. **Li, Z., Huang, D. Z., Liu, B., Anandkumar, A. (2023).** *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries (Geo-FNO).* *JMLR* 24, 23-0064. https://jmlr.org/papers/volume24/23-0064/23-0064/
   - **核心贡献**：学习把不规则物理域形变映射到均匀隐式网格，再在隐空间跑 FNO；同时支持点云/网格输入。
   - **与本工作差异**：通过可学习坐标映射绕开 FFT 网格限制，坐标映射本身不精确、且是黑箱；本工作用 NUFFT 直接在非均匀点集上做谱变换，更精确、无学习式插值误差。

7. **Li, Z., Kovachki, N., Choy, C., et al. (2023).** *Geometry-Informed Neural Operator for Large-Scale 3D PDEs (GINO).* NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
   - **核心贡献**：图神经算子把不规则点云映射到规则隐网格，再上/采样回任意几何；离散化收敛。
   - **与本工作差异**：邻居搜索使 GINO 对查询点不可微（White et al. 2023 修补）；本工作的 NUFFT 直接生成对输入点和网络参数均可微的导数。

8. **Lingsch, L., Michelis, M. Y., de Bézenac, E., Perera, S. M., Katzschmann, R. K., Mishra, S. (2024).** *Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains.* ICML 2024. https://arxiv.org/abs/2305.19663
   - **核心贡献**：观察到"少数傅里叶模式足以表达算子"，直接用谱变换在非等距点上的求值替代 FFT，把 FNO 类算子推广到任意点分布；训练更快、精度不损。
   - **与本工作差异（与 NUFFT 思想最接近的公开工作）**：它用直接（非 FFT）谱求值处理非均匀点，但目标是算子学习、仍假设周期基（复指数），**未处理非周期 Dirichlet/Neumann/Robin BC，也未与切比雪夫基耦合**。可作为"非均匀点上可微谱算子"的直接先例被对比。

9. **Barnett, A. H., Magland, J., af Klinteberg, L. (2019).** *A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel (FINUFFT).* *SIAM J. Sci. Comput.* 41(5), C479–C504. https://arxiv.org/abs/1808.06736
   - **核心贡献**：FINUFFT 库的原始算法论文；类型 1/2/3 变换，指数型核误差，无预计算。
   - **与本工作差异**：这是底层数值库，本身不带 autograd 接口、不涉及 PINN/BC；本工作的第 0 步就是为 FINUFFT 包一层可微 PyTorch 算子并嵌入 PINN 残差。

> **小结**：可微傅里叶算子已在算子学习中成熟（FNO/PINO/GINO/Geo-FNO），非周期问题主要靠 FC 延拓（FC-PINO）或坐标映射（Geo-FNO）绕过；**"把 NUFFT 包进 autograd、直接在非均匀配置点上给 PINN 提供谱导数"这件事，公开文献中只有 Lingsch et al. (2024) 在算子学习侧做了思想相近的一步，尚无人把它与非周期边界的切比雪夫基耦合。**

---

## 二、方向 2：切比雪夫谱方法的可微实现

核心结论：**切比雪夫基天然适合有界区间上的非周期函数，近两年出现了一批"切比雪夫神经网络/神经算子"，但它们多在均匀 Gauss–Lobatto 节点上做 Galerkin/配置，尚未与 NUFFT 式的任意点采样结合。**

1. **Fanaskov, V., Oseledets, I. (2022).** *Spectral Neural Operators (SNO).* https://arxiv.org/abs/2205.10573
   - **核心贡献**：输入和输出都用切比雪夫/傅里叶级数展开，算子在谱系数空间操作；输出透明、无混叠、含大量无损（lossless）谱运算；多个算子上优于 FNO/DeepONet。
   - **与本工作差异**：SNO 用固定全局谱基 + 标准稠密线性代数，是算子学习架构；切比雪夫变换在均匀节点上做，**不支持任意非均匀点求值，也不显式处理 Dirichlet/Neumann/Robin BC 的硬约束**。

2. **Yin, P., Ling, S., Ying, W. (2024).** *Chebyshev Spectral Neural Networks for Solving Partial Differential Equations (CSNN).* https://arxiv.org/abs/2407.03347
   - **核心贡献**：单层网络用切比雪夫谱方法构造满足边界条件的神经元，autograd 反向传播；避免求解非稀疏线性系统；多网络拼接可处理复杂域。
   - **与本工作差异**：这是最接近"可微切比雪夫层用于 PINN"的工作。但 CSNN 的切比雪夫节点是均匀 Gauss–Lobatto 型，导数仍依赖 autograd/切比雪夫微分矩阵，**不支持任意非均匀采样点，也没有 NUFFT 成分**。本工作与它的差异恰在于：非均匀点上的谱导数由 NUFFT 完成。

3. **Abid, M., San, O. (2025).** *Spectral Embedding via Chebyshev Bases for Robust DeepONet Approximation (SEDONet).* https://arxiv.org/abs/2512.09165
   - **核心贡献**：把 DeepONet 主干网的坐标输入换成固定切比雪夫谱字典，专门针对有界域上 Dirichlet/Neumann 问题的非周期结构；Poisson/Burgers/输运扩散/Allen–Cahn/Lorenz-96 上比 DeepONet 和 Fourier 嵌入主干平均低 30–40% L2 误差。
   - **与本工作差异**：SEDONet 只在"输入嵌入"层面用切比雪夫字典，空间导数仍走 autograd；它证明了"切比雪夫归纳偏置对非周期问题有效"，但**没有可微切比雪夫微分层，也不处理非均匀采样**——这正是本工作要补的。

4. **Tang, S., Li, B., Yu, H. (2020).** *ChebNet: Efficient and Stable Constructions of Deep Neural Networks with Rectified Power Units via Chebyshev Approximation.* *Communications in Computational Physics* 27(2), 379–411. https://arxiv.org/abs/1911.05467
   - **核心贡献**：用切比雪夫多项式逼近的分层结构构造 RePU 网络，比幂级数构造更稳定，可获得谱精度。
   - **与本工作差异**：函数逼近层面的经典工作，不涉及 PDE 残差、边界条件或非均匀点。

5. **Xu, Z., Chen, Y., Xiu, D. (2024).** *Chebyshev Feature Neural Network for Accurate Function Approximation (CFNN).* https://arxiv.org/abs/2409.19135
   - **核心贡献**：首层用可学习频率的切比雪夫函数做特征，配合多阶段训练可逼近到机器精度，维数到 20。
   - **与本工作差异**：纯函数逼近，未嵌入 PDE 物理残差或 BC 处理。

6. **Liu, Z., Wang, H., Bao, K., Xu, Q., Zhang, H., Song, S. (2022).** *Render unto Numerics: Orthogonal Polynomial Neural Operator for PDEs with Nonperiodic Boundary Conditions (OPNO/SOL).* https://arxiv.org/abs/2206.12698
   - **核心贡献（本工作最直接的竞品）**：提出谱算子学习 SOL，其中 OPNO 变体用正交多项式基，**理论证明严格满足 Dirichlet、Neumann、Robin 边界条件**，误差达 1e-6，比二阶精细网格 FDM 快近 5 个数量级。
   - **与本工作差异**：OPNO 在固定正交多项式配置（均匀节点）上做 Galerkin 型学习，是"算子学习"框架；它把 BC 编进基函数，但**不支持非均匀空间采样，也没有 NUFFT 层**。本工作把"严格 BC"这一思想从均匀配置推广到任意非均匀点集，并以切比雪夫–NUFFT 混合层替代纯多项式配置。

7. **Du, Y., Chalapathi, N., Krishnapriyan, A. S. (2024).** *Neural Spectral Methods: Self-supervised Learning in the Spectral Domain.* ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/7da97523bfe8b9ba3fa485da5d7e0745-Paper-Conference.pdf
   - **核心贡献**：用正交基在谱系数空间学解映射，利用 Parseval 恒等式在谱域构造损失，训练/推理复杂度与时空分辨率无关。
   - **与本工作差异**：自监督谱域算子学习，正交基固定在规则节点上，不处理任意点采样或三类 BC 的硬约束。

> **小结**：切比雪夫可微层在"函数逼近"（ChebNet/CFNN）和"算子学习"（SNO/SEDONet/OPNO）两条线都已被验证有效，且 OPNO 证明了正交多项式基可严格承载三类非周期 BC；**空白在于：这些切比雪夫层都停在均匀/规则配置节点，没有一个把可微切比雪夫微分算子放到任意非均匀点集上。**

---

## 三、方向 3：切比雪夫与傅里叶 / NUFFT 的混合、区域分解耦合

核心结论：**传统数值分析里"傅里叶处理周期方向 + 延拓/切比雪夫处理非周期方向"的混合谱方法已成熟（FC 族、多域谱方法），但把这种混合以"可 autograd 微分层"的形式搬进 PINN 的，只有 FC-PINO 一例，且它是傅里叶延拓路线而非切比雪夫–NUFFT 路线。**

1. **Bruno, O. P., canon of Fourier Continuation** —— 代表论文：Bruno & Paul (2020), *Two-dimensional Fourier Continuation and applications.* https://arxiv.org/abs/2010.03901 ；Fontana, Bruno, Mininni, Dmitruk (2020), *Fourier Continuation method for incompressible fluids with boundaries (SPECTER).* https://arxiv.org/abs/2002.01392
   - **核心贡献**：FC 通过在边界附近求解一个（带 Legendre/Gram 基的）小线性系统，把非周期函数光滑延拓为周期函数，从而在均匀网格上用 FFT 获得谱精度导数；与重叠 patch 区域分解结合可解一般域椭圆/波动问题。
   - **与本工作差异**：FC 是"为用 FFT 而做周期化"，延拓系统的条件数与代价是已知痛点；本工作反过来——用切比雪夫基原生承载非周期端、用 NUFFT 处理任意点，**避免求解延拓线性系统**。可在 Related Work 中把 FC-PINO 与本工作作"路线对比"。

2. **Stein, D. B., Guy, R. D., Thomases, B. (2020).** *Immersed Boundary Smooth Extension (IBSE): A high-order method for solving PDE on arbitrary smooth domains using Fourier spectral methods.* *J. Comput. Phys.* 404, 109158. （eScholarship 全文：https://escholarship.org/uc/item/qt27n8r19q ）
   - **核心贡献**：把解从物理域光滑延拓到更大的笛卡尔计算域，使傅里叶谱方法可在任意光滑域上以四阶（Dirichlet）/三阶（Neumann）精度求解。
   - **与本工作差异**：IBSE 是传统数值延拓方法，非深度学习、不可微、均匀网格；它与 FC 同属"延拓换周期"思想，可作为本工作"为什么不直接用延拓"的对照。

3. **Pfeiffer, H. P., Kidder, L. E., Scheel, M. A., Teukolsky, S. A. (2003).** *A multidomain spectral method for solving elliptic equations.* *Computer Physics Communications* 152, 253. https://arxiv.org/abs/gr-qc/0202096
   - **核心贡献**：多域拟谱配置 + 区域分解，切比雪夫子域（矩形块）与球谐子域（球壳）混用，支持接触与重叠子域；广义相对论初始数据求解器（Carpet/Cactus）。
   - **与本工作差异**：这是"切比雪夫子域耦合"的经典数值先例，证明跨子域拼接的接口条件可行；但它是传统直接求解器，**不可微、无 autograd、无 NUFFT**。本工作把这种多域拼接思想升级为可学习、可微的层。

4. **Canuto, C., Funaro, D. (1988).** *The Schwarz algorithm for spectral methods.* *SIAM J. Numer. Anal.* 25(1), 24–40.
   - **核心贡献**：Schwarz 交替法与谱方法（切比雪夫/勒让德配置）的能量范数收敛证明；重叠区域处理的理论基础。
   - **与本工作差异**：纯收敛性理论；本工作在重叠区域处理上可引用它作为"为什么重叠宽度 m 足够即可缝合"的理论依据。

5. **van der Sande, K., Appelö, D., Albin, N. (2021).** *Fourier Continuation Discontinuous Galerkin Methods for Linear Hyperbolic Problems.* https://arxiv.org/abs/2105.00123
   - **核心贡献**：把 FC 作为 DG 方法的新基，结合 DG 的区域分解灵活性。
   - **与本工作差异**：另一条"FC + 区域分解"的传统路线，非学习框架。

6. **Bruno, O. P., Elling, T., Sen, A. (2015).** *A Fourier Continuation Method for the Solution of Elliptic Eigenvalue Problems in General Domains.* *Mathematical Problems in Engineering* 2015, 184786. https://downloads.hindawi.com/journals/mpe/2015/184786.pdf
   - **核心贡献**：FC + 重叠 patch 区域分解 + Arnoldi 迭代解一般域椭圆特征值问题。
   - **与本工作差异**：传统混合谱-区域分解求解器，非可微层。

> **小结**："切比雪夫子域 + 傅里叶/延拓 + 重叠拼接"在传统数值计算中是成熟范式；**把这一范式重写为可微、可嵌入 autograd、并以 NUFFT 打通任意非均匀点的"混合层"，公开文献中尚属空白**——FC-PINO 只做了傅里叶延拓那一半，SNO/CSNN 只做了切比雪夫那一半。

---

## 四、方向 4：非周期边界条件下的 PINN（Dirichlet / Neumann / Robin）

核心结论：**PINN 处理 BC 的主流仍是软约束（loss 罚项），硬约束（精确施加）主要靠"距离函数 / R-函数 / transfinite interpolation"构造 trial function；谱方法 PINN 中系统处理三类 BC 的代表性工作是 OPNO（方向 2）。**

1. **Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* *J. Comput. Phys.* 378, 686–707. https://arxiv.org/abs/1711.10561
   - **核心贡献**：PINN 奠基作；BC/初始条件作为软罚项进入损失，导数全靠 autograd。
   - **与本工作差异**：软约束、autograd、无谱层；本工作以谱导数层 + 硬/软 BC 混合策略改进。

2. **Sukumar, N., Srivastava, A. (2022).** *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.* *Computer Methods in Applied Mechanics and Engineering* 389, 114333. https://arxiv.org/abs/2104.08426
   - **核心贡献**：用 R-函数构造近似距离函数 φ(x)，以 trial = φ·NN（齐次 Dirichlet）及其 transfinite interpolation 推广，先验精确满足非齐次 Dirichlet、Neumann、Robin；损失只剩内部残差。
   - **与本工作差异**：这是"硬约束构造"的标准引用。本工作的切比雪夫层本身在基函数层面就近似齐次 Dirichlet（如 Chebyshev–Gauss–Lobatto 端点信息），可与该 trial-function 技术互补/对比；NUFFT 端仍需另处理 Robin 通量项。

3. **Rao, C., Sun, H., Liu, Y. (2021).** *Physics-informed deep learning for computational elastodynamics without labeled data.* *J. Eng. Mech.* 147(8), 04021043.（扩展为 CMAME 论文）. https://arxiv.org/abs/2006.08472
   - **核心贡献**：混合变量输出（位移+应力），组合多个子网络以"硬"方式施加 I/BC，克服软约束罚项在复杂 BC 下难以满足的问题。
   - **与本工作差异**：硬约束靠网络结构拼接，与谱微分层正交；可作为"硬约束为什么重要"的动机引用。

4. **Dalton, D., Lazarus, A., Gao, H., Husmeier, D. (2024).** *Boundary constrained Gaussian processes for robust physics-informed machine learning of linear PDEs (BCGP).* *JMLR* 25. https://dl.acm.org/doi/abs/10.5555/3722577.3722849
   - **核心贡献**：为 Dirichlet/Neumann/Robin/混合 BC 系统设计边界约束核，证明 Dirichlet 下的万能表示能力，并与无限宽边界约束 NN 等价。
   - **与本工作差异**：GP 框架；其"三类 BC 统一硬约束"的分类法可被本工作借用，用于动机和对比。

5. **Gladstone, R. J., Nabian, M. A., Meidani, H. (2022).** *FO-PINNs: A First-Order formulation for Physics Informed Neural Networks.* NeurIPS 2022 MLPS Workshop. https://arxiv.org/abs/2210.14320
   - **核心贡献**：一阶 PDE 公式化去掉高阶反传，并与近似距离函数的精确 BC 施加兼容。
   - **与本工作差异**：减少 autograd 次数的思路与本工作"用谱层替代 autograd"同向，但它用的是一阶系统重写，非谱方法。

6. **Xia, M., Böttcher, L., Chou, T. (2023).** *Spectrally adapted physics-informed neural networks for solving unbounded domain problems.* *Machine Learning: Science and Technology* 4, 025005. https://iopscience.iop.org/article/10.1088/2632-2153/acd0a1
   - **核心贡献**：把自适应谱方法技术（映射、加权）嵌入 PINN，求解无界域 PDE；展示谱自适应与 PINN 结合的可行性。
   - **与本工作差异**：面向无界域的坐标映射谱自适应，非三类有界 BC 的硬约束，也非非均匀点谱导数。

7. **Sahli Costabal, F., Pezzuto, S., Perdikaris, P. (2023).** *Δ-PINNs: physics-informed neural networks on complex geometries.* *CMAME* 418, 116465. https://arxiv.org/abs/2209.03984
   - **核心贡献**：用 Laplace–Beltrami 特征函数做位置编码，把域拓扑信息告诉 PINN。
   - **与本工作差异**：几何/特征函数编码路线；与切比雪夫（区间上的正交基）形成"域几何自适应基"的对照。

> **小结**：非周期 BC 在 PINN 中已有成熟的硬约束工具（距离函数/R-函数 trial）和理论（BCGP），但这些工具默认与 autograd 型 PINN 配合；**把"谱层精确微分"与"硬/软 BC 处理"在非均匀点上统一起来，尚未见到系统工作。**

---

## 五、方向 5：非均匀数据上的谱方法 PINN / mesh-free PINN / RBF 结合

核心结论：**PINN 本身 mesh-free，但"用非均匀点上的数值微分替代 autograd"这条线主要由 RBF-FD / RBF-DQ 驱动；谱方法因 FFT 要求均匀网格，长期缺席于非均匀点场景——这正是 NUFFT 可以切入的缺口。**

1. **Sharma, R., Shankar, V. (2022).** *Accelerated Training of Physics-Informed Neural Networks (PINNs) using Meshless Discretizations (DT-PINNs).* NeurIPS 2022. https://papers.nips.cc/paper_files/paper/2022/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html
   - **核心贡献**：用无网格 RBF-FD 高阶差分替代 autograd 计算空间导数，稀疏矩阵–向量乘；不规则点云上可用；fp64 下训练比 vanilla PINN 快 2–4 倍。
   - **与本工作差异（非均匀点数值微分的直接竞品）**：RBF-FD 是局部多项式阶精度，**而非全局谱精度**；本工作的 NUFFT 谱导数在光滑解上有指数收敛，代价是 O(N log N)，而 RBF-FD 是局部 O(N)。两者可在收敛阶和非均匀分布鲁棒性上正面对比。

2. **Xiao, Y., Yang, L. M., Du, Y. J., Song, Y., Shu, C. (2024).** *Radial basis function-differential quadrature-based physics-informed neural network (RBFDQ-PINN) for steady incompressible flows.* *Physics of Fluids* 36, 033617.
   - **核心贡献**：用 RBF-DQ 替代 AD 计算配置点导数，高阶导计算比 AD 快、点分布比 FD 灵活；顶盖方腔、后台阶、圆柱绕流。
   - **与本工作差异**：同为"非均匀点数值微分进 PINN"，但 RBF-DQ 是无网格局部差分，精度与稳定性依赖形状参数；NUFFT 谱路线无形状参数、谱收敛。

3. **Ramabathiran, A. A., Ramachandran, P. (2021).** *SPINN: Sparse, Physics-based, and Interpretable Neural Networks for PDEs.* *Neural Computing and Applications* （扩展版）. https://arxiv.org/abs/2102.13037
   - **核心贡献**：把无网格表示重解释为稀疏可解释网络；显式把傅里叶级数表示为一类 SPINN。
   - **与本工作差异**：桥接 PINN 与无网格数值方法的早期代表；其"傅里叶表示即稀疏网络"的观点可引出"为什么把 NUFFT 作为可微层是自然的"。

4. **Hedayatrasa, S., Fink, O., Van Paepegem, W., Kersemans, M. (2024).** *k-space Physics-informed Neural Network (k-PINN).* https://arxiv.org/abs/2404.03966
   - **核心贡献**：在波数域用傅里叶基构造 PINN 解空间，稀疏表示缓解谱偏置并降低损失计算成本。
   - **与本工作差异**：仍在规则网格/傅里叶域，未处理非均匀点或三类 BC。

5. **Yu, T., Qi, Y., Oseledets, I., Chen, S. (2024).** *Fourier Spectral Physics Informed Neural Network: An Efficient and Low-Memory PINN.* https://arxiv.org/abs/2408.16414
   - **核心贡献**：用"谱空间乘法"替代微分算子，去掉空间导数 autograd；指数收敛 + 低显存；给出物理域↔谱域两种训练策略。
   - **与本工作差异**：与本工作动机最一致（"用谱层替代 autograd 求导"），但它在均匀网格上做傅里叶变换、周期假设；本工作把它升级为切比雪夫（非周期）+ NUFFT（非均匀点）。

6. **Lin, H., Wu, L., Xu, Y., Huang, Y., Li, S., Zhao, G., Li, S. Z. (2022).** *Non-equispaced Fourier Neural Solvers for PDEs (NFS).* https://arxiv.org/abs/2212.04689
   - **核心贡献**：自适应重采样到等距点 + FNO 变体，处理非等距数据；首个在非等距情形保持网格不变性的湍流建模方法。
   - **与本工作差异**：靠"重采样回等距点"绕开非均匀，引入插值误差；本工作用 NUFFT 直接在原始非均匀点上变换，不做重采样。

> **小结**：非均匀点上的 PINN 数值微分开辟了 RBF-FD/RBF-DQ 一条成熟路线，但它们是局部阶精度；**全局谱精度 + 非均匀点 + 可微这三点同时满足的工作几乎没有——NUFFT 正是连接"谱精度"与"非均匀点"的天然工具，而本工作的额外贡献是再叠加非周期 BC。**

---

## 六、方向 6：相关前期工作与最接近的公开先例（可微非均匀谱导数）

本方向对应"用户前期 finufft-PINN（周期边界）"的定位。公开检索中，**把 NUFFT/直接谱求值嵌入深度学习训练图、用于 PDE 求解的工作非常稀少**，主要有以下几篇可作为"前期工作/同代竞争"被引用。

1. **用户前期工作（按用户描述）：** *Differentiable non-uniform spectral derivatives for physics-informed learning（finufft-PINN）*，处理周期边界条件。
   - **定位**：本工作的直接前身；已证明"把 finufft 包进 PyTorch autograd、在非均匀点上给周期问题提供谱导数"可行。
   - **本工作的推进**：从周期 → 非周期（Dirichlet/Neumann/Robin），从纯傅里叶 NUFFT → 切比雪–NUFFT 混合层。Related Work 中应把它作为"已解决的子问题"和"本文的直接扩展"明确陈述，并据此划清 novelty 边界。
   - 注：该工作若已正式发表，请在最终稿中替换为正式引用信息（DOI/期刊）。

2. **Lingsch, L. et al. (2024)** （同方向 1 第 8 篇） *Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains.* ICML 2024. https://arxiv.org/abs/2305.19663
   - **相关性**：与"可微非均匀谱导数"最接近的公开方法——直接谱变换在非等距点上求值、端到端可微。
   - **差异**：算子学习、周期复指数基、未处理非周期 BC；与用户的 finufft-PINN 相比，它不针对单实例 PINN 残差。

3. **White, C., Berner, J., Kossaifi, J., Elleithy, M., Pitt, D., Leibovici, D., Li, Z., Azizzadenesheli, K., Anandkumar, A. (2023).** *Physics-Informed Neural Operators with Exact Differentiation on Arbitrary Geometries.* NeurIPS 2023. https://openreview.net/forum?id=qd7q9lB5hY
   - **相关性**：修复 GINO 的邻居搜索不可微问题，使任意几何上的查询点导数可精确计算。
   - **差异**：基于图/GINO，非谱方法；"任意点精确导数"的卖点与本工作重合，但机理完全不同（图消息传递 vs 谱变换）。

4. **Lin, R. Y., Berner, J., Duruisseaux, V., Pitt, D., Leibovici, D., Kossaifi, J., Azizzadenesheli, K., Anandkumar, A. (2025).** *Enabling Automatic Differentiation with Mollified Graph Neural Operators (mGNO).* https://arxiv.org/abs/2504.08277
   - **相关性**：在不规则点云上用软化 GNO + autograd 算精确梯度，物理损失可在随机采样点处算。
   - **差异**：仍是图神经算子路线；其"非均匀点上精确导数"的经验结论可作为对比基线。

5. **Dutt, A., Rokhlin, V. (1995).** *Fast Fourier transforms for nonequispaced data, II.* *Applied and Computational Harmonic Analysis* 2, 85–100.
   - **相关性**：NUFFT 的算法源头；类型 1/2 变换的复杂度 O(N log N + N log(1/ε))。
   - **差异**：经典算法，非学习；本工作的可微层建立在它（及 FINUFFT 库）之上。

6. **Greengard, L., Lee, J.-Y. (2004).** *Accelerating the Nonuniform Fast Fourier Transform.* *SIAM Review* 46(3), 443–456.
   - **相关性**：NUFFT 的标准综述/误差分析；高斯 gridding 加速。
   - **差异**：作为 NUFFT 误差与复杂度的理论引用。

7. **Keiner, J., Kunis, S., Potts, D. (2009).** *Using NFFT 3 — a software library for various nonequispaced fast Fourier transforms.* *ACM Trans. Math. Software* 36, 1–30.
   - **相关性**：NFFT3 库；与 finufft 同为常用非均匀 FFT 实现。
   - **差异**：库论文；可在"实现选型"处对比 finufft vs NFFT3。

> **小结**：公开文献中"可微非均匀谱导数"几乎是空白——Lingsch et al. (2024) 是最接近的思想先驱但停在算子学习/周期情形；用户的 finufft-PINN 是周期 PINN 残差侧的直接前驱。**"非周期 BC + 非均匀点 + 可微谱导数"三要素交集无人占据。**

---

## 七、研究空白分析（Gap Analysis）

### 7.1 已有工作解决了什么（不要声称的 novelty）

| 已有能力 | 代表工作 | 已解决程度 |
|---|---|---|
| 可微傅里叶谱卷积 / 算子学习 | FNO, DeepONet, PINO, GINO, Geo-FNO | 成熟，但限周期 + 均匀网格 |
| 非周期函数的傅里叶精确微分 | FC-PINO（FC 延拓）、IBSE | 可行，但需解延拓线性系统、限均匀网格、算子学习框架 |
| 可微切比雪夫基用于 PDE | SNO, CSNN, ChebNet, CFNN, SEDONet | 可行，但限均匀配置节点、无 NUFFT |
| 正交多项式基严格承载三类非周期 BC | OPNO（Dirichlet/Neumann/Robin） | 理论+实验可行，但均匀节点、算子学习 |
| PINN 硬/软 BC 施加 | Sukumar & Srivastava 距离函数；Rao et al. 硬约束；BCGP | 成熟工具，默认配 autograd |
| 非均匀点上的 PINN 数值微分 | DT-PINN (RBF-FD), RBFDQ-PINN, SPINN | 成熟，但是局部阶精度，非谱精度 |
| 非均匀点上的可微谱求值 | Lingsch et al. (ICML 2024) | 首次，但周期基、算子学习、未做三类 BC |
| 可微非均匀谱导数用于 PINN 残差 | 用户前期 finufft-PINN | 已解决周期情形 |

### 7.2 本工作真正占据的空白

1. **"切比雪夫基 × NUFFT"的可微混合层尚无先例。** 现有谱-神经网络要么纯傅里叶（FNO/傅里叶 PINN，周期），要么纯切比雪夫（SNO/CSNN/OPNO，均匀节点）。**"非周期端用切比雪夫、任意点端用 NUFFT、两端都接 autograd"这一具体组合在公开文献中未见。** 这是第一 novelty。

2. **非周期 BC（Dirichlet/Neumann/Robin）× 非均匀空间采样 × 谱精度三者交集无人占据。** OPNO 处理三类 BC 但均匀；Lingsch/NFS 处理非均匀点但周期；DT-PINN/RBF-DQ 处理非均匀点但是局部阶精度。**把这三点同时满足、并在光滑解上证明谱（指数）收敛，是第二 novelty。**

3. **绕过 FC 延拓的条件数痛点。** FC-PINO 路线需要为非周期函数求解一个（轻微病态的）延拓系统才能回到 FFT；本工作用切比雪夫基天然吸收边界跳变，**在 JCP 审稿人会关心的"误差界 / 条件数 / 计算复杂度"三个维度上提供新的理论对照**，可作为与 FC-PINO 的直接算法对比。

4. **与用户前期工作的清晰增量。** 前期 finufft-PINN = 周期 + 非均匀点；本工作 = 非周期 + 非均匀点 + 切比雪夫耦合。只要在 Related Work 中把这条线讲清楚，novelty 边界就清晰、不与前人重叠。

### 7.3 需要警惕的"近邻"与必须正面回应的审稿质疑

- **"与 OPNO (arXiv 2206.12698) 区别在哪？"** ——OPNO 是算子学习 + 均匀正交配置；本工作是单实例 PINN/可微层 + 任意非均匀点。务必在稿中设一节直接对比。
- **"与 FC-PINO (arXiv 2211.15960) 区别在哪？"** ——FC 路线 = 延拓换周期；本工作 = 切比雪夫原生非周期。建议加数值实验：相同非周期问题上比较方程损失随 N 的收敛阶与求解延拓系统的额外开销。
- **"与 Lingsch et al. ICML 2024 区别在哪？"** ——它是非均匀点上的周期谱算子；本工作补非周期 BC。需引用并承认其在先。
- **"为什么不用 RBF-FD (DT-PINN)？"** ——RBF-FD 是局部多项式阶，NUFFT 是全局谱阶；建议在光滑解收敛阶实验中正面展示两者差异，并承认 RBF-FD 在剧烈非均匀/局部加密时的鲁棒优势。
- **"NUFFT 的导数精度和自适应网格怎么处理？"** ——引 Dutt–Rokhlin / Greengard–Lee 误差理论，并讨论 finufft 类型 1/2 的用法、核宽度、over-sampling 与 autograd 反向传播的数值稳定性。

### 7.4 JCP 投稿建议

- JCP 偏好：**明确的误差界定理、计算复杂度分析、收敛阶数值验证、与经典方法的公平对比**。建议 Related Work 按本报告六节组织，但正文 prose 时把"FC-PINO / OPNO / DT-PINN / 用户前期 finufft-PINN"作为四个最直接的对比锚点。
- 理论贡献建议至少覆盖：(a) 切比雪夫–NUFFT 混合算子的一致性/稳定性；(b) 对光滑解的指数收敛阶；(c) 三类 BC 硬约束的相容性；(d) 反向传播可微性。
- 数值实验建议基线：vanilla PINN (autograd)、DT-PINN (RBF-FD)、FC-PINO / FC 延拓、OPNO、用户前期周期 finufft-PINN（在周期问题上对齐）。

---

## 附：核心参考文献一览（按引用顺序，附链接）

1. Li et al., *FNO*, ICLR 2021 — https://arxiv.org/abs/2010.08895
2. Lu et al., *DeepONet*, MLST 2021 — https://arxiv.org/abs/1910.03193
3. Wang, Wang, Perdikaris, *PI-DeepONet*, Sci. Adv. 2021 — https://www.science.org/doi/10.1126/sciadv.abi8605
4. Li et al., *PINO*, ACM/JDS 2024 — https://dl.acm.org/doi/10.1145/3648506
5. Maust et al. / Ganeshram et al., *FC-PINO*, NeurIPS 2022 / arXiv 2024 — https://arxiv.org/abs/2211.15960
6. Li et al., *Geo-FNO*, JMLR 2023 — https://jmlr.org/papers/volume24/23-0064/23-0064/
7. Li et al., *GINO*, NeurIPS 2023 — https://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
8. Lingsch et al., *Beyond Regular Grids*, ICML 2024 — https://arxiv.org/abs/2305.19663
9. Barnett, Magland, af Klinteberg, *FINUFFT*, SIAM J. Sci. Comput. 2019 — https://arxiv.org/abs/1808.06736
10. Fanaskov & Oseledets, *SNO*, 2022 — https://arxiv.org/abs/2205.10573
11. Yin, Ling, Ying, *CSNN*, 2024 — https://arxiv.org/abs/2407.03347
12. Abid & San, *SEDONet*, 2025 — https://arxiv.org/abs/2512.09165
13. Tang, Li, Yu, *ChebNet*, CICP 2020 — https://arxiv.org/abs/1911.05467
14. Xu, Chen, Xiu, *CFNN*, 2024 — https://arxiv.org/abs/2409.19135
15. Liu et al., *OPNO/SOL*, 2022 — https://arxiv.org/abs/2206.12698
16. Du, Chalapathi, Krishnapriyan, *Neural Spectral Methods*, ICLR 2024
17. Bruno & Paul, *2D Fourier Continuation*, 2020 — https://arxiv.org/abs/2010.03901
18. Fontana, Bruno, Mininni, Dmitruk, *FC SPECTER*, 2020 — https://arxiv.org/abs/2002.01392
19. Stein, Guy, Thomases, *IBSE*, JCP 2020 — https://escholarship.org/uc/item/qt27n8r19q
20. Pfeiffer et al., *multidomain spectral*, CPC 2003 — https://arxiv.org/abs/gr-qc/0202096
21. Canuto & Funaro, *Schwarz spectral*, SINUM 1988
22. van der Sande, Appelö, Albin, *FC-DG*, 2021 — https://arxiv.org/abs/2105.00123
23. Raissi, Perdikaris, Karniadakis, *PINN*, JCP 2019 — https://arxiv.org/abs/1711.10561
24. Sukumar & Srivastava, *exact BC via distance functions*, CMAME 2022 — https://arxiv.org/abs/2104.08426
25. Rao, Sun, Liu, *hard-constraint PINN*, 2021 — https://arxiv.org/abs/2006.08472
26. Dalton et al., *BCGP*, JMLR 2024 — https://dl.acm.org/doi/abs/10.5555/3722577.3722849
27. Gladstone, Nabian, Meidani, *FO-PINNs*, NeurIPS WS 2022 — https://arxiv.org/abs/2210.14320
28. Xia, Böttcher, Chou, *spectrally adapted PINNs*, MLST 2023 — https://iopscience.iop.org/article/10.1088/2632-2153/acd0a1
29. Sahli Costabal, Pezzuto, Perdikaris, *Δ-PINNs*, CMAME 2023 — https://arxiv.org/abs/2209.03984
30. Sharma & Shankar, *DT-PINN*, NeurIPS 2022
31. Xiao et al., *RBFDQ-PINN*, Phys. Fluids 2024
32. Ramabathiran & Ramachandran, *SPINN*, 2021 — https://arxiv.org/abs/2102.13037
33. Hedayatrasa et al., *k-PINN*, 2024 — https://arxiv.org/abs/2404.03966
34. Yu, Qi, Oseledets, Chen, *Fourier Spectral PINN*, 2024 — https://arxiv.org/abs/2408.16414
35. Lin et al., *NFS*, 2022 — https://arxiv.org/abs/2212.04689
36. White et al., *exact differentiation on arbitrary geometries*, NeurIPS 2023 — https://openreview.net/forum?id=qd7q9lB5hY
37. Lin et al., *mGNO*, 2025 — https://arxiv.org/abs/2504.08277
38. Dutt & Rokhlin, *NUFFT II*, ACHA 1995
39. Greengard & Lee, *accelerating NUFFT*, SIAM Rev. 2004
40. Keiner, Kunis, Potts, *NFFT3*, TOMS 2009

> 说明：以上 arXiv 编号均由学术检索命中并核对标题/作者一致；少数经典文献（Canuto–Funaro 1988、Dutt–Rokhlin 1995、Greengard–Lee 2004、Keiner et al. 2009）请在定稿时按 BibTeX 库补全卷期页码。用户前期 finufft-PINN 的正式引用信息请按已发表版本替换。
