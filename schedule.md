# CUDA 学习计划（2026-04-13 至 2026-07-05）

run cmake -S . -B out/clangd -DPython3_EXECUTABLE=/home/ljh/miniconda3/envs/AI_env/bin/python3

## 计划假设

- 起始日期默认从 `2026-04-13` 开始，连续 12 周，共 84 天。
- 默认环境以本仓库说明为准：WSL2 Ubuntu 24.04、zsh、`conda` 环境 `AI_env`、RTX 4060 Laptop GPU。
- 每次开始前固定执行：`conda activate AI_env`，然后按需设置 `export TORCH_CUDA_ARCH_LIST=Ada`。
- 每天的固定节奏保持一致：先读 README 和源码，再运行脚本或二进制，最后记录 3 件事：正确性、速度、一个关键优化点。
- 周日不引入新模块，只做复盘、重跑、补缺和口头复述；总结优先记在你自己的笔记里，不要求提交到仓库。

## 第 1 周：环境、仓库总览、基础逐元素算子

- [OK] `2026-04-13（周一）`：完成环境自检，命令建议用 `nvidia-smi`、`nvcc --version`、`python3 -c "import torch; print(torch.__version__, torch.version.cuda)"`；通读 `README.md` 里的 `Contents`、`200+ CUDA Kernels`、`HGEMM`、`FA2-MMA` 四部分；验收：能说清仓库学习路径是 `Easy -> Medium -> Hard -> Hard+ -> Hard++`。
- [ ] `2026-04-14（周二）`：阅读 `kernels/elementwise/README.md`、`kernels/elementwise/elementwise.cu`、`kernels/elementwise/elementwise.py`；重点看 `f32`、`float4`、`f16x2`、`f16x8 pack` 四类实现；验收：能解释“向量化加载”在这个目录里具体体现在哪些类型上。
- [ ] `2026-04-15（周三）`：运行 `cd kernels/elementwise && python3 elementwise.py`；对照输出记录哪几个版本更快，为什么 `pack` 版本通常更占优；验收：写出一次“访存合并 + 向量化 + 数据类型”三者关系的小结。
- [ ] `2026-04-16（周四）`：阅读并运行 `kernels/relu/relu.cu`、`kernels/relu/relu.py`，再对照 `kernels/sigmoid/sigmoid.cu`、`kernels/sigmoid/sigmoid.py`；重点关注激活函数里“计算简单但容易受访存限制”的特点；验收：能说出 ReLU 和 Sigmoid 在 kernel 复杂度上的主要差异。
- [ ] `2026-04-17（周五）`：阅读并运行 `kernels/swish/swish.cu`、`kernels/swish/swish.py` 与 `kernels/elu/elu.cu`、`kernels/elu/elu.py`；重点比较包含额外数学运算时，吞吐瓶颈是否从纯访存向算术侧偏移；验收：能指出这两个目录里最值得复用的代码结构。
- [ ] `2026-04-18（周六）`：阅读并运行 `kernels/hardswish/hardswish.cu`、`kernels/hardswish/hardswish.py` 与 `kernels/hardshrink/hardshrink.cu`、`kernels/hardshrink/hardshrink.py`；把本周见过的激活 kernel 按“分支多寡、向量化程度、是否容易合并访存”分类；验收：完成一张你自己的分类表。
- [ ] `2026-04-19（周日）`：不看新目录，回头重跑 `elementwise` 和任意两个激活目录；只做复盘，要求你口头复述一次线程索引、边界判断、向量化加载这三件事在本周代码中的落点；验收：不用看代码也能说出 `elementwise` 主体逻辑。

## 第 2 周：Reduce、转置、基础访存优化

- [ ] `2026-04-20（周一）`：阅读 `kernels/reduce/README.md` 和 `kernels/reduce/block_all_reduce.cu`、`kernels/reduce/block_all_reduce.py`；先把 `warp reduce`、`block reduce`、`block all reduce` 的层次关系理顺；验收：能画出从线程内累加到 block 内汇总的过程。
- [ ] `2026-04-21（周二）`：运行 `cd kernels/reduce && python3 block_all_reduce.py`；重点对比 `fp16 acc`、`fp32 acc`、`bf16`、`fp8`、`i8` 的输出和耗时；验收：能说明“acc dtype 为什么会影响精度和速度”。
- [ ] `2026-04-22（周三）`：阅读 `kernels/mat-transpose/README.md`、`kernels/mat-transpose/mat_transpose.cu`、`kernels/mat-transpose/mat_transpose.py`；重点看基础版、`float4` 版、shared memory 版的递进；验收：能说清楚转置为什么是观察 coalesced access 的好例子。
- [ ] `2026-04-23（周四）`：继续看 `kernels/mat-transpose/mat_transpose_cute.cu`；运行 `cd kernels/mat-transpose && python3 mat_transpose.py`；重点理解 `shared_bcf` 和 `swizzled` 版本分别在解决什么问题；验收：能解释 bank conflict 和 transpose 的关系。
- [ ] `2026-04-24（周五）`：阅读并运行 `kernels/nms/nms.cc`、`kernels/nms/nms.cu`、`kernels/nms/nms.py`；把它当成一个“规则判断多、访存不如 GEMM 规整”的例子看；验收：能说出 NMS 和 elementwise/reduce 在并行模式上的差别。
- [ ] `2026-04-25（周六）`：阅读并运行 `kernels/histogram/histogram.cu`、`kernels/histogram/histogram.py`；重点关注冲突、原子操作、统计型 kernel 的典型代价；验收：能说出 histogram 为什么天然容易出现竞争。
- [ ] `2026-04-26（周日）`：复盘本周所有目录；要求你把“合并访存、shared memory、bank conflict、原子操作”四个概念分别用一个仓库例子绑定起来；验收：能给每个概念配一个具体文件路径。

## 第 3 周：Softmax、Norm、Embedding、RoPE

- [ ] `2026-04-27（周一）`：阅读 `kernels/softmax/README.md`、`kernels/softmax/softmax.cu`、`kernels/softmax/softmax.py`；重点理解 `max -> exp -> sum -> normalize` 这条链；验收：能说清 softmax 为什么本质上是“多阶段 reduce + elementwise”。
- [ ] `2026-04-28（周二）`：运行 `cd kernels/softmax && python3 softmax.py`；对比输出与 PyTorch 结果，关注数值稳定性处理；验收：能解释“先减 max”这一步为什么必要。
- [ ] `2026-04-29（周三）`：阅读并运行 `kernels/layer-norm/README.md`、`kernels/layer-norm/layer_norm.cu`、`kernels/layer-norm/layer_norm.py`；重点看 `f16`、`f32 acc`、`pack` 版本；验收：能描述 LayerNorm 里的两次统计量计算。
- [ ] `2026-04-30（周四）`：阅读并运行 `kernels/rms-norm/README.md`、`kernels/rms-norm/rms_norm.cu`、`kernels/rms-norm/rms_norm.py`；把 RMSNorm 和 LayerNorm 的公式、访存和实现差异并排比较；验收：能说出 RMSNorm 为什么省掉了均值项。
- [ ] `2026-05-01（周五）`：阅读并运行 `kernels/embedding/README.md`、`kernels/embedding/embedding.cu`、`kernels/embedding/embedding.py`；重点把 embedding 看成“索引驱动的 gather”问题；验收：能说明它和连续访存的区别。
- [ ] `2026-05-02（周六）`：阅读并运行 `kernels/rope/README.md`、`kernels/rope/rope.cu`、`kernels/rope/rope.py`；重点理解 RoPE 在仓库里是怎么做最小实现的；验收：能说出 RoPE 和纯 elementwise 的联系与区别。
- [ ] `2026-05-03（周日）`：只复盘，不开新坑；把 Softmax、LayerNorm、RMSNorm、Embedding、RoPE 串成一个最小 Transformer 数据流；验收：可以口头走完一次 `embedding -> rope -> attention 前处理 -> norm` 的顺序。

## 第 4 周：GEMV 入门与 SGEMM 预热

- [ ] `2026-05-04（周一）`：阅读并运行 `kernels/sgemv/README.md`、`kernels/sgemv/sgemv.cu`、`kernels/sgemv/sgemv.py`；重点看 `k32`、`k128 float4`、`k16` 三种实现；验收：能解释 GEMV 为什么比 GEMM 更容易先学。
- [ ] `2026-05-05（周二）`：阅读并运行 `kernels/hgemv/README.md`、`kernels/hgemv/hgemv.cu`、`kernels/hgemv/hgemv_cute.cu`、`kernels/hgemv/hgemv.py`；重点看 half、CuTe、Tensor Core 三条分支；验收：能说出 HGEMV 比 SGEMV 多了哪些 dtype 和硬件特性。
- [ ] `2026-05-06（周三）`：把 `sgemv` 和 `hgemv` 放在一起比较；专门看一次 `PyTorch bindings` 是怎么接出来的；验收：你要能解释“算子主体”和“Python 绑定”在目录里的边界。
- [ ] `2026-05-07（周四）`：阅读 `kernels/sgemm/README.md` 的 `Supported Matrix`、`0x00 说明`、`目前性能` 三段；先不深挖代码，先建立 SGEMM 的优化地图；验收：能列出 README 中提到的至少 8 个优化关键词。
- [ ] `2026-05-08（周五）`：阅读 `kernels/sgemm/sgemm.cu`，重点看 `naive`、`sliced_k`、`thread tile 8x8` 这些基础版本；验收：能说出 block tile 和 thread tile 的区别。
- [ ] `2026-05-09（周六）`：运行 `cd kernels/sgemm && python3 sgemm.py`；先建立基准，不急着追每一行代码；验收：能从输出里找出“基础版、优化版、cuBLAS、TF32/WMMA”几类结果。
- [ ] `2026-05-10（周日）`：复盘 `GEMV -> GEMM` 的思维跳转；要求你自己总结一次“为什么 GEMM 是 CUDA 学习主线”；验收：不用看 README，也能把 SGEMM 优化路径按顺序说一遍。

## 第 5 周：SGEMM 深入，双缓冲与 WMMA

- [ ] `2026-05-11（周一）`：精读 `kernels/sgemm/README.md` 中 `共享内存 Bank Conflicts` 段落，并回到 `sgemm.cu` 对照 shared memory 布局；验收：能用自己的话解释 bank conflict 是如何拖慢 GEMM 的。
- [ ] `2026-05-12（周二）`：精读 `kernels/sgemm/README.md` 中 `双缓冲 Double Buffers` 段落，配合 `kernels/sgemm/sgemm_async.cu`；验收：能说明“为什么主循环从 bk = 1 开始”。
- [ ] `2026-05-13（周三）`：阅读 `kernels/sgemm/sgemm_wmma_tf32_stage.cu`；重点理解 `WMMA`、`TF32`、`stage`、`swizzle` 这些概念首次如何落地；验收：能区分 CUDA Cores 路线和 Tensor Cores 路线。
- [ ] `2026-05-14（周四）`：重跑一次 `cd kernels/sgemm && python3 sgemm.py`；本次只盯着每一种实现对应 README 里的哪种优化；验收：能把输出名称和代码版本一一对应上。
- [ ] `2026-05-15（周五）`：今天不跑新东西，专门手画一张 `gmem -> smem -> reg -> compute -> writeback` 的 SGEMM 数据流图；验收：图上要明确标出 block、warp、thread 三层。
- [ ] `2026-05-16（周六）`：选 `sgemm.cu` 或 `sgemm_async.cu` 中一个最顺眼的 kernel，做一次逐段阅读；关注索引计算、load/store、同步位置；验收：能指出最关键的两个 `__syncthreads()` 为什么必须存在或可以减少。
- [ ] `2026-05-17（周日）`：复盘 SGEMM 一整周；只回答三个问题：哪里在做 tile，哪里在做 overlap，哪里还会有 bank conflict；验收：你能不用代码复述一遍 SGEMM 优化主线。

## 第 6 周：Swizzle、Nsight、PTX/SASS

- [ ] `2026-05-18（周一）`：阅读 `kernels/swizzle/README.md`；进入 `cd kernels/swizzle && make` 构建默认二进制；验收：知道这个目录的目标是“通过 swizzle 消除或显著降低 bank conflicts”。
- [ ] `2026-05-19（周二）`：运行 `cd kernels/swizzle && python3 print_swizzle_layout.py --logical-col 64 --show-logical-col`；重点理解逻辑列和 shared memory 物理布局的映射；验收：能解释 ZigZag/Swizzle 在布局层面的作用。
- [ ] `2026-05-20（周三）`：运行 `cd kernels/swizzle && ./hgemm_mma_swizzle.bin 4096 4096 4096 1 10`；看有无 swizzle 时性能差异；验收：能把“性能变化”和“bank conflict 减少”联系起来。
- [ ] `2026-05-21（周四）`：如果本机已有 Nsight Compute，就按 `kernels/swizzle/README.md` 里的命令尝试 `ncu --metrics ... ./hgemm_mma_swizzle.bin ...`；验收：知道至少一个可以直接观察 shared memory conflict 的指标名。
- [ ] `2026-05-22（周五）`：阅读 `kernels/nvidia-nsight/README.md`，按文档用 `nvcc -arch=sm_89 -o relu.bin --generate-line-info -g relu.cu` 的思路编一次简单样例；验收：明白“为什么要带 line info”。
- [ ] `2026-05-23（周六）`：按 README 试一次 `nsys` 和 `ncu` 的最小流程；然后直接看 README 中给出的 PTX/SASS 示例，理解 `LDG.E.128`、`STG.E.128`、`HMNMX2` 这些指令的意义；验收：能说出 pack 版本为什么会出现更理想的 load/store 指令。
- [ ] `2026-05-24（周日）`：复盘这一周，只整理“什么时候用 padding，什么时候用 swizzle，什么时候需要 profile”三条经验；验收：每条经验都能配一个仓库目录。

## 第 7 周：HGEMM 总览、子模块、WMMA/MMA 跑通

- [ ] `2026-05-25（周一）`：通读 `kernels/hgemm/README.md` 前半部分，先看 `HGEMM CUDA Kernels` 列表、`Prerequisites`、`Installation`、`Python Testing`；验收：能说出 `WMMA`、`MMA PTX`、`CuTe` 三条线在这个目录里的角色。
- [ ] `2026-05-26（周二）`：如果本地还没初始化子模块，执行 `git submodule update --init --recursive --force`；然后浏览 `kernels/hgemm/` 目录结构；验收：知道 `naive`、`wmma`、`cutlass`、`pybind`、`bench`、`tools` 分别放什么。
- [ ] `2026-05-27（周三）`：运行 `cd kernels/hgemm && python3 hgemm.py --wmma`；先建立 WMMA 路线的直觉；验收：能从输出中区分“默认 WMMA kernel”和“cuBLAS 对照”。
- [ ] `2026-05-28（周四）`：运行 `cd kernels/hgemm && python3 hgemm.py --mma`；本次关注 MMA 路线和 WMMA 路线的差别；验收：能说出为什么仓库作者会同时保留这两类实现。
- [ ] `2026-05-29（周五）`：阅读 `kernels/hgemm/naive/hgemm.cu`、`kernels/hgemm/naive/hgemm_async.cu` 以及 `kernels/hgemm/wmma/hgemm_wmma.cu`、`kernels/hgemm/wmma/hgemm_wmma_stage.cu`；验收：能对齐“naive -> async -> wmma stage”的递进关系。
- [ ] `2026-05-30（周六）`：阅读 `kernels/hgemm/pybind/hgemm.cc`、`kernels/hgemm/setup.py`、`kernels/hgemm/tools/utils.py`；看懂 Python 调用链是怎么落到 C++/CUDA 的；验收：能画出 `python -> pybind -> kernel` 最小调用路径。
- [ ] `2026-05-31（周日）`：只复盘 HGEMM 第一周；要求你把 README 里那张优化关键词表重新按“访存、并行映射、Tensor Core、布局优化”四类重组一次；验收：四类都至少能写出两个关键词。

## 第 8 周：HGEMM 优化细节、CuTe 与 C++ Benchmark

- [ ] `2026-06-01（周一）`：精读 `kernels/hgemm/README.md` 里的 `Performance Optimization Notes`，重点看 `SMEM Padding`、`Double Buffers`、`Tile MMA/Warp`、`SMEM Swizzle/Permuted`；验收：能说出这些优化各自针对哪类瓶颈。
- [ ] `2026-06-02（周二）`：运行 `cd kernels/hgemm && python3 hgemm.py --wmma-all`；不要急着追求最优数值，先建立“同一种思路在不同参数下的表现差异”；验收：记下 3 个你觉得最稳定的 WMMA 版本。
- [ ] `2026-06-03（周三）`：运行 `cd kernels/hgemm && python3 hgemm.py --mma-all`；重点观察 MMA 路线和昨天的 WMMA 路线谁在你的显卡上更稳；验收：能根据结果给出一个初步判断。
- [ ] `2026-06-04（周四）`：运行 `cd kernels/hgemm && python3 hgemm.py --cute-tn --no-default`；今天只看 CuTe 版本；验收：能解释 CuTe 路线为什么更适合表达复杂 layout/swizzle。
- [ ] `2026-06-05（周五）`：如果编译链完整，运行 `cd kernels/hgemm && make && ./hgemm_mma_stage.bin`；把 Python benchmark 和 C++ benchmark 的差异感受一次；验收：能说出“为什么 C++ 二进制通常略快于 Python 调用”。
- [ ] `2026-06-06（周六）`：阅读 `kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu`；把昨天看到的 CuTe 输出和今天的源码联系起来；验收：能定位到代码里显式表达 layout/swizzle 的位置。
- [ ] `2026-06-07（周日）`：复盘 HGEMM 第二周；今天只做一件事：整理一份你自己的 HGEMM 优化路线图，从最朴素版本排到最强版本；验收：路线图中至少包含 8 个节点。

## 第 9 周：CuTe 与布局表达专项

- [ ] `2026-06-08（周一）`：阅读 `kernels/cutlass/cute/vector_add.cu`；把它当作 CuTe 语法和布局表达的最小入口；验收：你至少要看懂它如何定义基本 tile 或 layout。
- [ ] `2026-06-09（周二）`：阅读 `kernels/cutlass/cute/mma_tile_tex.cc`；重点关注 tile 组织和 MMA 抽象；验收：能说出这类代码和手写索引版 CUDA kernel 最大的风格差异。
- [ ] `2026-06-10（周三）`：回看 `kernels/mat-transpose/mat_transpose_cute.cu`；今天只关心 CuTe 在一个相对简单算子里如何表达寄存器、shared memory、swizzle；验收：能把 transpose 和 hgemm 两个目录里的 CuTe 用法联系起来。
- [ ] `2026-06-11（周四）`：再读一次 `kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu`；这次重点找和昨天 transpose 目录相似的 layout 语义；验收：能指出至少两处“同一思想，不同算子”的写法。
- [ ] `2026-06-12（周五）`：阅读 `kernels/hgemm/tools/print_swizzle_layout.py` 与 `kernels/swizzle/print_swizzle_layout.py`；重点理解这些脚本如何帮助你把抽象 layout 还原成可视化直觉；验收：能自己解释一次脚本输出。
- [ ] `2026-06-13（周六）`：今天不跑大 benchmark，只做概念串联：`transpose swizzle -> hgemm swizzle -> cute layout`；验收：能在纸上画出“逻辑矩阵”和“shared memory 物理布局”的区别。
- [ ] `2026-06-14（周日）`：复盘 CuTe 周；你要确认自己不是在记 API，而是在理解“布局表达 + 数据移动 + MMA 映射”；验收：能用不超过 5 句话说明你当前对 CuTe 的理解。

## 第 10 周：FlashAttention 基础版本

- [ ] `2026-06-15（周一）`：通读 `kernels/flash-attn/README.md` 的总览、Kernel 分类、`Prerequisites`、`Python Testing`；先把 `Split KV`、`Split Q`、`Shared KV`、`Shared QKV`、`Tiling QK/QKV` 五类方案记住；验收：能说出每类方案主要在减少什么开销。
- [ ] `2026-06-16（周二）`：如果环境里还没有官方依赖，按需执行 `python3 -m pip install flash-attn --no-build-isolation`；然后运行 `cd kernels/flash-attn && python3 flash_attn_mma.py --D 64`；验收：确认默认 case 可以跑通并输出多种 kernel 对比。
- [ ] `2026-06-17（周三）`：阅读 `kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu`、`kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu` 或 README 中对应的 `Split KV`、`Split Q` 描述；验收：能解释为什么 `Split Q` 通常比 `Split KV` 更优。
- [ ] `2026-06-18（周四）`：阅读 `kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu`、`kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu`；重点理解“共享 SRAM”带来的 occupancy 变化；验收：能说出 `Shared KV` 与 `Shared QKV` 的主要差别。
- [ ] `2026-06-19（周五）`：运行 README 里的代表性命令：`cd kernels/flash-attn && python3 flash_attn_mma.py --B 1 --H 8 --D 64 --N 8192 --iters 10 --torch`；验收：能从输出中指出哪个版本在这个 case 下最好。
- [ ] `2026-06-20（周六）`：运行另一组代表性命令：`cd kernels/flash-attn && python3 flash_attn_mma.py --B 1 --H 48 --D 64 --N 8192 --iters 10 --torch`；对比昨天和今天的差异，体会规模变化对最优 kernel 的影响；验收：能总结出一个“规模变化 -> 最优实现可能变化”的例子。
- [ ] `2026-06-21（周日）`：本周复盘只做数据流整理；把 `Q/K/V` 在 `Split KV`、`Split Q`、`Shared KV`、`Shared QKV` 四类方案里的移动路径画出来；验收：能不用 README 复述四类方案。

## 第 11 周：FlashAttention Tiling、Swizzle、长维度

- [ ] `2026-06-22（周一）`：阅读 `kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu` 及其相关说明；重点理解 `QK Fine-grained Tiling` 如何把 SRAM 复杂度压下来；验收：能说明为什么这种方法更适合更大 `D`。
- [ ] `2026-06-23（周二）`：阅读 `kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu`；把它和前一天的 `tiling_qk` 做对比；验收：能说出 `tiling_qkv` 进一步降低了哪些存储压力。
- [ ] `2026-06-24（周三）`：运行 `cd kernels/flash-attn && python3 flash_attn_mma.py --B 1 --H 8 --N 8192 --iters 10 --show-all --sdpa --D 512`；重点观察 `D=512` 时 `SDPA` 与 `QK Tiling/QKV Tiling` 的差异；验收：记录一组你机器上的真实输出。
- [ ] `2026-06-25（周四）`：阅读 `kernels/flash-attn/mma/swizzle/` 下的几个变体，先挑 `swizzle_q`、`swizzle_qk`、`swizzle_qkv` 三类看命名和布局含义；验收：能解释 swizzle 被引入到 FlashAttention 的目的。
- [ ] `2026-06-26（周五）`：今天做一次 focused benchmark，继续用 `flash_attn_mma.py` 跑你最熟的一个 `D=64` 或 `D=512` case，对比 stage1/stage2 以及带不带 swizzle 的版本；验收：你能说出 stage 增加到底解决了什么。
- [ ] `2026-06-27（周六）`：阅读 `kernels/flash-attn/tools/print_swizzle_layout.py` 与 `kernels/flash-attn/tools/utils.py`；把它和 `kernels/swizzle`、`kernels/hgemm/tools` 里的工具联系起来；验收：能理解仓库作者为什么反复写 layout 可视化工具。
- [ ] `2026-06-28（周日）`：复盘 FlashAttention 第二周；只回答三个问题：为什么要做 fine-grained tiling，为什么 stage 影响性能，为什么 swizzle 会和 shared memory 一起出现；验收：每个问题都能给一个源码文件作例子。

## 第 12 周：回放、抽象、形成自己的 CUDA 知识图谱

- [ ] `2026-06-29（周一）`：选一个你最熟的 `hgemm` 或 `flash-attn` case，再做一次 `ncu` 或 `nsys` 最小 profile；今天的目标不是学新命令，而是把 profile 输出和源码对应起来；验收：能指出一个最关键的性能指标。
- [ ] `2026-06-30（周二）`：从 `elementwise`、`reduce`、`sgemm`、`hgemm`、`flash-attn` 里各选一个你最喜欢的 kernel，挑其中 1 个做一次逐行精读；验收：你能完整解释这个 kernel 的线程映射和数据流。
- [ ] `2026-07-01（周三）`：回放基础模块，只跑 `elementwise`、`reduce`、`softmax`、`layer_norm`、`mat_transpose` 五个代表脚本；验收：确认你已经能快速识别“这是访存型还是规约型算子”。
- [ ] `2026-07-02（周四）`：回放高阶矩阵模块，只跑 `sgemv`、`hgemv`、`sgemm`、`hgemm`；验收：你要能从这些目录中总结出 `vectorize -> tile -> smem -> double buffer -> tensor core -> swizzle` 的链条。
- [ ] `2026-07-03（周五）`：回放 attention 模块，只跑 `flash_attn_mma.py` 里你最熟的两组参数；验收：能不用看 README 解释为什么最优 kernel 不一定固定。
- [ ] `2026-07-04（周六）`：整理你自己的 CUDA 优化检查清单，最少包含这 8 项：合并访存、向量化、共享内存、bank conflict、双缓冲、Tensor Core、warp/block tile、swizzle；验收：每一项都能配一个仓库目录或文件。
- [ ] `2026-07-05（周日）`：完成总复盘，给自己做一次闭卷检查：从仓库中任选一个新目录，你能否快速判断它主要在优化哪一层；然后决定下一阶段方向是继续深挖 `hgemm/flash-attn`，还是开始自己写一个新 kernel；验收：写出下一阶段 4 周目标，但不要求今天就动代码。

## 12 周结束后的判断标准

- 你已经能独立阅读本仓库大部分 `Easy`、`Medium` 和核心 `Hard` 目录，不再依赖 README 才能理解代码走向。
- 你已经能把常见优化手段和具体仓库文件一一对应，而不是只记名词。
- 你已经能解释至少一个 `SGEMM/HGEMM` kernel 和一个 `FlashAttention` kernel 的完整数据流。
- 你已经知道下一步该补的是哪一块：profile、CuTe、MMA PTX、还是自己动手改 kernel。
