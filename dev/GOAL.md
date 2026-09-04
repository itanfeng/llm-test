# GOAL — fused_infer_attention_score Skip 功能开发规约
> 功能：为 `fused_infer_attention_score` 算子实现 Skip（Blocked Attention Sparsity via Softmax Thresholding），在**不改变 tiling / block 维度 / pipeline 结构**前提下，通过“条件执行”跳过低贡献 attention block，降低时延。

---

## 0. 总目标与验收标准

### 0.1 总目标
在 `8.5.0-skip` 分支实现 Skip 稀疏优化，使算子在真实板端执行（910B2）时延显著下降，并确保数值正确性可验证、可回归、可复现。

### 0.2 验收标准（必须同时满足）
1. **功能正确性**
   - `fia_mla_decode.py` 在相同输入下输出与基线一致（允许误差阈值需明确，默认采用 `torch.allclose`）。
2. **Skip 生效可证明**
   - 运行日志可确认进入 Skip 分支逻辑（调试期允许 printf，最终提交必须关闭）。
   - 可调整阈值使稀疏率稳定达到目标（例如 50%）。
3. **性能有效**
   - 在相同测试条件下（同一板卡、同一输入、同一环境、同一脚本链路）`8.5.0-skip` 时延 < `8.5.0-base` 时延。
4. **约束遵守**
   - 不新增 TQue、不改变 TPipe stage 数、不修改 tiling 策略、不改变 block 维度、不改变 cube/vector 调度顺序。

---

## 1. 分支与目录

### 1.1 代码分支
- Baseline：`8.5.0-base`
- Skip：`8.5.0-skip`

### 1.2 关键脚本
- 远程交互脚本目录：`./scripts/dev`
  - 阅读 `README.ascend_remote.md` 掌握同步/编译/测试/拉回日志的使用方式
- 功能测试脚本（PyTorch 接口验证）：`./scripts/dev/fia_mla_decode.py`
  - 通过 `torch_npu.npu_fused_infer_attention_score` 触发算子

### 1.3 Kernel 相关源码（非量化 MLA 路径）
- 主模板：`./attention/common/op_kernel/arch32/fia_kernel_nonquant_mla.h`
- vector 模板：`./attention/common/op_kernel/arch32/fia_block_vec_nonquant_mla.h`
- cube 模板：`./attention/common/op_kernel/arch32/fia_block_cube_nonquant_mla.h`
- tiling：  
  - `./attention/common/op_host/fia_tiling_nonquant_mla.cpp`  
  - `./attention/common/op_host/fia_tiling_nonquant_mla.h`

---

## 2. 硬件模型（用于执行时序与内存流建模）
- 芯片：Ascend 910B2
- AICore：24 个
- 每个 AICore：
  - 1 × Cube 核（矩阵乘，L0A/L0B/L0C/L1）
  - 2 × Vector 核（向量算子，UB=192KB）
- 核间数据交换：通过 GM（HBM）读写

---

## 3. Skip 算法定义（Softmax Thresholding）

### 3.1 核心思想
复用在线 Softmax 中已计算的统计量（rowmax / running max），实时判断某个 attention block 的最大分数是否远小于当前 running max。若贡献近似为 0，则跳过该 block 的后续计算，减少：
- exp 计算
- Value block 从 GM/HBM 加载
- `P * V` 的矩阵乘

### 3.2 运行过程（按 Q block 行处理）
对每个 query block 行 `i`：
1. 初始化：运行最大值 `m_i`、累加器 `l_i`、输出 `O_i`
2. 遍历 key/value block 列 `j`：
   - 计算分数块：`S_ij = Q_i * K_j^T`
   - 计算局部最大：`m̃_i^(j) = rowmax(S_ij)`
   - 更新运行最大：`m_i^(j) = max(m_i^(j-1), m̃_i^(j))`
   - **Skip 判定（必须在 rowmax 后、exp 前）**  
     - 条件：`m̃_i^(j) - m_i^(j) < ln(λ)`  
     - 若成立：跳过该 block 的 exp / load V / P*V / l_i 更新
   - 非 Skip：  
     - `P = exp(S_ij - m_i^(j))`
     - `O_i += P_ij * V_j`
     - `l_i += sum(P)`
3. 归一化：`O_i = O_i / l_i`

---

## 4. 必做：基线 Kernel 执行时序建模（修改前置门禁）

> **在对 `8.5.0-skip` 做任何 kernel 修改之前，必须先在 `8.5.0-base` 完成建模报告。未完成报告禁止动 kernel。**

### 4.1 必读文件（基线分支）
- `fia_kernel_nonquant_mla.h`
- `fia_block_vec_nonquant_mla.h`
- `fia_block_cube_nonquant_mla.h`
- `fia_tiling_nonquant_mla.cpp`
- `fia_tiling_nonquant_mla.h`

### 4.2 报告输出要求
输出分析报告保存为：
- `./scripts/dev/REPORT.md`

报告必须包含以下四部分（缺一不可）：

#### (1) 循环结构建模
- Q block 循环层级（外层/内层）
- K/V block 循环层级（外层/内层）
- 每层循环对应 block 维度（M/N/K、head、block size 等）
- cube 与 vector 的调用顺序（先后关系）

#### (2) 执行阶段划分（严格按时间顺序）
明确列出并标注每个阶段：
- `S_ij` 计算阶段
- rowmax 阶段
- `m_i` 更新阶段
- exp 阶段
- `P*V` 阶段
- `l_i` 更新阶段
- 归一化阶段

对每个阶段标明：
- 所在文件
- 所在函数
- 所在循环层级（Q-loop / KV-loop）

#### (3) 内存流建模
- Q/K/V 的搬运路径（GM → L1 → L0 / UB）
- `S_ij` 存储位置（是否落 GM/UB/中间 buffer）
- `m_i / l_i` 存储位置与生命周期
- TQue 申请/释放位置（必须明确代码位置）
- flag/pipe 同步点（必须明确代码位置）

#### (4) Pipeline 与资源建模
- 当前 TPipe stage 数
- TQue 使用数量与用途
- 是否存在双缓冲（double buffer）
- cube 与 vector 是否并行、如何对齐（对齐点/同步点）

### 4.3 报告通过前的禁止项
- 禁止修改任何 kernel 代码（包括宏、buffer、pipe、que、算子流程）

---

## 5. Skip 插入点约束（硬性约束）

Skip 插入必须满足：
1. **位置约束**
   - 位于 rowmax 计算完成之后
   - 位于 exp 计算之前
2. **资源约束**
   - 不新增 TQue
   - 不改变 TPipe stage 数
3. **流水线约束**
   - 不破坏 cube/vector pipeline 对齐
4. **策略约束**
   - 不允许修改 tiling 策略
   - 不允许修改 block 维度
   - 不允许修改 cube/vector 调度顺序
5. **实现方式约束**
   - Skip 只能通过“条件执行（conditional execution）”实现
   - 不允许通过改变计算图结构/重排算子流程实现

---

## 6. 调试与性能对比流程（强制执行）

### 6.1 基线测量（8.5.0-base）
1. 切到 `8.5.0-base`
2. 运行 `ascend_fused_infer_attention_score_decode_cycle.sh` 完整链路：
   - 同步代码 → 编译算子 → 执行测试 → 拉回日志
3. 记录基线时延与环境信息（commit / device / driver / cann / 输入）

### 6.2 Skip 开发调试（8.5.0-skip）
1. 切到 `8.5.0-skip`
2. 实现 Skip 功能（允许加入 printf/日志用于定位）
3. 反复运行 `ascend_fused_infer_attention_score_decode_cycle.sh`：
   - 可能失败在编译阶段 / 执行阶段
   - 必须基于日志快速定位修复
4. 必须验证：
   - 确认运行在 `8.5.0-skip`（打印 commit 或版本信息）
   - 确认进入 Skip 新逻辑（打印 Skip 计数/比例）
   - 调整阈值使稀疏率达到目标（例如 50%）

### 6.3 Skip 最终测量（关闭调试打印）
1. 关闭所有 printf/调试日志
2. 再跑一次完整链路：同步→编译→测试→拉回日志
3. 记录最终时延，与基线对比
4. 若无收益：继续优化 Skip 逻辑（仍需遵守全部约束）

### 6.4 注意
1. 一次算子kernel编译时间很长，估计30分钟以上
2. 如果编译时日志一直没有更新可能是还在继续编译，通过进程进行确认，不要随便终止编译

---

## 7. Golden 参考实现（数值正确性闭环）

### 7.1 golden.py（基线对齐）
1. 在 `8.5.0-base` 完整编译与测试，保存 `fia_mla_decode.py` 的输入与输出（pt/npz 均可）
2. 编写纯 Python + Torch 的 `golden.py`：
   - 在相同输入下输出与 `torch_npu.npu_fused_infer_attention_score` 一致

### 7.2 golden_skip.py（Skip 对齐）
1. 在 `8.5.0-skip` 实现 Skip 后
2. 编写 `golden_skip.py`：
   - 在相同输入下输出与 `8.5.0-skip` 上的 `torch_npu.npu_fused_infer_attention_score` 一致
3. `golden.py / golden_skip.py` 均需可被 CI/脚本单独运行（可复现）

---

## 8. 错误码 Debug 规则（最低要求）

### 8.1 507046（stream sync timeout）
排查优先级：
1. 检查是否存在 **TQue 申请内存未释放** / 生命周期不匹配
2. 添加 printf/关键路径埋点，定位阻塞点（尽量只在调试分支开启）
3. 构造相同输入，使用孪生调试（CPU 侧调试）定位根因（可直接显示错误原因）

---

## 9. 交付物清单（最终必须存在）
- `./scripts/dev/REPORT.md`：基线执行模型分析报告（修改前完成）
- `./scripts/dev/golden.py`：基线对齐参考实现
- `./scripts/dev/golden_skip.py`：Skip 对齐参考实现
- 性能对比日志：base vs skip（同环境同输入）
- 提交记录：清晰标注每次关键改动目的（插入点、skip 统计、阈值策略等）

---