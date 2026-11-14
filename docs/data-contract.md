# 资料与现有计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

visits 含 person、day、states，day 为相对研究起点的天数，states 为非空允许状态集合。相同person/day的多个有效记录取集合交，空交是资料冲突。state_names 是状态全集，allowed_edges 为直接转移边。cutpoints 给日历分段边界；每人条件于首个观察入组。baseline.analyze 按人整理相邻访问跨度，不作潜在转移推断。

各集合必要字段：{"visits": ["person", "day", "states"]}。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。

## 连续时间多状态分析

入口为app.mservice.run(payload, scheme=None)，命令行为`python -m app.mservice 资料.json [方案.json]`。模型是分段恒定强度的连续时间马尔可夫链，仅用标准库实现。

### 强度方案（scheme）

- cutpoints：分段边界，严格递增，首元素为研究起点；默认取metadata.cutpoints，缺省为[0]。分段数等于cutpoints长度，末段延伸到正无穷。
- allowed_edges：允许直接转移边，默认取metadata.allowed_edges；forbidden_edges从允许集合中移除（必须原本允许）。离村不自动成为吸收终点，是否可返回完全由边配置决定。
- occupancy_states：计入在村比例的状态，默认除“离村”外的全部状态。
- prediction：from_day（默认最大观察日）、to_day（默认from_day+180）、step（默认30）、start（{"state": 状态}或{"distribution": {状态: 权重}}，默认按各人末访允许集合等权的经验分布）。
- optimizer：max_iter（默认2000）、restarts（默认3），用于拟合诊断与测试。

### 模型口径

- 观察间隔的转移概率由生成矩阵的矩阵指数给出，跨分段时按时间序连乘各段矩阵指数，即对该区间内所有允许的中间路径求和；不把相邻状态差当成一次精确转移。
- “无法确定”是允许状态集合：前向递归对集合内状态求和，不增设新状态。
- 每人条件于首个观察入组：首访之前不计暴露，末访之后不补写；仅一次访问的人只贡献入组条件，不进入似然。
- 每个（分段, 允许边）一个自由强度参数，在对数尺度上优化（多起点Nelder-Mead）。

### 输出与诊断

- status：ok / not_converged / failed。未收敛或失败时估计与预测不应当作精确结果，warnings会说明。
- fit.parameters：逐参数给出强度估计、标准误（不可用时为null）、boundary（触及强度上下界）、support_spans/support_days（覆盖该分段的随访区间数与天数）、likelihood_range（值域探测的对数似然变化，近零即数据不含该参数信息）、identifiable。
- diagnostics：data_range（数据支持范围）、single_visit_persons、boundary_parameters、unidentifiable_parameters、信息矩阵特征值与covariance_available。
- span_support：把baseline.analyze的既有访问明细接到拟合结果——每段随访的对数似然贡献、跨越的分段、涉及的参数与该时点的过滤状态分布，用于定位哪几段随访支持哪种流动解释。
- prediction：未来各网格日的状态占据概率与在村比例；在村比例给出delta法标准误（协方差不可用时为null）；extrapolated标记超出数据支持范围的网格点。预测与拟合共用同一input_digest与scheme_digest，资料修订后digest变化即另起一版，不会与旧版观察或方案混淆。

### 关键假设

结果依赖观察时点可忽略假设：访视安排不携带潜在状态过程的信息；若离村者更难被访到，估计会有偏。稀疏观察不等于稳定留驻：两次访问之间可能发生未观察到的往返，强度反映的是观察间隔内的净流动，不能证明期间持续留驻。边界解、未收敛与不可识别参数都显式标记，不伪装成精确估计。
