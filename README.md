# 青年入乡间歇观察的可逆状态转移分析

本项目提供间歇面板与集合状态观测相关资料的分析：资料修订读取、基础统计摘要，以及连续时间多状态面板分析。

app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；app/baseline.py 承载当前业务计算；app/service.py 输出摘要及输入摘要哈希。
数据字段及现有接口见 [资料契约](docs/data-contract.md)。fixtures/sample.json 是合成小数据，不对应真实村庄或个人。
运行环境为 Python 3.14，现有模块仅使用标准库。

读取并计算当前摘要：

    python -m app.service fixtures/sample.json

## 连续时间多状态分析

app/mservice.py 在既有访问明细之上拟合分段恒定强度的连续时间马尔可夫模型：短住、常驻、离村之间按允许转移边往返，离村不自动成为吸收终点；观察间隔内用矩阵指数对所有允许的中间路径求和，不把相邻状态差当成一次精确转移。“无法确定”按允许状态集合求和，不是新状态；首访之前不计暴露，末访之后不补写。

    python -m app.mservice fixtures/sample.json
    python -m app.mservice fixtures/sample.json scheme.json

强度方案（第二参数，可省略）可配置 cutpoints、allowed_edges、forbidden_edges、occupancy_states、prediction（from_day/to_day/step/start）与 optimizer（max_iter/restarts）。输出包含各分段转移强度及诊断、未来在村比例预测、逐跨度支持（哪几段随访支持哪种流动解释）与数据支持范围；边界解、未收敛与不可识别参数显式标记，不伪装成精确估计。结果通过 input_digest 与 scheme_digest 绑定同一版观察和强度方案，资料修订后 digest 变化即另起一版。口径与输出字段见 [资料契约](docs/data-contract.md)。

运行当前模块检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests
