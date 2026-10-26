# 第三波·Pro：数字孪生预演增强（SepsisICU Harness Pro Demo）

脓毒症 ICU 智能体 POC 系列第 7 个概念验证。在 POC6「第三波：在线学习闭环」之上叠加四件
事——数字孪生（HemoSim）、反事实推演、Harness 自进化（不变量硬编码守卫）、可信度审计
（ECE / DCA / 精确 Shapley）。

> **免责声明（按需求不显示在界面，仅保留于文档）**：本 DEMO 为学术仿真原型。不接入任何真实
> 患者、医院监护、HIS/EHR 系统；全部病例、随机分配、结局反馈均为模拟生成。所有输出仅作
> 科研演示，不构成任何诊疗建议。

## 一、如何运行（PyCharm）

1. 用 PyCharm 打开本工程文件夹（`sepsis_harness_evolution_pro_demo`）。
2. 配置 Python 3.10+ 解释器，安装依赖：
   
   ```
   pip install -r requirements.txt
   ```
3. 右键运行 `main.py`（或终端 `python main.py`）。服务启动于
   **http://127.0.0.1:8000** ，浏览器自动可访问（PyCharm 内点控制台链接即可）。
4. **本地大模型（可选）**：不装也能完整演示（规则引擎离线降级）。如需 LLM 解读与向量
   检索，启动 LM-Studio 并加载 `qwen3-8b`（对话）与 `nomic-embed-text`（嵌入），开启
   本地 API 服务（默认 1234 端口），然后在首页勾选「启用本地模型增强」。
5. 前端所有资源（Tailwind / Chart.js）均已本地化在 `static/vendor/`，**无需外网**。

## 二、十分钟演示脚本

| 步骤  | 页签            | 动作                              | 看点                                                                                                                       |
| --- | ------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | ① 首页总览        | 开 2–3 床会话（中档精度）                 | 五波演进对比图；数据全部本地模拟                                                                                                         |
| 2   | ② ICU 数字孪生仪表盘 | 开「▶ 实时流入」、拧「收缩力 0.6×」旋钮、勾「表型调试」 | 床旁监护横幅：ECG 心电波 + 脉搏波（SpO₂）双示波器滚动，MAP/SpO₂/乳酸/NE/尿量大数字体征联动变色闪烁；实时流入每 1.2 s 推进一步（复现 Flink 采流体感，无需部署流引擎）；四联趋势图 40 点滑动窗口连续滚动 |
| 3   | ③ 多目标决策中心     | 调不确定性阈值 → 执行决策                  | 蒙特卡洛采样点逐批涌现（每臂 42 点），帕累托前沿动态成型；六项安全闸门卡片逐条点亮；不确定性数字滚动                                                                     |
| 4   | ④ 反事实推演       | 添加「如果选 B」分支 → 运行                | 现实 vs 反事实双世界曲线从左到右生长；分支生存概率数字滚动至目标值；ATE 逐指标标注（生存 X.Xpp）与置信区间                                                             |
| 5   | ⑤ Harness 进化  | 运行一轮进化；再点「红队攻击」                 | Propose→Guard→Probe→Compose 流水线逐节点点亮；红队注入非法提案被守卫拦截，节点变红 + 抖动动画                                                           |
| 6   | ⑥ 结局反馈        | 提交反馈（HemoSim / 人工）              | 后验系数条从 0 生长到目标宽度，臂 A/B/AB 切换；不确定性随学习下降                                                                                   |
| 7   | ⑦ 审计与消息板      | 看审计表；会诊轮询；掉线容错演示                | TOON vs JSON 压缩对比；多学科消息流逐条滑入                                                                                             |
| 8   | ⑧ 可信度与解释      | 看 ECE / DCA / Shapley           | 校准曲线 / 决策曲线 / Shapley 条形图带动画绘制                                                                                           |

> 界面为 ICU 监护风格深色主题，html 基准字号 15px，全部 8 个页签在 1920×1080 一屏内完整展示（仅实时流入开启时仪表盘约 12px 溢出）。

## 三、架构速览

### M1–M7 需求映射（对照前期策划文档）

| 模块  | 需求                      | 实现                                                                      | 出处                                    |
| --- | ----------------------- | ----------------------------------------------------------------------- | ------------------------------------- |
| M1  | 层递进式 Harness 组件化（8 类组件） | `config/harness_registry.json` 注册 8 类组件（6 可进化 + 均势池/不变量锁定），页签⑤可视化       | Harness²、Agent Harness 综述             |
| M2  | Fresh 独立复核子代理           | `core/fresh_reviewer.py`：高风险/分歧触发独立第二意见（数值重采样 + 可选 LLM 文本复核，与决策链无共享上下文） | 密歇根大学 EECS498 L14 Subagents（fresh 模式） |
| M3  | 轨迹级生成-验证（保留合法前缀）        | `core/trajectory_validator.py`：`legal_prefix()` 在首个违规步截断、保留合法前缀用于回放/学习  | SpecAgent                             |
| M4  | Order-flip 决策稳定性闸门      | `core/order_flip_gate.py` + 决策流内选项顺序置换 3 次重跑（M4 order_flip 字段），不稳即标记    | AnyJev、Unsloth Decision Model         |
| M5  | RSI 经验回灌（三层）            | `core/experience_replay.py`：病例级/轨迹级/策略级三层，反馈后自动沉淀高奖励策略文本                | Terminal-RSI、CS329A 自我改进 Agent        |
| M6  | 可扩展协调消息板                | `core/message_board.py` + `agent_orchestrator.py`：发布/搜索/热度排序，读取有界，掉线容错  | SCALEMAS（Qi et al., arXiv:2610.16120） |
| M7  | TOON 紧凑轨迹上报             | `core/toon_codec.py`：扁平键值行 + 同构记录列式编码，演示负载 300B→149B（约 50%）             | JSON vs TOON                          |

双层递归闭环（外循环 Propose→Probe→Compose / 内循环床旁照护即学习）在页签⑤展示。

### 目录与模块

```
main.py            FastAPI 后端（约 15 个 API，JSONL 追加审计 + TOON 体积对比）
config/            决策点/患者模板/指南知识/药物图谱/不变量/Harness 注册表
core/
  hemo_sim.py      M8  HemoSim 教学级数字孪生（Frank-Starling + NE PK/PD + 乳酸动力学）
  mab.py           M9  情境化汤普森采样（贝叶斯 logistic + 在线 Laplace 后验）
  pareto.py        M10 三目标帕累托前沿（蒙特卡洛采样非支配排序）
  counterfactual.py M11 反事实分叉推演（10/90 分位带 + ATE）
  calibration.py   M12 ECE / DCA / 精确 Shapley（2^9=512 子集枚举）
  harness.py + harness_evolver.py  Harness 注册/进化 + 不变量守卫（红队演示）
  order_flip_gate.py   U1 决策-生成分离（冻结哈希 + 顺序翻转测试）
  fresh_reviewer.py    U2 反思复核（独立重采样一致性检查）
  trajectory_validator.py U3 轨迹不变量校验
  experience_replay.py U5 经验回放（RSI 三层回灌：案例/轨迹/策略）
  message_board.py + agent_orchestrator.py  M13 SCALEMAS 式消息板（Qi et al., arXiv:2610.16120）+ 多学科 Agent + 掉线容错
  toon_codec.py    M7  TOON 紧凑编码（审计带宽对比）
  knowledge_base.py    指南检索（embedding 可降级关键词）
  llm_client.py        LM-Studio 本地客户端（全部失败优雅降级）
data/              10 例演示 CSV + 200 例校准队列 + 各 JSONL 日志
static/            八页签 SPA（vendor 已本地化，无外网依赖）
```

## 四、安全红线（本 POC 如何遵守）

- 全部患者数据为人工构造模拟数据，不涉及真实患者。
- 大模型全部本地运行（LM-Studio），数据不出本机；不勾选则完全离线。
- AI 不出诊断结论、治疗医嘱、预后判断；输出均为「参考 + 须医生审阅」。
- MAB 决策由 numpy 独立计算；LLM 仅做结果解读、向量库仅输出参考证据，
  二者绝不干预决策数值（INV-3，代码级硬编码 + 红队测试演示）。
- 不变约束（INV-1~5）在任何 Harness 版本下不可变；守卫逻辑不可被进化器修改。
- 不引入 Docker / WSL / 虚拟机 / 外部数据库；不训练、不微调任何模型。

## 五、局限性

- HemoSim 是**教学级集总参数模型**（几十行公式），不是商用生理数字孪生；
  反事实结论只在模拟器内部成立，不能外推真实患者。
- 三类奖励目标权重（0.5/0.3/0.2）与剂量步长为演示默认值，真实部署须由 ICU
  专家依据指南与本院数据标定。
- 表型在演示中直接可观测；真实场景应由因果推断/潜变量模型在线识别。
- 校准曲线基于 200 例预生成模拟队列（方法演示），非真实回顾性数据。
- 多智能体（微生物/检验/影像等）为脚本化规则引擎模拟消息流，非真实系统对接。
