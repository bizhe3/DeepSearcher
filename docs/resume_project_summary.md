# 项目：自主深度研究 Agent（LLM + Tool Use + RAG）

基于 Plan-and-Execute 架构实现自主网络研究 Agent，支持复杂研究任务的自动分解、多步执行、信息检索与答案综合，覆盖仿真与真实 Web 双环境。

---

## 一、任务规划 / Agent 核心能力

- 设计基于 Plan-and-Execute 的 Agent 执行框架，通过 SubGoalDecomposer 将复杂研究任务自动分解为 2-6 个可验证子目标，逐步执行并整合结果
- 构建支持多轮决策的任务状态机（pending/active/completed/failed），含停滞检测与动态 replan 机制（每 5 步根据已完成目标与新信息自动重新规划剩余子目标），提升长链路任务的执行稳定性
- 实现基于 LLM 推理 + 结构化决策表的 6 种 action 路由（search/extract/click/scroll/cross_check/terminate），通过 prompt 内嵌决策规则约束模型选择，减少无效调用

**子目标分解机制详解**

SubGoalDecomposer 通过 LLM 将用户的自然语言研究任务拆解为结构化的子目标序列，是整个 Plan-and-Execute 架构的规划核心。

1. **分解阶段（decompose）**：将用户 query 发送给 LLM，通过系统 prompt 中的规划规则约束输出：
   - 要求生成 2-6 个具体、可通过网络搜索独立验证的子目标
   - 强制要求至少包含一个交叉验证子目标（cross-check），对关键定量数据进行二次核实
   - 要求数据收集类子目标排在验证类子目标之前，最终子目标负责综合结论
   - 输出格式为严格 JSON 数组（`[{"id": "sg_1", "description": "..."}]`），解析失败时自动追加纠错 prompt 重试一次

2. **重规划阶段（replan）**：在任务执行过程中每 5 步触发一次动态重规划：
   - 将已完成子目标及其状态、新发现的信息作为上下文发送给 LLM
   - LLM 输出更新后的剩余子目标列表
   - 通过 ID 与描述双重去重，过滤掉已完成的目标，仅保留真正新增或修改的子目标
   - 与已有子目标列表合并时，保留所有 completed/failed 状态的目标，替换 pending 目标

3. **子目标生命周期管理（get_active_goal + 状态机）**：
   - `get_active_goal()` 按顺序激活第一个 pending 状态的子目标，将其标记为 active
   - Agent 主循环中，每步执行后通过关键词覆盖率（>= 60%）检测子目标是否完成
   - 完成后由 LLM 压缩该子目标的观测记录为 1-2 句摘要，供后续子目标决策时作为长期记忆
   - 若某子目标连续尝试超过 stall_threshold 次仍未完成，自动标记为 failed 跳过

```
分解流程示意:

用户 Query: "比较 2024 年中美欧量子计算政策差异及对产业链的影响"
                          |
                    SubGoalDecomposer.decompose()
                          |
                          v
    ┌─────────────────────────────────────────────────────────┐
    │ sg_1: 检索中国 2024 年量子计算政策文件与资金投入情况        │ ← 数据收集
    │ sg_2: 检索美国 2024 年量子计算政策（CHIPS Act 等）         │ ← 数据收集
    │ sg_3: 检索欧盟量子旗舰计划 2024 年进展                    │ ← 数据收集
    │ sg_4: 交叉验证三方关键投资数据（金额、专利数）              │ ← 交叉验证
    │ sg_5: 综合对比政策差异并分析对产业链上下游的影响            │ ← 综合结论
    └─────────────────────────────────────────────────────────┘
                          |
                    Agent 逐个执行
                          |
              ┌───── 每 5 步 replan ─────┐
              │ 输入: 已完成目标 + 新信息   │
              │ 输出: 调整后的剩余子目标    │
              └──────────────────────────┘
```

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| Agent 主循环 | `agent/agent.py` | `DeepResearchAgent.run()` — 状态机驱动的多步执行循环 |
| 子目标分解 | `agent/planner.py` | `SubGoalDecomposer.decompose()` — LLM 驱动的任务分解 + 解析失败重试 |
| 动态重规划 | `agent/planner.py` | `SubGoalDecomposer.replan()` — 基于已完成目标与新证据的重规划 + ID/描述双重去重 |
| 子目标激活 | `agent/planner.py` | `get_active_goal()` — 按序激活首个 pending 子目标 |
| 完成检测 | `reward/subgoal_reward.py` | `detect_completion()` — 关键词覆盖率 >= 60% 判定完成 |
| 摘要压缩 | `agent/agent.py` | `_compress_sub_goal()` — LLM 将观测记录压缩为 1-2 句摘要 |
| 决策路由 | `agent/agent.py` | `_decide_action()` — 结构化 action schema + 决策表引导 |

---

## 二、工具调用（Tool Use）

- 封装统一工具调用接口（BaseEnv.execute_action），将 6 种 action 映射到搜索引擎 API（Bing/SerpAPI）、Playwright 无头浏览器页面抓取（集成 Readability.js 正文提取）、分页浏览等能力
- 构建可插拔环境层，SimEnv 基于本地语料库 + FAISS 向量检索支持离线实验，RealWebEnv 接入真实搜索 API + Chromium 支持生产级网络研究，两者共享同一 BaseEnv 接口，零改动切换
- 设计 LLM 输出解析容错机制（`_normalize_action_payload` 自动修复缺失字段）、解析失败精简 prompt 重试、extract/click 失败自动回退缓存 search snippet 的三级 fallback 策略，保障任务不因单点失败而中断

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| 统一调用接口 | `envs/base_env.py` | `BaseEnv.execute_action()` — 6 种 action 统一分发 |
| 仿真环境 | `envs/sim_env.py` | `SimEnv` — FAISS IndexFlatIP 向量检索 + 本地语料库 |
| 真实环境 | `envs/real_web_env.py` | `RealWebEnv` — Playwright + Bing/SerpAPI |
| 容错解析 | `agent/agent.py` | `_normalize_action_payload()` — 自动修复畸形 action JSON |
| Fallback | `agent/agent.py` | `_find_cached_snippet()` — extract/click 失败回退缓存 |

---

## 三、RAG + Agent 协同

- 基于 FAISS IndexFlatIP + SentenceTransformer 实现页面语义分块检索（PageChunkRetriever），以当前子目标为 query 对长页面 body 做语义过滤，将送入 LLM 的上下文压缩 82%（6,148 -> 1,088 chars），相关内容密度提升 6.4 倍
- 实现 trajectory 观测历史的语义检索（ObservationRetriever），替代原始末尾截断（`observations[-5:]`），解决长 trajectory 下早期关键信息被丢弃的问题。在 10 步对抗测试中，RAG 检索命中率 100%，末尾截断命中率 0%
- 检索-过滤在 Agent 层执行（非环境层），以子目标描述作为检索 query，实现"检索-过滤-决策"三阶段解耦，环境层保持单一职责

**PageChunkRetriever 详解（页面语义过滤）**

问题背景：Agent 通过 Playwright 抓取的真实网页通常包含大量无关内容（导航栏、广告、侧边栏文本等），原始 body 动辄数千甚至上万字符，但与当前子目标相关的段落可能只占 5-10%。直接将全文送入 LLM 既浪费 token，又引入噪声干扰决策。

工作流程：

1. **短文本跳过**：若 body 长度 <= `min_body_len`（默认 1000 chars），直接返回原文，避免对短页面做不必要的检索
2. **固定窗口分块**：将 body 按 `chunk_size`（默认 300 chars）切分为若干文本块，不做重叠（简单高效，避免分块本身成为瓶颈）
3. **语义编码**：使用 SentenceTransformer（all-MiniLM-L6-v2，384 维）将所有 chunk 和当前子目标描述分别编码为向量
4. **L2 归一化 + 内积检索**：对所有向量做 L2 归一化后，通过 FAISS IndexFlatIP 计算余弦相似度，取 top-k（默认 5）个最相关的 chunk
5. **按原序拼接**：将命中的 chunk 按原始位置排序后拼接返回，保持文本的阅读连贯性

设计决策——为什么放在 Agent 层而非环境层：
- 环境层（BaseEnv）的 `fetch_page` 职责是"忠实抓取"，不应依赖任务语义
- Agent 层拥有 `active_sub_goal.description` 作为 query，这是过滤的语义锚点
- 这样环境层可被多种 Agent 复用，过滤策略可按 Agent 需求独立替换

```
PageChunkRetriever 流程:

  fetch_page 返回的 body（~6000 chars，含大量噪声）
      │
      ▼
  body 长度 > 1000?  ─── 否 ──→ 直接返回原文
      │ 是
      ▼
  按 300 chars 固定窗口切分 → [chunk_0, chunk_1, ..., chunk_31]
      │
      ▼
  SentenceTransformer.encode(chunks)  → chunk_vecs (32 × 384)
  SentenceTransformer.encode([query]) → query_vec  (1 × 384)
      │
      ▼
  faiss.normalize_L2 + IndexFlatIP.search → top-5 indices
      │
      ▼
  按原始位置排序 → 拼接返回（~1000 chars，只含相关段落）
```

**ObservationRetriever 详解（观测历史 RAG）**

问题背景：Agent 在多步执行中积累的 trajectory 观测记录会不断增长。原始实现使用 `observations[-5:]` 截取最近 5 条观测作为决策上下文，存在严重的**近因偏差（recency bias）**——当 trajectory 超过 5 步时，早期步骤中发现的关键信息会被无条件丢弃，即使它们与当前子目标高度相关。

这是一个真正的 RAG 场景：
- **Corpus（语料库）**：不断增长的 trajectory.observations 列表
- **Query**：当前活跃子目标的描述（active_goal.description）
- **Retrieval**：通过语义相似度从全部历史观测中检索最相关的 top-k 条
- **Augmented Generation**：将检索结果拼接进 prompt，增强 LLM 的决策上下文

工作流程：

1. **观测文本化**：将每条 AgentObservation 转化为可编码的文本，拼接 action_type、thought、result 摘要（PageContent 取 body 前 200 chars，SearchResult 取 snippet 前 80 chars）
2. **语义编码 + 检索**：与 PageChunkRetriever 相同的 FAISS 流程，但检索粒度是**整条观测**而非文本块
3. **按原序返回**：命中的观测按 step 序号排序后返回，保持时间线顺序

与末尾截断的本质区别：

```
10 步 trajectory，query = "Berlin Airlift political consequences"
目标观测在 step 1（唯一直接相关的观测）

末尾截断 [-5:]  → 选中 steps [6,7,8,9,10] → step 1 丢失 → MISS
ObservationRetriever → 选中 steps [1,4,6,9,10] → step 1 命中 → HIT

根本原因：末尾截断假设"越新越重要"，但研究任务中
早期搜索发现的核心证据可能比后续的验证/补充步骤更关键。
语义检索按相关性而非时间排序，消除了这个假设。
```

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| 页面语义过滤 | `utils/chunk_retriever.py` | `PageChunkRetriever.filter()` — 分块 + FAISS 语义排序 |
| 观测历史检索 | `utils/chunk_retriever.py` | `ObservationRetriever.search()` — trajectory 级 RAG |
| 观测文本化 | `utils/chunk_retriever.py` | `ObservationRetriever._obs_to_text()` — 结构化观测转文本 |
| Agent 集成点（改动一） | `agent/agent.py` | `run()` 循环中 PageContent 过滤（第 151-166 行） |
| Agent 集成点（改动二） | `agent/agent.py` | `_build_hierarchical_context()` 中观测检索（第 346-349 行） |

---

## 四、记忆与上下文管理

- 实现分层上下文管理：已完成子目标由 LLM 压缩为 1-2 句摘要（压缩历史层，降低 token 消耗），当前子目标的相关观测通过语义检索从全 trajectory 中动态召回（RAG 工作层，按相关性而非时间选取），两层拼接构建决策上下文，兼顾信息完整性与 token 效率
- SynthesisWriter 在终止阶段将全 trajectory 成功观测提取为带编号的 evidence context，由 LLM 生成带内联引用的 markdown 答案，自动解析 References 段落提取 URL 列表

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| 分层上下文构建 | `agent/agent.py` | `_build_hierarchical_context()` — 摘要层 + 观测层拼接 |
| 子目标摘要压缩 | `agent/agent.py` | `_compress_sub_goal()` — LLM 驱动的观测记录压缩 |
| 答案综合 | `agent/synthesizer.py` | `SynthesisWriter.synthesize()` — evidence 拼接 + LLM 生成 |
| 引用提取 | `agent/synthesizer.py` | `_extract_citations()` — markdown References 段解析 |

**上下文结构示意**

```
送入 LLM 的 prompt 结构:
  ┌─────────────────────────────────────┐
  │ 已完成目标摘要（长期记忆）             │
  │   [子目标A] -> 1-2句摘要              │
  │   [子目标B] -> 1-2句摘要              │
  ├─────────────────────────────────────┤
  │ 当前目标相关观测（短期记忆，RAG 检索）  │
  │   #1 action=search ... result=...    │
  │   #2 action=extract ... result=...   │
  ├─────────────────────────────────────┤
  │ Action Schema + 决策规则表            │
  └─────────────────────────────────────┘
```

---

## 五、稳定性与防护

- 通过 action JSON schema + `<think>` 标签分离推理与输出，约束模型结构化输出；引入 LLM Judge 三维度评分（relevance/completeness/citation_quality）对最终答案进行质量校验
- 设计 Playwright 页面加载超时（15s）、搜索 API 指数退避重试（3 次、间隔递增）、asyncio.Lock 请求限速等防护机制
- 子目标停滞超过阈值自动标记失败并推进下一目标，避免单个子目标死循环消耗全部步数

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| 思维链分离 | `agent/agent.py` | `_extract_thought_and_action()` — `<think>` 解析 + JSON 提取 |
| LLM Judge | `reward/llm_judge.py` | `LLMJudge.judge()` — 三维度评分 |
| 限速防护 | `envs/real_web_env.py` | `_apply_fetch_rate_limit()` — asyncio.Lock 节流 |
| 搜索重试 | `envs/real_web_env.py` | `search()` — 3 次重试 + 指数退避 |
| 停滞检测 | `agent/agent.py` | `stall_threshold` — 子目标尝试次数超限自动标记失败 |
| 非 HTML 跳过 | `envs/real_web_env.py` | `_SKIP_EXTENSIONS` — 跳过 PDF/ZIP 等二进制资源 |

---

## 六、评估体系

- 构建多维度评估指标：子目标完成率（sub_goal_reward）、答案 ROUGE-L F1（answer_reward）、引用覆盖率（citation_reward）、步数效率惩罚（step_penalty），加权合成总分
- 实现 Golden Set 确定性评测流程（eval_golden.py），脚本化 LLM 行为保证可复现；支持接入 LLMJudge 做开放式评测
- 实现 RAG 组件专项 benchmark（eval_rag.py），量化页面过滤压缩率与精准率提升，以及观测检索 vs 末尾截断基线的命中率对比，无需 LLM 即可离线运行

**评估架构总览**

本项目的评估分为三层：组件级（RAG 检索质量）、轨迹级（RewardEngine 多维度打分）、端到端级（Golden Set + LLM Judge）。三层互补，从不同粒度衡量 Agent 性能。

```
评估三层架构:

┌────────────────────────────────────────────────────┐
│ 第三层：端到端评测（eval_golden.py）                  │
│   Golden Set JSONL → ScriptedLLM 确定性 rollout      │
│   → RewardEngine 自动打分 → 按 task 输出分数表        │
│   可选：接入 LLMJudge 做开放式评测                    │
├────────────────────────────────────────────────────┤
│ 第二层：轨迹级评分（RewardEngine）                    │
│   每条 trajectory 计算四维度加权分:                    │
│   sub_goal + answer(ROUGE-L) + citation - penalty   │
│   可作为 RL 训练的 reward signal                     │
├────────────────────────────────────────────────────┤
│ 第一层：组件级 benchmark（eval_rag.py）               │
│   Suite A: PageChunkRetriever 压缩率 + 精准率        │
│   Suite B: ObservationRetriever 命中率 vs 基线        │
│   无需 LLM，纯本地，确定性可复现                      │
└────────────────────────────────────────────────────┘
```

**第一层：RAG 组件级 benchmark（eval_rag.py）**

无需 LLM API，纯本地运行，用于量化 RAG 两个组件的检索质量。

Suite A — PageChunkRetriever 评估：
- 构造一个**可控的合成长页面**：1 段相关内容（Berlin Airlift）+ 15 段 Lorem ipsum 噪声，共 6,148 chars
- 度量 1 — **压缩率**：`len(filtered) / len(original)`，衡量送入 LLM 的 token 节省量
- 度量 2 — **关键词精准率**：将过滤前后的 body 各自分块，统计包含 query 关键词（Berlin, NATO, blockade 等）的 chunk 占比。过滤前 6.25%（32 个 chunk 里 2 个相关），过滤后 40%（5 个 chunk 里 2 个相关）

Suite B — ObservationRetriever 评估：
- 构造一个**对抗性 10 步 trajectory**：step 1 是唯一高度相关的观测（Berlin Airlift），steps 2-8 是完全无关的噪声（法国料理、量子计算、世界杯...），steps 9-10 是半相关（冷战背景）
- 这是一个刻意设计的**最坏情况**——相关信息在最早的位置，末尾截断必然丢失
- 度量 — **命中率**：RAG 和 `[-5:]` 各自选出的 top-5 观测是否包含 step 1。RAG 命中，末尾截断未命中
- 同时输出 RAG 选中的具体 steps 和 snippet 摘要，便于定性分析检索是否合理

```
Suite A 结果:
  Body 压缩率          : 17.70%（压缩了 82%）
  关键词精准率（过滤前）: 6.25%   (2/32 chunks)
  关键词精准率（过滤后）: 40.00%  (2/5 chunks)
  Precision lift       : +33.75pp

Suite B 结果:
  RAG 选中 steps       : [1, 4, 6, 9, 10]  → step 1 命中 [HIT]
  末尾截断 [-5:] steps  : [6, 7, 8, 9, 10]  → step 1 丢失 [MISS]
```

**第二层：轨迹级多维度评分（RewardEngine）**

每条 trajectory 执行完毕后，RewardEngine 计算四个维度的加权分数：

| 维度 | 计算方式 | 权重 | 衡量什么 |
|------|----------|------|----------|
| sub_goal | 已完成子目标数量 × weight | 0.2 | 任务分解执行的覆盖程度 |
| answer | ROUGE-L F1（agent 答案 vs 参考答案）× weight | 1.0 | 最终答案质量 |
| citation | min(实际引用数 / 期望引用数, 1.0) × weight | 0.3 | 引用是否充分 |
| efficiency | step 数 × penalty（扣分项） | -0.01 | 步数越少越好 |

```
评分公式:
total = 0.2 × completed_sub_goals
      + 1.0 × ROUGE_L_F1(answer, reference)
      + 0.3 × min(citation_count / 3, 1.0)
      - 0.01 × step_count

total = max(0.0, total)  // 下限截断为 0
```

子目标完成检测（detect_completion）的判定逻辑：
- 从子目标描述中提取关键词（去停用词）
- 在 trajectory 全部观测的文本中搜索这些关键词
- 若关键词命中率 >= 60%，判定该子目标已完成

可选扩展——LLM Judge（compute_with_judge）：
- 在 ROUGE-L 机械评分之上，可接入 LLMJudge 进行开放式评测
- LLMJudge 从 relevance、completeness、citation_quality 三个维度各给 1-5 分
- 加权合成：`total = (relevance×0.4 + completeness×0.4 + citation_quality×0.2) / 5`
- Judge 结果可替换 RewardEngine 的 answer 分，提供更接近人类判断的评分

**第三层：端到端评测（eval_golden.py）**

Golden Set 评测是最完整的端到端测试，模拟 Agent 从接收任务到输出答案的全流程：

1. **数据准备**：从 `data/golden_set.jsonl` 加载测试记录，每条包含 task_id、query、expected_sub_goals、reference_answer
2. **确定性环境**：使用 GoldenSimEnv（SimEnv 的子类），内置 FakeEncoder 和 FakeIndex 替代真实的 SentenceTransformer 和 FAISS，确保每次运行结果完全一致
3. **脚本化 LLM**：ScriptedLLMClient 根据 prompt 内容自动返回预设响应——规划请求返回 expected_sub_goals，action 请求依次返回 search → terminate（带 reference_answer），消除 LLM 随机性
4. **自动评分**：每条 task 的 trajectory 经 RewardEngine 计算四维度分数
5. **汇总输出**：按 task 打印分数表 + 全局平均值

```
eval_golden.py 输出格式:
task_id                            | sub_goal | answer  | citation | total
-----------------------------------+----------+---------+----------+--------
task_001                           |    0.400 |   0.850 |    0.300 |  1.530
task_002                           |    0.200 |   0.720 |    0.300 |  1.200
-----------------------------------+----------+---------+----------+--------
AVERAGE                            |    0.300 |   0.785 |    0.300 |  1.365
```

**如何用评估体系量化 RAG 的改进**

RAG 改动的效果可以在三层评估中分别体现：

| 层次 | 改动前（baseline） | 改动后 | 量化指标 |
|------|---------------------|--------|----------|
| 第一层 | 无过滤 / 末尾截断 | PageChunkRetriever / ObservationRetriever | 压缩率 82%、精准率 +33.75pp、命中率 100% vs 0% |
| 第二层 | body[:120] 硬截断丢信息 | body[:600] 过滤后保留更多相关内容 | answer ROUGE-L 分数提升（因 LLM 获得更优质上下文） |
| 第三层 | 相同 Golden Set | 接入真实 LLM 运行新旧两版 | total reward 分数对比 |

第一层已实现并验证（eval_rag.py），无需 LLM 即可离线运行。
第二、三层需要接入真实 LLM API 运行新旧两版 Agent，对比 reward delta。

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| Reward 引擎 | `reward/reward_engine.py` | `RewardEngine.compute()` — 四维度加权评分 |
| ROUGE-L 评分 | `reward/reward_engine.py` | `_answer_score()` — rouge_scorer 计算 F1 |
| LLM Judge | `reward/llm_judge.py` | `LLMJudge.judge()` — 三维度开放式评分 |
| Judge 集成 | `reward/reward_engine.py` | `compute_with_judge()` — Judge 分数替换 answer 分 |
| 子目标完成检测 | `reward/subgoal_reward.py` | `detect_completion()` — 关键词覆盖率 >= 60% 判定 |
| Golden Set 评测 | `tests/eval_golden.py` | ScriptedLLMClient + GoldenSimEnv 确定性端到端评测 |
| RAG 专项评测 | `tests/eval_rag.py` | Suite A 压缩率/精准率 + Suite B 命中率 vs 基线对比 |

---

## 七、性能与工程化

- SentenceTransformer 懒加载，短 trajectory 零开销；token 用量逐次追踪（input/output tokens），支持成本监控
- YAML 配置驱动，支持 SimEnv/RealWebEnv 切换、多 LLM 供应商（Anthropic/DeepSeek/OpenAI-compatible）热切换、Planner 与 Decision Model 分离部署（如 Haiku 做规划 + DeepSeek-R1 做决策）
- 原子化 checkpoint 持久化（tmp 写入 + rename），支持长任务断点续跑；progress_callback 实时输出每步执行状态

**对应代码**

| 模块 | 文件 | 核心实现 |
|------|------|----------|
| 懒加载 | `utils/chunk_retriever.py` | `_get_model()` — 首次调用时初始化 |
| Token 追踪 | `utils/llm_client.py` | `AnthropicClient.total_input_tokens / total_output_tokens` |
| 配置加载 | `utils/config.py` | `build_agent_from_config()` — YAML 驱动依赖组装 |
| 多供应商 | `utils/llm_client.py` | `AnthropicClient` + `OpenAICompatibleClient` |
| Checkpoint | `utils/checkpoint.py` | `save_checkpoint()` — 原子写入 + `load_checkpoint()` 恢复 |
| 环境变量 | `main.py` | `_resolve_env_placeholders()` — `${VAR}` 语法解析 |

**配置矩阵**

| 配置文件 | 环境 | 决策模型 | 规划模型 | 用途 |
|----------|------|----------|----------|------|
| `sim_train.yaml` | SimEnv | claude-sonnet-4-6 | 同左 | 仿真训练 |
| `real_eval.yaml` | RealWebEnv (Bing) | claude-sonnet-4-6 | 同左 | 真实环境评测 |
| `deepseek_eval.yaml` | RealWebEnv (SerpAPI) | deepseek-chat (V3) | 同左 | DeepSeek 评测 |
| `deepseek_r1_v3.yaml` | RealWebEnv (SerpAPI) | deepseek-reasoner (R1) | deepseek-chat (V3) | R1 推理评测 |
| `deepseek_haiku.yaml` | RealWebEnv (SerpAPI) | deepseek-chat | claude-haiku-4-5 | 混合架构评测 |

---

## 量化成果

| 指标 | 数值 | 来源 |
|------|------|------|
| 页面语义过滤压缩率 | 82%（6,148 -> 1,088 chars） | eval_rag.py Suite A |
| 关键词精准率提升 | +33.75pp（6.25% -> 40.00%） | eval_rag.py Suite A |
| 观测 RAG 检索命中率 | 100%（vs 末尾截断 0%） | eval_rag.py Suite B |
| 支持配置数 | 5 套（覆盖仿真 + 多模型真实评测） | configs/ |
| Action 类型数 | 6 种（search/extract/click/scroll/cross_check/terminate） | base_env.py |

---

## 项目结构

```
deepresearch/
  agent/
    agent.py          # Agent 主循环、决策、上下文管理
    planner.py         # 子目标分解与动态重规划
    synthesizer.py     # 最终答案综合与引用提取
    types.py           # Pydantic 数据契约
  envs/
    base_env.py        # 环境抽象基类 + 统一 action 执行
    sim_env.py         # FAISS 向量检索仿真环境
    real_web_env.py    # Playwright + Search API 真实环境
  reward/
    reward_engine.py   # 多维度加权评分引擎
    subgoal_reward.py  # 子目标完成检测
    llm_judge.py       # LLM-as-Judge 评分
  utils/
    chunk_retriever.py # PageChunkRetriever + ObservationRetriever
    config.py          # YAML 配置加载与依赖组装
    llm_client.py      # Anthropic + OpenAI-compatible 客户端
    checkpoint.py      # 原子化断点持久化
  training/
    rollout.py         # Rollout 脚手架（待实现）
    ppo_trainer.py     # PPO 训练脚手架（待实现）
    sft_trainer.py     # SFT 训练脚手架（待实现）
  tests/
    eval_golden.py     # Golden Set 端到端评测
    eval_rag.py        # RAG 组件专项 benchmark
  configs/
    sim_train.yaml     # 仿真训练配置
    real_eval.yaml     # 真实环境评测配置
    deepseek_*.yaml    # DeepSeek 系列评测配置
  main.py              # CLI 入口
```
