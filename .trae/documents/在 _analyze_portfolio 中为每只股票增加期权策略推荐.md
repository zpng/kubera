## 目标
- 在 `/analyze_portfolio` 流程的每只股票分析结果中新增“期权推荐策略”与“详细理由”，通过调用现有大模型接口生成。
- 输出与现有信息并列展示于 Telegram 消息中，保持中文叙述，结构清晰且不依赖精确期权报价。

## 集成位置
- 工作流入口：`TelegramBot.analyze_portfolio_command` 触发新工作流并发送结果 `src/bot/telegram_bot.py:317-356`。
- 工作流主体：`PortfolioAnalysisWorkflow.run` 的阶段 4 深度研究后，新增“期权策略”阶段并写回 `state` `portfolio_agent_orchestration/workflows/main_workflow.py:161-167`。
- 研究代理：沿用 `DeepResearcherAgent` 的聚合数据方式 `aggregate_portfolio_data` `portfolio_agent_orchestration/agents/deep_researcher.py:781-799`。
- 展示输出：扩展 `_send_new_portfolio_results`，为每只股票追加“期权策略 + 理由” `src/bot/telegram_bot.py:361-484`。

## 数据模型变更
- 在 `state["research_results"][symbol]` 下新增 `options_strategy` 字段（对象）：
  - `strategy`: 推荐策略名称（如“Covered Call”“Cash-Secured Put”“Protective Put”“Collar”“Bull Call Spread”“Iron Condor”等）
  - `summary`: 策略一句话总结（中文）
  - `parameters`: 建议参数的文字化描述（如“略 OTM”“到期 30–45 天”“等级偏保守/中性/进取”）
  - `rationale`: 详细理由（中文，条目化）
  - `risk_notes`: 风险与适用性说明（含持股数量要求、保证金要求等）
  - `suitability`: 适配情景（如“已有仓位且倾向持有”“希望加仓但控制下行风险”“对冲波动”等）
- 保持向后兼容：现有消费方仅读取 `decision/conviction/target_price/rationale`；新增字段不会破坏现有逻辑。

## 代理与提示词设计
- 新增 `OptionsStrategyAgent`（继承 `BaseAgent`，模型默认 `deepseek-reasoner`，与其他代理一致使用 OpenRouter）：
  - 输入：每只股票的聚合数据（价格趋势与波动、基本面与分析师观点、社交与新闻情绪、风险评分、`DeepResearcher` 的 `decision` 与 `conviction`）。
  - 产出：上述 `options_strategy` 对象。
  - 约束：不输出精确价格/希腊值/保证金数额，使用相对描述（“略 OTM”“delta 约 0.3”以文字表述，必要时不写具体数字）。
  - 语言：中文。
- 提示词要点：
  - 明确可选策略集合与选择规则（基于`decision`、`conviction`、`risk_level`、`volatility`）。
  - 要求结构化 JSON 输出（含 `strategy/summary/parameters/rationale/risk_notes/suitability`）。
  - 若不适合期权（流动性差/风险极端/不满足持股要求），给出“暂不建议期权”的说明与原因。

## 解析与容错
- 主路径：尝试 `json.loads` 解析模型输出；解析成功则写入 `state["research_results"][symbol]["options_strategy"]`。
- 失败回退：保留纯文本到 `options_strategy = {"strategy": "N/A", "summary": "模型输出未结构化", "rationale": text}`。
- 超时与短响应：沿用 `BaseAgent.invoke` 的重试与退避机制；必要时降级至更快模型（如 `deepseek-v3.1`）。

## 工作流改动
- `PortfolioAnalysisWorkflow.__init__`：新增 `self.options_strategy = OptionsStrategyAgent()` `portfolio_agent_orchestration/workflows/main_workflow.py:40-66`。
- `PortfolioAnalysisWorkflow.run`：在深度研究完成后调用 `self.options_strategy.process(state)` 并合并结果 `portfolio_agent_orchestration/workflows/main_workflow.py:161-167`。

## Telegram 展示
- 在 `_send_new_portfolio_results` 每只股票的块中，追加：
  - `**期权策略：** {options_strategy.strategy}`
  - `**理由：**` 取 `rationale` 前若干条的摘要（控制在 3–5 条，避免超长）。
  - 若 `parameters` 存在，追加一行“**参数建议：** {parameters}”。
- 长文分片逻辑复用现有 4000/4096 限制处理 `src/bot/telegram_bot.py:416-456`。

## 配置与依赖
- 在 `portfolio_agent_orchestration/config.py` 的 `AGENT_MODELS` 中新增键：`"options_strategy": "deepseek-reasoner"`（与现有模型风格一致）`portfolio_agent_orchestration/config.py:31-40`。
- 沿用 OpenRouter 环境变量与超时配置，无新增敏感信息。

## 测试与验证
- 单元测试：
  - 构造最小 `state`，覆盖不同组合情景（多头高信心、持有中性、减仓/卖出、高风险高波动），断言 `options_strategy.strategy` 合理（如持有盈利→Covered Call；高风险下行→Protective Put/Collar；看多但控制风险→Bull Call Spread；中性高波动→Iron Condor）。
  - 解析容错（非 JSON 响应）路径覆盖。
- 端到端：运行工作流一次，检查 `research_results` 是否包含 `options_strategy`，并在 Telegram 预览中是否正确展示。

## 边界与安全
- 无期权链与精确报价：策略仅提供类型与原则，不给出具体行权价与保证金数值。
- 持股约束：如推荐 Covered Call/Collar，明确需要至少 100 股；否则给出替代方案（如 Cash-Secured Put）。
- 免责声明：沿用现有“非投资建议”落款 `src/bot/telegram_bot.py:463-482`。

## 实施步骤
1. 新增 `OptionsStrategyAgent` 文件与实现（聚合输入、提示词、解析、写入 state）。
2. 在工作流中注册并调用该代理，合并结果到 `research_results`。
3. 更新 Telegram 输出函数，追加期权策略与理由展示。
4. 更新配置 `AGENT_MODELS`，为新代理指定模型。
5. 添加测试样例并本地运行验证（字符长度与分片、空响应降级）。

## 交付物
- 代码改动：新增代理文件、工作流改动、配置更新、输出展示更新。
- 可运行演示：`/analyze_portfolio` 调用后，每只股票消息块出现“期权策略 + 详细理由”。