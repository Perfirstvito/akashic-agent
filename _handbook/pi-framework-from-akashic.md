# 从 Akashic 理解 Pi Agent：记忆检索、上下文管理、工具调用

这份文档面向正在学习 agent 框架的人。阅读顺序不是按 Pi 的目录树，而是从 Akashic Agent 里已经比较清楚的三个概念出发：记忆检索、上下文管理、工具调用。核心问题是：如果 Akashic 是一个长期运行、主动触达、带长期记忆的个人 agent，那么 Pi 是怎样组织一个编码 agent 的？

## 0. 先建立对照心智模型

Akashic 的被动回复链路可以简化成：

```text
InboundMessage
  -> BeforeTurn: 取 session，准备上下文
  -> memory retrieval: 根据本轮消息和历史检索长期记忆
  -> PromptRender: 渲染 system/user/history/memory/skills
  -> Reasoner: LLM + tool loop
  -> AfterReasoning: 解析、持久化、构造出站消息
  -> AfterTurn: 事件广播、发送、记忆后处理
```

对应源码主要在：

- `agent/core/passive_turn.py`
- `agent/retrieval/default_pipeline.py`
- `plugins/default_memory/engine.py`
- `memory2/retriever.py`
- `agent/tools/registry.py`

Pi 的核心链路不是“服务收到一条消息”，而是“CLI/SDK 收到一个 prompt 后驱动一个可恢复的 coding session”：

```text
CLI / print / RPC / SDK
  -> AgentSession.prompt()
  -> SessionManager.buildSessionContext()
  -> buildSystemPrompt()
  -> Agent.run() / agent-loop
  -> LLM streaming
  -> executeToolCalls()
  -> append tool results
  -> maybe compact
  -> persist JSONL session
```

对应源码主要在 Pi 仓库：

- `packages/coding-agent/src/main.ts`
- `packages/coding-agent/src/core/agent-session.ts`
- `packages/coding-agent/src/core/session-manager.ts`
- `packages/coding-agent/src/core/system-prompt.ts`
- `packages/agent/src/agent.ts`
- `packages/agent/src/agent-loop.ts`
- `packages/coding-agent/src/core/tools/`
- `packages/coding-agent/src/core/extensions/`

最重要的差异是：Akashic 有显式的 memory engine 和 retrieval pipeline；Pi 没有内置一个类似 `MemoryEngine.retrieve()` 的长期语义记忆层。Pi 的“记忆”主要是 session 文件、compaction summary、branch summary、项目指令文件、skills/resources，以及扩展可以额外注入的 custom message。

## 1. 记忆检索：Akashic 是 RAG，Pi 是 session reconstruction

### 1.1 Akashic 的记忆检索

Akashic 在每轮被动回复前，会先做预检索。

`DefaultContextStore.prepare()` 读取 session history，然后构造 `RetrievalRequest`，交给 `DefaultMemoryRetrievalPipeline.retrieve()`。这个 pipeline 本身只做协议转换，把 query、session scope、history、metadata 转成 `MemoryEngineRetrieveRequest`，真正检索由 memory engine 决定。

默认记忆插件里，`plugins/default_memory/engine.py` 的 `retrieve()` 会调用 memory2 的 retriever。`memory2/retriever.py` 的思路是多路召回：

- 向量 lane：对 query 和辅助 query 做 embedding，再查 `MemoryStore2.vector_search_batch()` 或 `vector_search()`。
- 关键词 lane：从原始 query 抽关键词，查 `keyword_search_summary()`。
- 融合：用 RRF，把向量结果和关键词结果合成最终列表。
- 注入：`build_injection_block()` 把命中的长期记忆格式化成可塞进 prompt 的文本块。

所以 Akashic 的“记忆”是一个明确的外部知识层：

```text
用户消息 + 最近历史
  -> rewrite / aux queries
  -> vector search + keyword search
  -> RRF merge
  -> memory injection block
  -> PromptRender
```

这套设计适合个人助理，因为它要跨 session 记住用户事实、偏好、procedure、事件。

### 1.2 Pi 的“记忆”在哪里

Pi 的 session 管理在 `packages/coding-agent/src/core/session-manager.ts`。它把一次 agent 会话存成 JSONL，而且不是简单线性列表，而是 append-only tree。

重要 entry 类型包括：

- `message`：用户、assistant、tool result 等 agent 消息。
- `compaction`：上下文压缩摘要。
- `branch_summary`：会话分叉时的分支摘要。
- `thinking_level_change` / `model_change`：模型和思考级别变化。
- `custom`：扩展私有状态，不进入 LLM context。
- `custom_message`：扩展注入的上下文消息，会进入 LLM context。
- `label` / `session_info`：UI 和会话元信息。

Pi 每次需要给 LLM 上下文时，调用 `SessionManager.buildSessionContext()`。它做三件关键事：

1. 从当前 leaf 往 parent 走，恢复当前分支路径。
2. 扫描路径，恢复当前 thinking level 和 model。
3. 把路径上的 entry 转成 `AgentMessage[]`。如果路径上有 compaction，就先放 compaction summary，再保留 `firstKeptEntryId` 之后的消息，再加 compaction 之后的消息。

也就是说，Pi 的核心“检索”不是语义检索，而是结构化重建：

```text
current leaf
  -> walk parent chain
  -> collect path entries
  -> apply compaction / branch summary
  -> AgentMessage[]
```

这和 Akashic 的差异非常大。Akashic 问的是“长期记忆库里有没有和当前 query 相关的东西”；Pi 问的是“当前会话分支应该恢复成哪些消息给模型看”。

### 1.3 为什么 Pi 这样设计

Pi 是 coding agent。编码任务的主要上下文通常来自：

- 当前会话之前做了什么；
- 当前项目的文件；
- 工具读到的输出；
- agent 修改过什么；
- 分叉、恢复、压缩后的任务状态。

这些东西天然适合 session replay，而不是个人画像式长期记忆。Pi 不需要默认知道“用户喜欢什么”，它更需要知道“这个 branch 之前读过哪些文件、运行过哪些命令、修改过哪些代码、为什么 compact 成现在这样”。

如果类比 Akashic：

| 概念 | Akashic | Pi |
| --- | --- | --- |
| 长期记忆入口 | `MemoryEngine.retrieve()` | 无内置等价物 |
| 每轮上下文来源 | session history + memory block + skills | JSONL session path + project context + skills + tools |
| 压缩机制 | markdown memory / consolidation / optimizer | compaction summary |
| 分支记忆 | 基本按 session key | session tree + branch summary |
| 可扩展注入 | plugin phase modules | extension custom messages / resources |

## 2. 上下文管理：Akashic 是生命周期 pipeline，Pi 是 AgentSession 组装

### 2.1 Akashic 的上下文管理

Akashic 的上下文准备分散在生命周期 phase 中，但主线很清楚：

1. `BeforeTurn` 获取或创建 session，准备 `TurnState`。
2. `DefaultContextStore.prepare()` 读取历史、做 memory retrieval、识别 skill mention，形成 `ContextBundle`。
3. `BeforeReasoning` 同步工具上下文、触发事件、做 prompt warmup。
4. `PromptRender` 调 `ContextBuilder.render()`，把 system prompt、历史、记忆块、skills、媒体能力等渲染出来。
5. `Reasoner.run_turn()` 在这个上下文上进行 LLM/tool loop。

Akashic 把上下文管理做成 lifecycle pipeline，优点是插件能插入固定位置：BeforeTurn、BeforeReasoning、PromptRender、BeforeStep、AfterStep、AfterReasoning、AfterTurn。

### 2.2 Pi 的上下文入口：AgentSession.prompt()

Pi 的用户输入入口是 `AgentSession.prompt()`。它不是直接把字符串扔给模型，而是先经过一组 preflight：

1. 如果是 `/` 开头，优先尝试执行 extension command。
2. 触发 extension 的 `input` 事件，扩展可以处理或改写输入。
3. 展开 skill command 和 prompt template。
4. 如果 agent 正在 streaming，根据 `steer` 或 `followUp` 入队。
5. 如果当前不在运行，校验 model 和 auth。
6. 检查是否需要 compaction。
7. 构造 user message，进入 agent run。

这里可以把 Pi 的 `AgentSession.prompt()` 看作 Akashic 的 `BeforeTurn + BeforeReasoning` 的合体，只是它更偏 CLI 交互和 extension command，而不是 channel/message bus。

### 2.3 Pi 的 system prompt 和项目上下文

Pi 的 system prompt 在 `packages/coding-agent/src/core/system-prompt.ts` 构造。它包含：

- 默认身份：coding assistant inside pi。
- 当前启用工具及工具 prompt snippet。
- 工具相关 guideline。
- Pi 自身文档路径提示。
- 项目上下文文件。
- skills。
- 当前日期。
- 当前工作目录。

项目上下文由 `ResourceLoader` 负责。`loadProjectContextFiles()` 会读取：

- 全局 agent 目录里的 `AGENTS.md` 或 `CLAUDE.md`。
- 当前项目及父目录上的 `AGENTS.md` 或 `CLAUDE.md`，前提是项目被 trust。

然后 `buildSystemPrompt()` 把这些文件包成：

```xml
<project_context>
  <project_instructions path="...">
  ...
  </project_instructions>
</project_context>
```

这点和 Akashic 的 memory block 不同。Pi 的项目上下文是规则/说明，不是检索命中的事实。它通常稳定注入 system prompt；Akashic 的 memory block 是每轮按 query 动态检索出来的。

### 2.4 Pi 的上下文压缩

Pi 的 `AgentSession` 会在运行后检查 context 是否需要 compact。相关逻辑在：

- `packages/coding-agent/src/core/agent-session.ts`
- `packages/coding-agent/src/core/compaction/`

当上下文接近模型窗口，Pi 会生成 compaction summary，并把它作为 `compaction` entry 追加到 session JSONL。之后 `buildSessionContext()` 会优先把这个摘要放入上下文，再保留 compaction 指定的近期消息。

可以理解为：

```text
完整历史太长
  -> LLM 生成摘要
  -> append compaction entry
  -> 后续上下文 = 摘要 + 近期消息
```

Akashic 的长期记忆是“提炼成可检索事实”；Pi 的 compaction 是“把当前任务历史压缩成继续工作所需摘要”。前者面向跨会话个人记忆，后者面向当前任务连续性。

## 3. 工具调用：Akashic 有工具注册/搜索/Hook，Pi 有 ToolDefinition/AgentTool/Extension 包装

### 3.1 Akashic 的工具调用

Akashic 的工具由 `ToolRegistry` 管理。每个工具有：

- `name`
- `description`
- `parameters`
- `risk`
- `always_on`
- `search_hint`
- source metadata

Akashic 还有 `tool_search` 机制。不是所有工具都必须一开始暴露给 LLM，工具可以 deferred，需要时先搜索工具目录再启用。`ToolExecutor` 会执行 tool hook，因此插件可以在工具执行前后阻断、改写或记录。

被动链路里，`DefaultReasoner.run()` 会处理 LLM 的 tool calls：

```text
assistant tool_calls
  -> before_step phase
  -> ToolExecutor / tool hooks
  -> execute tool
  -> append tool result
  -> after_step phase
  -> 继续 LLM
```

这适合长期服务，因为 Akashic 的工具来源很多：内置工具、MCP、记忆工具、调度工具、message push、peer agent、插件工具。

### 3.2 Pi 的工具定义

Pi 的内置 coding tools 在：

```text
packages/coding-agent/src/core/tools/
  read.ts
  bash.ts
  edit.ts
  write.ts
  grep.ts
  find.ts
  ls.ts
```

`tools/index.ts` 提供 `createAllToolDefinitions()`，默认会创建：

- `read`
- `bash`
- `edit`
- `write`
- `grep`
- `find`
- `ls`

但默认 active tools 是：

```text
read, bash, edit, write
```

`AgentSession._buildRuntime()` 会创建 base tool definitions，再初始化 `ExtensionRunner`，然后 `_refreshToolRegistry()` 合并：

- builtin tool definitions；
- extension 注册的 tools；
- SDK custom tools；
- allowed/excluded tool filters。

之后它会把 definition 包装成真正给 agent loop 使用的 `AgentTool`，并根据 active tool names 设置当前暴露给模型的工具。

### 3.3 Pi 的 tool loop

Pi 的通用 tool loop 在 `packages/agent/src/agent-loop.ts`。

核心流程：

```text
streamAssistantResponse()
  -> assistant message
  -> filter content where type == "toolCall"
  -> executeToolCalls()
  -> create ToolResultMessage
  -> append to currentContext.messages
  -> next LLM turn
```

`executeToolCalls()` 会判断并行还是顺序执行：

- 如果全局 `toolExecution` 是 sequential，顺序执行。
- 如果某个工具声明 `executionMode === "sequential"`，顺序执行。
- 否则并行执行 tool calls。

每个 tool call 经过：

1. 找工具，不存在则返回 error tool result。
2. `prepareArguments()` 可预处理参数。
3. `validateToolArguments()` 校验 schema。
4. `beforeToolCall` hook 可阻断。
5. `tool.execute()` 真正执行。
6. 工具可发 `tool_execution_update` 进度事件。
7. `afterToolCall` hook 可改写结果、标记 error、设置 terminate。
8. 生成 tool result message，继续喂给 LLM。

这和 Akashic 的工具 hook 很像，但 Pi 把 hook 抽象放在 `AgentLoopConfig` 里，并由 `AgentSession`/extension 层绑定。Akashic 是服务框架里有明确 `ToolExecutor` 和 plugin tool hooks；Pi 是 SDK 化、事件化、extension 化。

### 3.4 工具调用对照

| 维度 | Akashic | Pi |
| --- | --- | --- |
| 工具注册中心 | `ToolRegistry` | `AgentSession` 内部 tool definition registry |
| 工具动态发现 | `tool_search` deferred tools | active tools + extension tools，默认无 Akashic 式工具搜索 |
| 工具来源 | builtin / MCP / plugin / peer agent | builtin coding tools / extension / SDK custom |
| hook 位置 | `ToolExecutor` + plugin hooks | `beforeToolCall` / `afterToolCall` in `AgentLoopConfig` |
| 并发策略 | Akashic reasoner 内部控制 | explicit sequential / parallel |
| 结果入上下文 | append tool result 到 session history | append `ToolResultMessage` 到 current context 和 session |

## 4. Pi 的整体运行框架

把三条线合起来，Pi 的整体框架可以这样理解：

```text
main.ts
  -> parse CLI args
  -> resolve mode: interactive / print / json / rpc
  -> load settings, auth, models, resources
  -> create SessionManager
  -> create AgentSessionRuntime
  -> create AgentSession

AgentSession
  -> load project context, skills, prompts, extensions
  -> build tool registry
  -> build system prompt
  -> restore session context from JSONL
  -> bind extension events

prompt()
  -> extension command/input
  -> skill/template expansion
  -> queue or start agent
  -> Agent.run()

Agent / agent-loop
  -> stream LLM response
  -> execute tool calls
  -> append tool results
  -> repeat until no tool calls / no queued messages
  -> emit events

AgentSession post-run
  -> retry if needed
  -> compact if needed
  -> persist session entries
  -> update UI/RPC/print mode
```

其中 `packages/agent` 是通用 agent core，基本不关心 CLI；`packages/coding-agent` 是 coding agent 的产品层，负责文件工具、会话、扩展、TUI、RPC、settings、auth、模型选择。

## 5. 该怎么读 Pi 源码

建议按下面顺序读，不要从文件树顶部一口气扫：

1. `packages/coding-agent/src/main.ts`

   先看 CLI 如何变成一个 `AgentSession`。重点看 mode 选择、session 选择、model 解析、project trust。

2. `packages/coding-agent/src/core/agent-session.ts`

   这是 Pi 的中枢。重点看 `prompt()`、`_buildRuntime()`、`_refreshToolRegistry()`、`_rebuildSystemPrompt()`、`_checkCompaction()`。

3. `packages/coding-agent/src/core/session-manager.ts`

   看 JSONL session 如何变成上下文。重点看 `buildSessionContext()`、`appendMessage()`、`appendCompaction()`、`branch()`。

4. `packages/agent/src/agent.ts`

   看通用 agent 状态机。重点看 queued steering/follow-up message、`run()`、`continue()`、事件订阅。

5. `packages/agent/src/agent-loop.ts`

   看 LLM/tool loop。重点看 `runLoop()`、`streamAssistantResponse()`、`executeToolCalls()`。

6. `packages/coding-agent/src/core/tools/`

   看 coding tools 的 schema、执行函数和文件修改约束。

7. `packages/coding-agent/src/core/extensions/`

   最后看扩展系统。Pi 很多能力不是写死在核心里，而是通过 extension event、command、tool、resource 注入。

## 6. 用一句话总结

Akashic 的核心是：每轮消息通过生命周期 pipeline，把长期记忆检索、工具调用、插件副作用、主动触达串成一个服务型个人 agent。

Pi 的核心是：围绕一个可恢复的 coding session，把项目上下文、session tree、compaction、工具执行、扩展事件、TUI/RPC/print 模式组合成一个编码 agent harness。

如果你正在学习 agent，建议把 Akashic 当作“个人助理型 agent”的参考，把 Pi 当作“编码任务型 agent”的参考。前者教你怎样做长期记忆和主动性，后者教你怎样做 session、工具循环、上下文压缩、扩展系统和 CLI 产品化。
