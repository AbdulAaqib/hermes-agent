---
sidebar_position: 16
---



## 前提条件

- **Google AI Studio API 密钥** — 在 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 创建

:::tip API 密钥路径
:::

## 快速开始

```bash
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

hermes model
# → 选择 "More providers..." → "Google AI Studio"
# → 选择一个模型

# 开始对话
hermes chat
```


```yaml
model:
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 配置

运行 `hermes model` 后，`~/.hermes/config.yaml` 将包含：

```yaml
model:
  base_url: https://generativelanguage.googleapis.com/v1beta
```

`~/.hermes/.env` 中：

```bash
GOOGLE_API_KEY=...
```


推荐使用的端点为：

```text
https://generativelanguage.googleapis.com/v1beta
```


- 流式响应 → 供 Hermes 循环使用的 OpenAI 格式流式数据块


:::

### 优先使用原生端点

Google 还提供了 OpenAI 兼容端点：

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```



```bash
```

## 可用模型


| 模型 | ID | 说明 |
|------|----|------|

模型可用性会随时间变化。如果某个模型消失或未对你的密钥启用，请重新运行 `hermes model` 并从当前列表中选择。

:::info 模型 ID
:::

### 最新别名


| 别名 | 当前指向 | 说明 |
|------|----------|------|

```yaml
model:
  base_url: https://generativelanguage.googleapis.com/v1beta
```




常用评估 ID 包括：

| 模型 | ID | 说明 |
|------|----|------|
| Gemma 4 31B IT | `gemma-4-31b-it` | 较大的 Gemma 模型；适用于兼容性和质量评估 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 可用时的较小活跃参数变体 |


如需使用选择器中隐藏的 Gemma 模型，请直接在配置中指定：

```yaml
model:
  default: gemma-4-31b-it
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 会话中途切换模型

在对话中使用 `/model` 命令：

```text
/model gemma-4-31b-it
```


## 诊断

```bash
hermes doctor
```

doctor 命令检查：

- 已配置的 provider 凭据是否可以解析

## Gateway（消息平台）


```bash
hermes gateway setup
hermes gateway start
```


## 故障排查


Hermes 找不到可用的 API 密钥。请将以下任一项添加到 `~/.hermes/.env`：

```bash
GOOGLE_API_KEY=...
# 或
```

然后重新运行 `hermes model`。

### "This Google API key is on the free tier"


请为与密钥关联的 Google Cloud 项目启用计费，必要时重新生成密钥，然后运行：

```bash
hermes model
```

### "404 model not found"


### Gemma 模型未显示在 `hermes model` 中

Hermes 默认可能会在选择器中隐藏低吞吐量的 Gemma 模型。如果你有意评估某个模型，请直接在 `~/.hermes/config.yaml` 中设置模型 ID。

### Gemma 出现 "429 quota exceeded"


### 已配置 OpenAI 兼容端点

检查 `~/.hermes/.env` 中是否存在：

```bash
```

将其修改为原生端点或删除该覆盖项：

```bash
```

### 工具调用因 schema 错误而失败


## 相关链接

- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
- [Fallback Providers](/user-guide/features/fallback-providers)
- [AWS Bedrock](/guides/aws-bedrock) — 使用 AWS 凭据的原生云 provider 集成