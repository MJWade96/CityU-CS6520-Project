# 必须遵守

- 完成任务之后删除过程性文件、临时测试文件
- 阶段性提交到 GitHub 仓库，每个阶段提交一个 commit
- 执行过程中的任意步骤无法完成，请如实告诉我，不要使用有缺陷、不完整的方法实现
- DRY 原则
- 在生成、修改或重构任何代码时，若检测到逻辑、结构或文本层面的重复（包括跨文件重复），应优先通过以下方式消除冗余：

1. **提取函数/方法**：将重复逻辑、重复配置封装为具有清晰职责的可复用单元。
2. **创建工具类/模块**：对通用功能（如日期处理、API 调用、验证等）集中管理。
3. **使用配置或数据驱动**：若重复源于相似但参数不同的代码块，改用配置表、映射或策略模式。

**约束条件**：

- 不得以牺牲可读性或可维护性为代价追求“不重复”。
- 若重复代码处于不同业务上下文且未来可能独立演化，允许暂时保留，但需添加 `// TODO: Consider deduplication` 注释并说明理由。
- 在首次引入可能被复用的逻辑时，即应考虑其通用性，而非等待重复出现两次以上才重构。

**输出要求**：

- 所有生成的代码必须附带简要注释，说明其设计如何避免重复；若项目已有类似实现，应优先复用而非新建。

# 禁止

- 在脚本中使用命令行参数（正确做法：在脚本中设置变量存储参数值）
- 使用回退机制

# 最佳实践

- 并发机制采用流水线模式：任一任务完成后立即补充一个新任务。

# 项目信息

libgomp: Invalid value for environment variable OMP_NUM_THREADS

- 天翼云模型调用参数说明：
  - enable_thinking: 是否开启思考模式，默认值为 True，使用 OpenAI 协议时，需要在 extra_body 中设置
- 天翼云 API 调用示例：

```python
import openai
from openai import OpenAI

# 需要补充一下属性
baseUrl = "https://wishub-x6.ctyun.cn/v1/"
# 从环境变量获取API密钥，如果没有设置，也可以直接终端执行export XIRANG_app_key="xxx"
appKey="your_app_key" # 替换为实际的App Key
model_id = "xirang_model_id" # 替换为实际的modelId
prompt="你好，3.01+103.1等于多少" #对话问题，可替换

def main():
    client = OpenAI(base_url=baseUrl, api_key=appKey)
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=False
        )
        print(res.choices[0].message.content or "", end="", flush=True)

    except openai.APIStatusError as e:
        print(f"APIStatusError: {e.status_code}, {e.message}, {e.body}")
    except openai.APIError as e:
        print(f"APIError: {e.body}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()

```
