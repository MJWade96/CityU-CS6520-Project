每次执行任务时，都要严格遵守个人规则中的要求

## 1st Interim Report 撰写要求

### Page Requirements

- 15~18 pages

### Content Requirements

(1) Introduction – Clearly describe background of the selected topic and the objectives and why the topic is useful and potential outcome (prototype of application software, performance analysis or theoretical results, etc.).  
(2) Related work – Illustrate the problem as associated with existing and related systems/solutions. Description of advantages and disadvantages between existing solutions. Discussion of the relationship between the existing solutions and the proposed solution.  
(3) System modeling and structure – the proposed solution. Discussion of the justifications of the design choices, and how they address existing limitations.  
(4) The methodology and algorithms you will design/implement for the proposed solution.  
(5) Preliminary performance analysis or experiments of your algorithm/system.  
(6) Milestones and overall schedule for your project.  
(7) Work to be completed for the next report.  
(8) References.

### Specific Requirements

- References:
  - 引用文献数量至少为 18 篇

## Report 格式要求

- Cover Page: Student name, student number, project title and supervisor name should be written on a cover page of each report.
- Margins (top/bottom/left/right): 2.54 cm
- Font Name: Times New Roman (font size: 12)
- Line Spacing: Single Line

## Project Assessment Timeline

| **Item**           | **Deadline**                                       |
| ------------------ | -------------------------------------------------- |
| Topic Selection    | Semester B, Week 1, Tuesday<br>**13 January 2026** |
| Project Proposal   | Semester B, Week 3, Tuesday<br>**27 January 2026** |
| 1st Interim Report | Semester B, Week 12, Wednesday<br>**8 April 2026** |
| 2nd Interim Report | Summer Term, Week 1, Tuesday<br>**9 June 2026**    |
| Final Report       | Summer Term, Week 7, Tuesday<br>**21 July 2026**   |
| Presentation       | Student Revision Week<br>**27–31 July 2026**       |

# 项目信息

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
