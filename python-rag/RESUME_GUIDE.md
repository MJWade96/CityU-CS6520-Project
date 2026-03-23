# 长时间运行任务断点续传使用指南

## 问题背景

医学 RAG 评估任务通常需要数小时才能完成（特别是处理数百道 MedQA 题目时）。在长时间运行过程中，可能会遇到以下中断情况：

- 网络连接中断（API 调用失败）
- 系统崩溃或断电
- 内存不足导致进程被终止
- 手动中断（Ctrl+C）

**之前的问题**：一旦中断，必须从头开始，浪费大量时间和 API 配额。

## 解决方案

现在项目支持**断点续传**功能，具有以下特性：

✅ **自动保存进度**：每处理完一道题目，立即保存检查结果  
✅ **自动恢复**：重新启动时自动从上次中断的位置继续  
✅ **错误容忍**：即使发生错误，已处理的进度也不会丢失  
✅ **透明操作**：无需手动干预，自动检测和恢复  

## 支持的任务

以下评估脚本已支持断点续传：

1. **complete_eval.py** - 完整医学 RAG 评估（开发集 + 测试集）
2. **enhanced_eval.py** - 增强版医学 RAG 评估（包含所有优化）

## 使用方法

### 方法 1：直接运行（推荐）

直接运行评估脚本，系统会自动检测并恢复中断的任务：

```bash
# 完整评估
python complete_eval.py

# 增强版评估
python enhanced_eval.py
```

**如果之前中断过**：
```
⚠️  Found interrupted evaluation: 150/500
🔄 Resuming test set evaluation from question 151
```

**如果是首次运行**：
```
✓ No checkpoint found
Starting fresh evaluation...
```

### 方法 2：使用智能启动脚本

使用专门的启动脚本，提供更多控制选项：

```bash
# 自动检测并选择要恢复的任务
python run_with_resume.py --auto

# 运行特定脚本（自动恢复）
python run_with_resume.py complete_eval

# 运行特定脚本（强制从头开始）
python run_with_resume.py complete_eval --no-resume
```

### 方法 3：检查中断状态

查看是否有中断的任务：

```bash
python run_with_resume.py --auto
```

输出示例：
```
📋 Found 1 interrupted evaluation(s):

1. complete_eval
   Dataset: Test Set
   Progress: 150/500
   Timestamp: 2026-03-22T14:30:45.123456

Which evaluation to resume?
Enter number (1-1) or 'all' to resume all:
```

## 工作原理

### 进度保存机制

1. **检查点文件**：保存在 `results/evaluation/` 目录下
   - `checkpoint_complete_eval.json` - complete_eval 的检查点
   - `checkpoint_enhanced_eval.json` - enhanced_eval 的检查点
   - `checkpoint.backup.json` - 备份检查点（防止损坏）

2. **保存内容**：
   ```json
   {
     "timestamp": "2026-03-22T14:30:45.123456",
     "dataset_name": "Test Set",
     "total_questions": 500,
     "processed_questions": 150,
     "correct_count": 75,
     "total_count": 150,
     "elapsed_time": 3600.5,
     "results": [...],  // 详细结果
     "config": {...}    // 配置信息
   }
   ```

3. **保存频率**：每处理完一道题目立即保存

### 恢复流程

```
1. 启动脚本
   ↓
2. 检查是否存在检查点文件
   ↓
3. 如果存在：
   - 加载检查点
   - 显示进度信息
   - 从上次中断的位置继续
4. 如果不存在：
   - 从头开始
   ↓
5. 正常运行，定期保存进度
   ↓
6. 完成后清除检查点
```

## 实际使用示例

### 场景 1：API 调用中断

**情况**：评估进行到一半，API 返回错误

```bash
# 第一次运行
python complete_eval.py

# 输出：
# Question 150/500 | Accuracy: 0.5200 | Speed: 0.85 q/s
# ERROR on question 151: API timeout
# 💾 Progress saved: 150/500 (30.0%) | Accuracy: 0.5200

# 程序退出

# 修复网络后，再次运行
python complete_eval.py

# 输出：
# ⚠️  Found interrupted evaluation: 150/500
# 🔄 Resuming test set evaluation from question 151
# Question 151/500 | Accuracy: 0.5200 | Speed: 0.84 q/s
# ...
# ✅ Evaluation completed successfully!
```

### 场景 2：手动中断

**情况**：需要临时停止，稍后继续

```bash
# 按 Ctrl+C 中断
# 💾 Progress saved: 200/500 (40.0%)

# ...处理其他事情...

# 稍后继续
python complete_eval.py

# 自动从 201 题继续
```

### 场景 3：系统崩溃恢复

**情况**：电脑崩溃重启后

```bash
# 重启后直接运行
python complete_eval.py

# 系统自动检测到最后一次保存的进度
# 从崩溃前的位置继续
```

## 高级功能

### 手动清理检查点

如果需要强制从头开始：

```bash
# 方法 1：使用 --no-resume 参数
python run_with_resume.py complete_eval --no-resume

# 方法 2：手动删除检查点文件
rm results/evaluation/checkpoint_complete_eval.json
rm results/evaluation/checkpoint_enhanced_eval.json
```

### 查看检查点详情

```bash
# 使用 Python 查看
python -c "
import json
with open('results/evaluation/checkpoint_complete_eval.json', 'r') as f:
    data = json.load(f)
    print(f'Dataset: {data[\"dataset_name\"]}')
    print(f'Progress: {data[\"processed_questions\"]}/{data[\"total_questions\"]}')
    print(f'Accuracy: {data[\"correct_count\"]}/{data[\"total_count\"]}')
    print(f'Timestamp: {data[\"timestamp\"]}')
"
```

## 技术实现

### 核心模块

**progress_manager.py** - 进度管理核心模块

主要类：
- `EvaluationProgressManager` - 进度管理器
- `CheckpointData` - 检查点数据结构

关键方法：
- `save_checkpoint()` - 保存检查点
- `load_checkpoint()` - 加载检查点
- `should_resume()` - 判断是否应该恢复
- `clear_checkpoint()` - 清除检查点

### 代码集成

在评估脚本中的使用：

```python
from app.rag.progress_manager import EvaluationProgressManager

# 初始化进度管理器
progress_mgr = EvaluationProgressManager(output_dir="./results/evaluation")

# 在循环中保存进度
for i, question in enumerate(questions, 1):
    result = evaluate_question(question)
    results.append(result)
    
    # 每处理完一道题保存一次
    progress_mgr.save_checkpoint(
        dataset_name="Test Set",
        total_questions=len(questions),
        processed_questions=i,
        results=results,
        correct_count=correct,
        total_count=total,
        elapsed_time=time.time() - start_time,
        config=config_dict,
        script_name="complete_eval",
    )

# 完成后清除检查点
progress_mgr.clear_checkpoint(script_name="complete_eval")
```

## 注意事项

1. **检查点文件会占用磁盘空间**
   - 每个检查点约几 MB（取决于结果详细程度）
   - 任务完成后会自动清除

2. **确保有足够的磁盘空间**
   - 建议至少保留 100MB 可用空间

3. **不要同时运行多个相同任务**
   - 可能导致检查点文件冲突

4. **定期检查点可能影响性能**
   - 每道题保存一次，影响很小（<10ms）
   - 与 API 调用时间相比可忽略不计

## 故障排除

### 问题 1：检查点无法加载

**症状**：
```
Warning: Could not load checkpoint: JSON decode error
```

**解决方案**：
```bash
# 删除损坏的检查点
rm results/evaluation/checkpoint_*.json
# 从头开始
python complete_eval.py
```

### 问题 2：进度没有保存

**症状**：中断后重新运行，发现从头开始

**可能原因**：
- 输出目录没有写权限
- 磁盘空间不足
- 程序在处理第一题前就崩溃了

**解决方案**：
```bash
# 检查目录权限
ls -la results/evaluation/

# 检查磁盘空间
df -h

# 手动创建目录
mkdir -p results/evaluation
```

### 问题 3：恢复后重复处理

**症状**：恢复后，发现某些题目被重复处理

**解决方案**：
- 这是正常现象，最后一道题可能会被重新处理
- 结果会覆盖，不影响最终准确性

## 最佳实践

1. **定期保存**：系统已自动每道题保存，无需手动操作

2. **使用稳定网络**：减少 API 调用失败导致的重跑

3. **监控系统资源**：避免内存不足导致崩溃

4. **保留日志**：便于排查问题

5. **使用开发集测试**：先用小规模数据测试配置

## 性能影响

| 操作 | 时间开销 |
|------|---------|
| 保存检查点（每题） | ~5-10ms |
| 加载检查点 | ~50-100ms |
| 恢复流程 | ~1-2 秒 |

**总结**：断点续传功能对性能影响极小（<0.1%），但可以避免长时间重跑，收益巨大。

## 未来改进

- [ ] 支持可配置的保存频率（每 N 题保存一次）
- [ ] 支持多个检查点版本回滚
- [ ] 添加进度可视化（Web UI）
- [ ] 支持分布式评估的进度同步

---

**更新日期**：2026-03-22  
**版本**：v1.0
