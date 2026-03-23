"""
简化测试脚本 - 验证 evaluate_no_rag.py 的核心功能
"""

import asyncio
import time
import json
import sys
from pathlib import Path

# 导入必要的模块
sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI


async def test_simple():
    """简单测试"""
    print("开始简单测试...")

    # 创建客户端
    client = AsyncOpenAI(
        api_key="6fcecb364d0647d2883e7f1d3f19d5b9",
        base_url="https://wishub-x6.ctyun.cn/v1",
    )

    # 测试一个简单的问题
    test_question = "What is the capital of France?"

    try:
        print(f"发送问题: {test_question}")
        completion = await client.chat.completions.create(
            model="2656053fa69c4c2d89c5a691d9d737c3",
            messages=[{"role": "user", "content": test_question}],
            temperature=0.1,
            max_tokens=50,
        )

        response = completion.choices[0].message.content
        print(f"收到响应: {response}")
        print("测试成功！")

    except Exception as e:
        print(f"测试失败: {e}")


async def test_rate_limiter():
    """测试速率限制器"""
    print("\n测试速率限制器...")

    from evaluate_no_rag import RateLimiter

    rate_limiter = RateLimiter(requests_per_second=2.0, burst=3)

    async def test_task(i):
        print(f"任务 {i}: 开始")
        await rate_limiter.acquire()
        print(f"任务 {i}: 获取到令牌")
        await asyncio.sleep(0.1)
        print(f"任务 {i}: 完成")

    tasks = [test_task(i) for i in range(5)]
    await asyncio.gather(*tasks)
    print("速率限制器测试完成")


async def main():
    """主函数"""
    print("=" * 60)
    print("简化测试脚本")
    print("=" * 60)

    # 测试速率限制器
    await test_rate_limiter()

    # 测试 API 连接
    await test_simple()

    print("\n所有测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
