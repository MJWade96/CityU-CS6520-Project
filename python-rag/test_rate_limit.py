"""
测试速率限制器是否正常工作
"""

import asyncio
import time


class SimpleRateLimiter:
    """简化版速率限制器"""

    def __init__(self, requests_per_second: float, burst: int = 10):
        self.requests_per_second = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()

    async def acquire(self):
        """获取令牌"""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst, self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.requests_per_second
            print(f"等待 {wait_time:.2f} 秒...")
            await asyncio.sleep(wait_time)


async def test_task(task_id: int, rate_limiter):
    """测试任务"""
    print(f"任务 {task_id}: 开始")
    await rate_limiter.acquire()
    print(f"任务 {task_id}: 获取到令牌，执行中...")
    await asyncio.sleep(0.1)  # 模拟工作
    print(f"任务 {task_id}: 完成")


async def main():
    """主函数"""
    print("测试简化速率限制器")
    
    # 使用较低的速率限制进行测试
    rate_limiter = SimpleRateLimiter(requests_per_second=2.0, burst=5)
    
    # 创建多个任务
    tasks = [test_task(i, rate_limiter) for i in range(10)]
    
    print("开始执行任务...")
    start_time = time.time()
    
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"所有任务完成，耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())