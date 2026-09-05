---
title: E1001 错误详细排查
category: troubleshooting
source: troubleshooting/e1001_detailed.md
version: 1.0
---

# E1001 错误详细排查指南

## 错误描述
E1001 表示认证失败，API 无法验证请求者的身份。

## 排查步骤

### Step 1: 确认 API Key 正确
```bash
# 检查 API Key 是否设置正确
echo $CLOUDDESK_API_KEY
# 应该是 sk-xxxxx 格式，长度 32-64 位
```

### Step 2: 确认 API Key 未过期
- 登录控制台 → API 管理
- 查看 Key 的创建时间和最后使用时间
- 过期的 Key 需要重新生成

### Step 3: 确认使用正确的环境
- 测试环境 Key：sk-test-xxxxx
- 生产环境 Key：sk-prod-xxxxx
- 不要混用不同环境的 Key

### Step 4: 确认请求头格式正确
```
Authorization: Bearer YOUR_API_KEY
```
注意：Bearer 和 Key 之间有一个空格

### Step 5: 检查账号状态
- 账号是否被禁用
- 是否有未完成的实名认证
- 是否有欠费停机

## 代码示例
```python
import httpx

client = httpx.Client(
    base_url="https://api.clouddesk.com/v1",
    headers={"Authorization": f"Bearer {api_key}"}
)
response = client.get("/account")
```
