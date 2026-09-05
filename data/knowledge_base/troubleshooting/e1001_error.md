---
title: E1001 错误码详解
category: troubleshooting
source: troubleshooting/e1001_error.md
version: 1.0
---

# E1001 错误码详解

## 错误信息
```
E1001: Invalid authentication credentials
```

## 错误含义
认证凭据无效，通常发生在 API 调用时。

## 可能原因

### 1. API Key 错误
- API Key 输入错误
- API Key 已被撤销或过期
- 使用了错误的 API Key（测试环境 vs 生产环境）

### 2. Token 过期
- Access Token 已过期（有效期 24 小时）
- Refresh Token 已失效
- 需要重新获取 Token

### 3. 权限不足
- API Key 权限范围不足
- 缺少必要的 Scope 授权
- 账号已被禁用

## 解决方案

1. **检查 API Key**
   ```bash
   # 确认 API Key 是否正确
   curl -H "Authorization: Bearer YOUR_API_KEY" \
        https://api.cloud Desk.com/v1/account
   ```

2. **重新获取 Token**
   - 登录控制台 → 设置 → API 密钥
   - 点击「重新生成」获取新的 API Key

3. **检查权限**
   - 确认 API Key 具有所需的操作权限
   - 联系管理员确认账号状态

## 预防措施
- 定期轮换 API Key
- 将 API Key 存储在环境变量中，不要硬编码
- 监控 API 调用日志，及时发现异常
