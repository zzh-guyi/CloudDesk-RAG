---
title: 认证 API 文档
category: api_docs
source: api_docs/auth_api.md
version: 1.0
---

# CloudDesk 认证 API

## 获取 Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "grant_type": "password",
  "username": "user@example.com",
  "password": "your_password"
}
```

**响应：**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "refresh_token": "dGhpcyBpcy..."
}
```

## 刷新 Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "dGhpcyBpcy..."
}
```

## 废除 Token

```http
POST /api/v1/auth/revoke
Authorization: Bearer ACCESS_TOKEN
```

## 错误码

| 错误码 | 含义 |
|--------|------|
| E1001 | 认证凭据无效 |
| E1002 | Token 已过期 |
| E1003 | Token 已被废除 |
| E1004 | 账号已被禁用 |

## 注意事项

- Access Token 有效期 24 小时
- Refresh Token 有效期 30 天
- 建议安全存储 Token，不要硬编码
