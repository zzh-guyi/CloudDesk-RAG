---
title: 项目 API 文档
category: api_docs
source: api_docs/project_api.md
version: 1.0
---

# CloudDesk 项目 API

## 创建项目

```http
POST /api/v1/projects
Authorization: Bearer ACCESS_TOKEN
Content-Type: application/json

{
  "name": "项目名称",
  "description": "项目描述",
  "template": "blank"
}
```

**响应：**
```json
{
  "id": "proj_abc123",
  "name": "项目名称",
  "description": "项目描述",
  "created_at": "2024-01-01T00:00:00Z",
  "owner": "user@example.com"
}
```

## 获取项目列表

```http
GET /api/v1/projects?page=1&limit=20
Authorization: Bearer ACCESS_TOKEN
```

## 获取项目详情

```http
GET /api/v1/projects/{project_id}
Authorization: Bearer ACCESS_TOKEN
```

## 更新项目

```http
PUT /api/v1/projects/{project_id}
Authorization: Bearer ACCESS_TOKEN
Content-Type: application/json

{
  "name": "新名称",
  "description": "新描述"
}
```

## 删除项目

```http
DELETE /api/v1/projects/{project_id}
Authorization: Bearer ACCESS_TOKEN
```

## 错误码

| 错误码 | 含义 |
|--------|------|
| E2001 | 项目不存在 |
| E2002 | 项目名称已存在 |
| E2003 | 资源未找到 |
| E2004 | 权限不足 |
