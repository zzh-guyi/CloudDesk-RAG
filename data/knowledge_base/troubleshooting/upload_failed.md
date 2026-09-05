---
title: 上传失败错误排查
category: troubleshooting
source: troubleshooting/upload_failed.md
version: 1.0
---

# 上传失败错误排查

## 问题描述
上传文件时失败，提示各种错误信息。

## 常见错误码

### E4001: File too large
- 文件大小超出限制
- 免费版：10MB，团队版：100MB，企业版：500MB
- 解决方案：压缩文件或升级套餐

### E4002: Unsupported format
- 文件格式不在支持列表中
- 支持的格式：PDF, DOC, DOCX, TXT, MD, JPG, PNG, CSV, XLSX
- 解决方案：转换文件格式

### E4003: Upload timeout
- 上传超时（超过 300 秒）
- 解决方案：检查网络，分批次上传

### E4004: Storage full
- 存储空间已满
- 免费版：1GB，团队版：10GB，企业版：100GB+
- 解决方案：清理无用文件或升级套餐

### E4005: Invalid file content
- 文件内容损坏或加密
- 解决方案：重新下载或联系文件提供方

## 排查步骤

1. 确认文件大小是否在限制内
2. 确认文件格式是否支持
3. 检查网络连接是否稳定
4. 确认存储空间是否充足
5. 检查文件是否损坏

## 解决方案
- 小文件分批上传
- 使用压缩格式
- 清理不需要的文件
- 升级套餐获取更大空间
