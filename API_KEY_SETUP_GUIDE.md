# API密钥管理指南

本指南介绍如何为BMC Benchmark项目安全地存储和管理API密钥。

## 🔑 支持的存储方式

### 1. 配置管理器（推荐）
使用内置的配置管理系统，提供加密存储和便捷管理。

```bash
# 交互式设置
python setup_api_keys.py

# 或者
python -m src.scripts.config
```

### 2. 环境变量
传统的环境变量方式，适合CI/CD和临时使用。

```bash
# Gemini API密钥
export GOOGLE_API_KEY='your-gemini-api-key-here'

# OpenAI API密钥  
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 3. 直接传参
在代码中直接传递API密钥，适合脚本和测试。

```python
from src.scripts.Experiments.llm_refit_curve import analyze_equation_with_llm

# 使用Gemini
results = analyze_equation_with_llm("instance_name", api_key="your-key", provider="gemini")
```

## 🛠️ 配置管理器详细使用

### 交互式设置

```bash
python setup_api_keys.py
```

这将启动交互式菜单，引导您完成以下操作：
- 设置Gemini API密钥
- 设置OpenAI API密钥
- 选择加密选项
- 测试API密钥有效性
- 设置默认提供商

### 命令行操作

```bash
# 查看当前配置
python setup_api_keys.py --list

# 测试特定提供商的API密钥
python setup_api_keys.py --test gemini
python setup_api_keys.py --test openai

# 移除API密钥
python setup_api_keys.py --remove gemini
python setup_api_keys.py --remove openai

# 设置默认提供商
python setup_api_keys.py --set-default gemini
```

### 程序化使用

```python
from src.scripts.config import get_config_manager

# 获取配置管理器
config = get_config_manager()

# 设置API密钥（加密存储）
config.set_api_key("gemini", "your-api-key", encrypt=True)

# 获取API密钥
api_key = config.get_api_key("gemini")

# 查看配置状态
config.list_stored_keys()
```

## 🔒 安全特性

### 加密存储
- **强加密**：使用Fernet对称加密（如果安装了cryptography包）
- **基础编码**：如果没有cryptography，使用base64编码作为备选
- **权限保护**：配置文件设置为仅用户可读（权限600）

### 多重备选
API密钥查找优先级：
1. 环境变量（最高优先级）
2. 加密存储的密钥
3. 基础编码的密钥

### 安全建议
- 安装cryptography包以获得真正的加密：`pip install cryptography`
- 不要将API密钥提交到版本控制系统
- 定期轮换API密钥
- 使用环境变量在生产环境中

## 📁 文件位置

配置文件存储在用户主目录下：

```
~/.bmc_benchmark/
├── config.json          # 主配置文件
├── keys.enc             # 加密的密钥文件（如果使用）
└── .keyfile             # 加密密钥（如果使用）
```

## 🚀 获取API密钥

### Gemini API密钥
1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登录Google账户
3. 点击"Create API Key"
4. 复制生成的API密钥

### OpenAI API密钥
1. 访问 [OpenAI平台](https://platform.openai.com/api-keys)
2. 登录OpenAI账户
3. 点击"Create new secret key"
4. 复制生成的API密钥

## 📊 使用示例

### 基本使用

```python
from src.scripts.Experiments.llm_refit_curve import analyze_equation_with_llm

# 使用配置管理器中的默认设置
results = analyze_equation_with_llm("oc8051gm0caddr")

# 指定提供商
results = analyze_equation_with_llm("oc8051gm0caddr", provider="gemini")
```

### 批量处理

```python
from src.scripts.Experiments.llm_refit_curve import batch_analyze_equations

instance_names = ["instance1", "instance2", "instance3"]
results = batch_analyze_equations(instance_names)
```

### 自定义分析器

```python
from src.scripts.Experiments.llm_refit_curve import LLMEquationAnalyzer

# 使用配置管理器
analyzer = LLMEquationAnalyzer()

# 或指定参数
analyzer = LLMEquationAnalyzer(provider="gemini", model="gemini-pro")
```

## 🔧 故障排除

### 常见问题

**Q: 提示"No API key found"**
A: 运行 `python setup_api_keys.py` 设置API密钥，或设置环境变量

**Q: API密钥测试失败**
A: 检查密钥是否正确，网络连接是否正常，API服务是否可用

**Q: 加密存储不工作**
A: 安装cryptography包：`pip install cryptography`

**Q: 权限错误**
A: 确保有权限写入用户主目录，或指定自定义配置目录

### 重置配置

如需重置所有配置：

```bash
# 删除配置目录
rm -rf ~/.bmc_benchmark/

# 重新设置
python setup_api_keys.py
```

## 📚 相关文档

- [LLM分析模块文档](src/scripts/Experiments/README_llm_analysis.md)
- [项目主要README](README.md)
- [API参考文档](src/scripts/Experiments/README_llm_analysis.md#api-reference)

## 🆘 获取帮助

如果遇到问题：
1. 检查此指南的故障排除部分
2. 运行 `python setup_api_keys.py --list` 查看当前状态
3. 使用 `python setup_api_keys.py --test <provider>` 测试API密钥
