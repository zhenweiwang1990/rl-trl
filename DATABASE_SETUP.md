# 数据库设置指南

Link Search Agent 需要一个包含 LinkedIn 个人资料数据的 SQLite 数据库。本文档说明如何准备数据库。

## 方式 1: 从 PostgreSQL 导出（推荐）

如果你有 PostgreSQL 数据库中的 LinkedIn 个人资料数据，可以使用提供的脚本将其导出到 SQLite。

### 前置条件

1. **PostgreSQL 数据库**：包含 `dev_set` schema，其中有以下表：
   - `dev_set.profiles`
   - `dev_set.experiences`
   - `dev_set.educations`

2. **数据库连接信息**：
   - 主机地址
   - 端口（默认 5432）
   - 用户名
   - 密码
   - 数据库名

### 步骤

#### 1. 配置数据库连接

创建 `.env` 或 `env.linksearch` 文件：

```bash
# 复制示例文件
cp env.linksearch.example env.linksearch

# 编辑配置
nano env.linksearch
```

添加 PostgreSQL 连接信息：

```bash
# PostgreSQL 连接配置
PG_HOST=your-postgres-host.com
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your-password
PG_DATABASE=your-database
PG_SSLMODE=require
```

或者直接设置环境变量：

```bash
export PG_HOST="your-postgres-host.com"
export PG_PORT="5432"
export PG_USER="postgres"
export PG_PASSWORD="your-password"
export PG_DATABASE="your-database"
```

#### 2. 运行数据库生成脚本

```bash
cd /home/zhlmmc/rl-trl

# 确保 Docker 镜像已构建
bash scripts/build.sh

# 生成数据库
bash scripts/generate_database.sh
```

脚本会：
1. 连接到 PostgreSQL
2. 从 `dev_set` schema 导出数据
3. 创建 SQLite 数据库：`link_search_agent/data/profiles.db`
4. 显示导出统计信息

#### 3. 验证数据库

```bash
# 使用 sqlite3 命令行工具
sqlite3 link_search_agent/data/profiles.db

# 查看表
.tables

# 查看记录数
SELECT COUNT(*) FROM profiles;
SELECT COUNT(*) FROM experiences;
SELECT COUNT(*) FROM educations;

# 查看示例数据
SELECT * FROM profiles LIMIT 5;

# 退出
.quit
```

或使用 Python：

```python
import sqlite3

conn = sqlite3.connect('link_search_agent/data/profiles.db')
cursor = conn.cursor()

# 统计
cursor.execute("SELECT COUNT(*) FROM profiles")
print(f"Profiles: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM experiences")
print(f"Experiences: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM educations")
print(f"Educations: {cursor.fetchone()[0]}")

conn.close()
```

## 方式 2: 使用现有数据库

如果你已经有 SQLite 数据库：

### 1. 复制数据库文件

```bash
# 复制到项目目录
cp /path/to/your/profiles.db /home/zhlmmc/rl-trl/link_search_agent/data/profiles.db
```

### 2. 验证数据库结构

数据库必须包含以下表和字段：

**profiles 表**：
```sql
CREATE TABLE profiles (
    id TEXT PRIMARY KEY,
    name TEXT,
    about TEXT,
    summary TEXT,
    skills TEXT,
    linkedin_handle TEXT UNIQUE,
    updated_at TEXT,
    meta TEXT
);
```

**experiences 表**：
```sql
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    title TEXT,
    company TEXT,
    location TEXT,
    start_date TEXT,
    end_date TEXT,
    is_current INTEGER,
    employment_type TEXT,
    description TEXT,
    FOREIGN KEY (profile_id) REFERENCES profiles(id)
);
```

**educations 表**：
```sql
CREATE TABLE educations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    school TEXT,
    degree TEXT,
    field_of_study TEXT,
    start_date TEXT,
    end_date TEXT,
    grade TEXT,
    activities TEXT,
    description TEXT,
    FOREIGN KEY (profile_id) REFERENCES profiles(id)
);
```

### 3. 验证数据

```bash
sqlite3 link_search_agent/data/profiles.db "SELECT COUNT(*) FROM profiles;"
```

## 方式 3: 在 Docker 中生成

如果你想在 Docker 容器中生成数据库：

### 1. 启动容器

```bash
export PG_HOST="your-host"
export PG_USER="postgres"
export PG_PASSWORD="your-password"
export PG_DATABASE="your-database"

bash scripts/run.sh
```

### 2. 在容器内运行生成脚本

```bash
# 在容器内
cd /workspace
python scripts/export_to_sqlite.py
```

### 3. 数据库文件会保存到宿主机

由于容器挂载了 `link_search_agent/data` 目录，生成的数据库会自动保存到宿主机。

## 数据库使用

### 在训练中使用

设置环境变量指向数据库：

```bash
# 本地训练
export PROFILE_DB_PATH="/home/zhlmmc/rl-trl/link_search_agent/data/profiles.db"
python train_grpo_linksearch.py --mode masked

# Docker 训练（自动挂载）
export PROFILE_DB_PATH="/home/zhlmmc/rl-trl/link_search_agent/data/profiles.db"
bash scripts/run.sh
```

Docker 容器会自动挂载数据库到：`/workspace/link_search_agent/data/profiles.db`

### 测试数据库连接

```python
# 运行测试
python test_linksearch_setup.py

# 或手动测试
python -c "
from link_search_agent.tools import search_profile, read_profile
import os

# 设置数据库路径
os.environ['PROFILE_DB_PATH'] = 'link_search_agent/data/profiles.db'

# 测试搜索
result = search_profile('SELECT * FROM profiles LIMIT 5')
print(f'Found {result[\"rowCount\"]} profiles')

# 测试读取
if result['rows']:
    handle = result['rows'][0]['linkedin_handle']
    profile = read_profile(handle)
    print(f'Read profile: {profile[\"profile\"][\"name\"]}')
"
```

## 数据库维护

### 查看数据库大小

```bash
du -h link_search_agent/data/profiles.db
```

### 优化数据库

```bash
sqlite3 link_search_agent/data/profiles.db "VACUUM;"
```

### 备份数据库

```bash
cp link_search_agent/data/profiles.db link_search_agent/data/profiles.db.backup
```

### 清理和重新生成

```bash
rm link_search_agent/data/profiles.db
bash scripts/generate_database.sh
```

## 数据库内容要求

### 最小数据量

- 至少 1000 个 profiles 用于有意义的训练
- 推荐 5000+ profiles 用于良好的训练效果

### 数据质量

确保数据包含：
- **linkedin_handle**: 必需，唯一标识
- **name**: 必需，用于显示
- **summary**: 推荐，用于搜索
- **about**: 推荐，详细描述
- **skills**: 推荐，技能列表
- **experiences**: 推荐，工作经历
- **educations**: 可选，教育背景

### 数据格式

- **linkedin_handle**: 小写，不含 URL 前缀
  - 正确：`john-doe`
  - 错误：`https://linkedin.com/in/john-doe`

- **skills**: 逗号分隔的字符串
  - 示例：`Python, Machine Learning, Data Science`

- **dates**: 使用标准日期格式
  - 示例：`2020-01-01` 或 `2020-01`

## 故障排查

### 问题 1: 连接 PostgreSQL 失败

```bash
# 检查连接信息
echo $PG_HOST
echo $PG_USER
echo $PG_DATABASE

# 测试连接
docker run --rm \
    -e PG_HOST=$PG_HOST \
    -e PG_USER=$PG_USER \
    -e PG_PASSWORD=$PG_PASSWORD \
    -e PG_DATABASE=$PG_DATABASE \
    qwen3-grpo \
    python -c "import psycopg2; conn = psycopg2.connect(host='$PG_HOST', user='$PG_USER', password='$PG_PASSWORD', database='$PG_DATABASE'); print('Connected!')"
```

### 问题 2: 数据库文件未生成

```bash
# 检查 Docker 容器日志
docker logs -f <container-id>

# 检查目录权限
ls -la link_search_agent/data/

# 手动创建目录
mkdir -p link_search_agent/data
```

### 问题 3: 数据库损坏

```bash
# 检查数据库完整性
sqlite3 link_search_agent/data/profiles.db "PRAGMA integrity_check;"

# 如果损坏，重新生成
rm link_search_agent/data/profiles.db
bash scripts/generate_database.sh
```

### 问题 4: 训练时找不到数据库

```bash
# 检查环境变量
echo $PROFILE_DB_PATH

# 检查文件是否存在
ls -la $PROFILE_DB_PATH

# 使用绝对路径
export PROFILE_DB_PATH="/home/zhlmmc/rl-trl/link_search_agent/data/profiles.db"
```

## 性能优化

### 添加索引

如果数据库很大，添加额外的索引：

```sql
-- 为常用搜索字段添加索引
CREATE INDEX IF NOT EXISTS idx_profiles_name ON profiles(name);
CREATE INDEX IF NOT EXISTS idx_profiles_summary ON profiles(summary);
CREATE INDEX IF NOT EXISTS idx_experiences_company ON experiences(company);
CREATE INDEX IF NOT EXISTS idx_experiences_title ON experiences(title);
CREATE INDEX IF NOT EXISTS idx_educations_school ON educations(school);
```

### 启用 WAL 模式

```sql
-- Write-Ahead Logging 模式提高并发性能
PRAGMA journal_mode=WAL;
```

### 优化查询

训练时使用的搜索查询应该：
- 使用索引字段
- 添加 LIMIT 子句
- 使用 WHERE 条件缩小范围

## 下一步

数据库准备好后：

1. **测试设置**：
   ```bash
   python test_linksearch_setup.py
   ```

2. **快速测试训练**：
   ```bash
   export TRAIN_DATASET_SIZE="50"
   python train_grpo_linksearch.py --mode simple
   ```

3. **完整训练**：
   ```bash
   bash scripts/run.sh  # 启动 Docker
   ./scripts/train_linksearch.sh --mode masked
   ```

## 相关文档

- **Docker 训练指南**: `DOCKER_LINKSEARCH.md`
- **Link Search 文档**: `LINKSEARCH_README.md`
- **主文档**: `README.md`
