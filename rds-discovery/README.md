<<<<<<< HEAD
# Strands RDS Discovery Tool v2.1.2

**SQL Server to AWS RDS Migration Assessment with Pricing Integration - Strands Tool**

A production-ready **Strands tool** that provides comprehensive SQL Server compatibility assessment for AWS RDS migration planning with **PowerShell-compatible CSV output**, **cost estimation**, and **triple output format** (CSV + JSON + LOG).

## **🎯 Overview**

This tool enables comprehensive SQL Server assessment for AWS RDS migration, providing detailed analysis, migration recommendations, AWS instance sizing with pricing, and complete documentation through three output files.

### **Key Features**
- **Simplified Usage**: No action parameters - just run with server file like original PowerShell script
- **10% Tolerance Logic**: Consistent tolerance matching in both AWS API and fallback modes
- **PowerShell CSV Output**: Generates identical `RdsDiscovery.csv` format as original PowerShell tool
- **Cost Estimation**: Hourly and monthly pricing for recommended AWS instances
- **Triple Output**: CSV + JSON + LOG files with matching timestamps
- **Real SQL Server Data**: All data collected from live SQL Server queries (no mock data)
- **PowerShell-Compatible Storage**: Uses `xp_fixeddrives` logic matching PowerShell behavior exactly
- **AWS Instance Sizing**: Intelligent RDS instance recommendations with scaling explanations
- **Comprehensive Analysis**: 25+ SQL Server feature compatibility checks
- **Production Ready**: Enterprise-grade error handling and performance monitoring

## **🚀 Quick Start**

### **Strands Tool Usage**

This tool is now a **Strands tool** and can be used within the Strands framework:

```python
# Import as Strands tool
from src.rds_discovery import strands_rds_discovery

# Use within Strands conversations
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='sql',
    username='your_username',
    password='your_password'
)
```

### **Strands AI Integration**
You can now use natural language with Strands AI:
- *"Assess SQL Server 3.81.26.46 for RDS migration"*
- *"Generate RDS discovery report for my servers"*
- *"What AWS instance size is recommended for my SQL Server?"*

### Installation

```bash
git clone <repository-url>
cd strands-rds-discovery
source venv/bin/activate  # Linux/Mac: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Basic Usage

```python
from rds_discovery import strands_rds_discovery

# Windows Authentication
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='windows'
)

# SQL Server Authentication  
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='sql',
    username='your_username',
    password='your_password'
)
```

### Server File Format

Create a CSV file with your SQL Server instances:
```csv
server_name
server1.domain.com
192.168.1.100
sql-prod-01
```

## **📋 Complete Run Guide**

See [RUN_GUIDE.md](RUN_GUIDE.md) for detailed step-by-step instructions including:
- Virtual environment setup
- Authentication options
- Troubleshooting common issues
- Output file explanations

## **📊 Output Files**

The tool generates three files with matching timestamps (clean, single-log approach):

- **`RDSdiscovery_[timestamp].csv`** - PowerShell-compatible results
- **`RDSdiscovery_[timestamp].json`** - Detailed JSON with pricing and metadata  
- **`RDSdiscovery_[timestamp].log`** - Assessment log (no persistent log files)

## **💰 AWS Pricing Integration**

- Real-time AWS RDS pricing via API
- Fallback pricing when API unavailable
- Monthly cost estimates for migration planning
- Instance scaling explanations (exact_match, within_tolerance, scaled_up, fallback)

## **🔧 Parameters**

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `input_file` | Yes | CSV file with server names | - |
| `auth_type` | No | 'windows' or 'sql' | 'windows' |
| `username` | Conditional | SQL username (required if auth_type='sql') | None |
| `password` | Conditional | SQL password (required if auth_type='sql') | None |
| `timeout` | No | Connection timeout in seconds | 30 |
```

### Basic Usage

```python
from src.rds_discovery import strands_rds_discovery

# 1. Create server template
result = strands_rds_discovery(action="template", output_file="servers.csv")

# 2. Edit servers.csv with your SQL Server names/IPs

# 3. Run assessment
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv", 
    auth_type="sql",
    username="your_username",
    password="your_password",
    output_file="assessment_results"
)
```

## **📄 Output Files**

The tool generates **3 files** with matching timestamps:

### 1. CSV File (`RDSdiscovery_[timestamp].csv`)
- **PowerShell-compatible** format with 41 columns
- Server specifications and feature matrix  
- AWS instance recommendations
- RDS compatibility status

### 2. JSON File (`RDSdiscovery_[timestamp].json`)
- Complete assessment data with metadata
- **Pricing summary** with total monthly costs
- Performance metrics and batch processing details
- AWS recommendation explanations

### 3. Log File (`RDSdiscovery_[timestamp].log`)
- Complete success/failure documentation
- Connection attempts and errors
- Feature detection results
- AWS API calls and fallback logic

## **💰 Pricing Integration**

### Cost Estimates Include:
- **Hourly rates** for recommended instances
- **Monthly estimates** (24/7 usage)
- **Currency** (USD)
- **Pricing source** (AWS API or fallback estimates)

### Instance Scaling Explanations:
- **exact_match**: Perfect match for server specifications
- **scaled_up**: Scaled up to meet minimum requirements  
- **closest_fit**: Closest available instance match
- **fallback**: Estimated when AWS API unavailable

**Example Pricing Output:**
```json
{
  "aws_recommendation": {
    "instance_type": "db.m6i.2xlarge",
    "match_type": "scaled_up", 
    "explanation": "Scaled up from 6 CPU/8GB to meet minimum requirements",
    "pricing": {
      "hourly_rate": 0.768,
      "monthly_estimate": 562.18,
      "currency": "USD"
    }
  }
}
```

## **🔍 Feature Detection**

### RDS Blocking Features (Detected)
- Linked Servers
- Log Shipping  
- FILESTREAM
- Resource Governor
- Transaction Replication
- Extended Procedures
- TSQL Endpoints
- PolyBase
- File Tables
- Buffer Pool Extension
- Stretch Database
- Trustworthy Databases
- Server Triggers
- Machine Learning Services
- Data Quality Services
- Policy Based Management
- CLR Enabled
- Online Indexes

### RDS Compatible Features (Not Blocking)
- **Always On Availability Groups** ✅
- **Always On Failover Cluster Instances** ✅
- **Service Broker** ✅
- **SQL Server Integration Services (SSIS)** ✅
- **SQL Server Reporting Services (SSRS)** ✅

## **☁️ AWS Instance Types**

### General Purpose
- db.m6i.large through db.m6i.24xlarge
- db.m5, db.m4 families

### Memory Optimized  
- db.r6i.large through db.r6i.16xlarge
- db.r5, db.r4 families

### High Memory
- db.x2iedn.large through db.x2iedn.24xlarge
- db.x1e family

## **⚙️ Configuration**

### Authentication Types
- **Windows Authentication**: Uses current Windows credentials
- **SQL Server Authentication**: Requires username/password

### Timeout Settings
- Default: 30 seconds
- Configurable: 5-300 seconds
- Handles connection timeouts gracefully

### AWS Integration
- **Real-time pricing** via AWS Pricing API (when credentials available)
- **Fallback pricing** with estimated costs
- **Regional pricing** support (defaults to us-east-1)

## **📊 Return Value**

```json
{
  "status": "success",
  "outputs": {
    "csv_file": "RDSdiscovery_1234567890.csv",
    "json_file": "RDSdiscovery_1234567890.json", 
    "log_file": "RDSdiscovery_1234567890.log"
  },
  "summary": {
    "servers_assessed": 5,
    "successful_assessments": 4,
    "rds_compatible": 3,
    "success_rate": 80.0
  }
}
```

## **🛠️ Requirements**

- Python 3.8+
- pyodbc (SQL Server connectivity)
- boto3 (AWS integration)
- ODBC Driver 18 for SQL Server

## **🔧 Error Handling**

Robust error handling for:
- Connection failures and authentication errors
- Network timeouts and invalid server names
- Permission issues and SQL query failures
- AWS API failures with fallback logic
- File I/O errors and CSV parsing issues

## **📈 Performance**

- **Batch processing** multiple servers
- **Concurrent assessments** with timeout management
- **Progress tracking** and performance metrics
- **Memory efficient** processing of large server lists

## **🎯 Production Ready**

- Enterprise-grade logging and monitoring
- Comprehensive error handling and recovery
- Performance optimization and resource management
- Complete documentation and audit trails

### **Installation**
```bash
# Clone repository
git clone https://github.com/your-org/strands-rds-discovery
cd strands-rds-discovery

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Basic Usage**
```python
from src.rds_discovery import strands_rds_discovery

# Create server list CSV
strands_rds_discovery(action="template", output_file="servers.csv")

# Edit servers.csv with your SQL Server names
# server_name
# server1.domain.com
# server2.domain.com

# Run assessment - generates PowerShell-compatible CSV
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows"  # or "sql" with username/password
)

# Generates: RdsDiscovery_[timestamp].csv
```

### **Strands AI Conversation Examples**
```
"Assess SQL Server 3.81.26.46 using SQL authentication with user test"

"Generate RDS discovery report for prod-sql01.company.com"

"What AWS instance size is recommended for my 8-core SQL Server?"
```

## **📊 Output Format**

### **PowerShell-Compatible CSV**
Generates `RdsDiscovery_[timestamp].csv` with **identical format** to original PowerShell tool:

```csv
"Server Name","SQL Server Current Edition","CPU","Memory","Instance Type","RDS Compatible","Total DB Size in GB","Total Storage(GB)","SSIS","SSRS"
"3.81.26.46","Enterprise Edition: Core-based Licensing (64-bit)","8","124","db.m6i.2xlarge ","Y","0.80","51.32","N","N"
```

### **Real Data Collection**
- **Server Info**: Real SQL Server edition, version, clustering status
- **Resources**: Actual CPU cores and memory from `sys.dm_os_sys_info`
- **Database Size**: Real user database sizes from `sys.master_files`
- **Total Storage**: PowerShell `xp_fixeddrives` logic for drive capacity
- **Features**: 25+ compatibility checks from live SQL queries
- **AWS Sizing**: Instance recommendations based on actual server specs

## **🔧 System Requirements**

### **Prerequisites**
- **Python 3.8+**
- **Microsoft ODBC Driver 18 for SQL Server**
- **Network access to SQL Servers** (port 1433)
- **Strands Framework** (strands-agents, strands-agents-tools)

### **SQL Server Requirements**
- **xp_fixeddrives enabled** (for accurate storage calculation)
- **Appropriate permissions** for assessment queries
- **SQL Server 2008+** (all versions supported)

## **📋 Assessment Coverage**

### **Real SQL Server Data Collection**
- **Server Information**: Edition, version, clustering from `SERVERPROPERTY()`
- **CPU & Memory**: Real values from `sys.dm_os_sys_info` and `sys.configurations`
- **Database Sizes**: User databases only (`WHERE database_id > 4`)
- **Total Storage**: PowerShell-compatible `xp_fixeddrives` + SQL file calculation
- **27+ Feature Checks**: All compatibility queries from original PowerShell tool plus SSIS/SSRS

### **Enhanced Feature Detection**
- **SSIS Detection**: Checks for SSISDB catalog and custom packages (excludes system collector packages)
- **SSRS Detection**: Checks for ReportServer databases  
- **PowerShell RDS Blocking**: Uses exact same blocking logic as original PowerShell script
- **Always On AG**: Status only (not a blocker - supported in RDS)
- **Service Broker**: Status only (not a blocker - supported in RDS)

### **PowerShell Storage Logic**
```sql
-- Step 1: Get drive free space
EXEC xp_fixeddrives

-- Step 2: Get SQL file sizes per drive
SELECT LEFT(physical_name, 1) as drive,
       SUM(CAST(size AS BIGINT) * 8.0 / 1024.0 / 1024.0) as SQLFilesGB
FROM sys.master_files
GROUP BY LEFT(physical_name, 1)

-- Step 3: Total = Free Space + SQL Files (for drives with SQL files)
```

### **AWS Instance Sizing**
- **CPU-based sizing**: Matches core count to RDS instance types
- **Memory optimization**: Selects appropriate instance families
- **Modern instances**: Recommends latest generation (m6i, r6i, x2iedn)

## **🎯 Current Status**

### **✅ Production Complete**
- **PowerShell CSV Output**: Identical format to original RDS Discovery tool
- **Real SQL Server Data**: All data from live SQL queries, no mock data
- **PowerShell Storage Logic**: Exact `xp_fixeddrives` implementation
- **AWS Instance Sizing**: Intelligent recommendations based on real server specs
- **27+ Compatibility Checks**: Complete feature parity plus SSIS/SSRS detection
- **PowerShell RDS Blocking**: Exact same blocking logic as original PowerShell script
- **Enhanced Detection**: SSIS/SSRS detection with system package filtering
- **Error Handling**: Graceful failure handling matching PowerShell behavior
- **Authentication Support**: Windows and SQL Server authentication
- **Performance Monitoring**: Timing metrics and success rate tracking

### **✅ Verified Results**
- **Real Server Testing**: Tested with SQL Server 2022 Enterprise Edition
- **Data Accuracy**: All values match or closely approximate PowerShell output
- **Storage Calculation**: `51.32 GB` vs PowerShell `53.55 GB` (within expected variance)
- **Feature Detection**: All 27+ compatibility checks working correctly
- **SSIS Detection**: Accurate detection excluding system collector packages
- **SSRS Detection**: Proper ReportServer database detection
- **RDS Blocking Logic**: Matches PowerShell script exactly (Always On AG not a blocker)

## **📄 Documentation**

- 🔧 **[Technical Requirements](TECHNICAL_REQUIREMENTS.md)** - Installation and dependencies
- 📖 **[Usage Guide](USAGE_GUIDE.md)** - Complete tool reference
- 🧪 **[Testing Guide](TESTING_GUIDE.md)** - Testing procedures
- 🚀 **[Production Deployment](PRODUCTION_DEPLOYMENT.md)** - Production setup
- 💡 **[AWS Instance Sizing](AWS_INSTANCE_SIZING.md)** - Sizing logic and algorithms
- 📋 **[Development Plan](strands-rds-discovery-tool-1month-plan.md)** - Project timeline

## **🧪 Testing**

### **Quick Test**
```bash
cd strands-rds-discovery
source venv/bin/activate

# Test with real SQL Server
python3 -c "
from src.rds_discovery import strands_rds_discovery
result = strands_rds_discovery(
    action='assess',
    input_file='real_servers.csv',
    auth_type='sql',
    username='test',
    password='Password1!'
)
print(result)
"
# Generates: RdsDiscovery_[timestamp].csv
```

### **Expected Output**
```
✅ Assessment complete! Report saved to: RdsDiscovery_1759694195.csv

Servers assessed: 1
Successful: 1
RDS Compatible: 1
Success Rate: 100.0%
```

## **🎯 Migration Scenarios**

### **RDS Compatible Servers**
- **CSV Output**: `"RDS Compatible","Y"`
- **Recommendation**: Direct migration to Amazon RDS for SQL Server
- **Instance**: Specific sizing (e.g., `db.m6i.2xlarge`)

### **RDS Custom Candidates**  
- **CSV Output**: `"RDS Custom Compatible","Y"`
- **Recommendation**: Amazon RDS Custom for SQL Server
- **Use Case**: Some enterprise features or custom configurations

### **EC2 Migration Required**
- **CSV Output**: `"EC2 Compatible","Y"`
- **Recommendation**: Amazon EC2 with SQL Server
- **Use Case**: Complex features like Always On AG, FileStream

## **🔒 Security & Compliance**

### **Security Features**
- **Credential Protection**: Passwords never logged or stored
- **Network Security**: SSL/TLS encryption support
- **Input Validation**: Comprehensive parameter validation
- **Error Handling**: Secure error messages without sensitive data

### **Data Collection**
- **No Customer Data**: Only metadata and configuration information
- **Real-time Assessment**: No data stored locally beyond CSV output
- **Audit Trail**: Complete logging of assessment activities

## **🚀 GitHub Setup & Deployment**

### **Initial Repository Setup**
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Strands RDS Discovery Tool v2.1.2"

# Add GitHub remote
git remote add origin https://github.com/bobtherdsman/RDSMCP.git
git branch -M main
```

### **GitHub Personal Access Token Setup**
1. Go to GitHub.com → Profile Picture → Settings
2. Scroll to bottom of left sidebar → "Developer settings"
3. Click "Personal access tokens" → "Tokens (classic)"
4. Click "Generate new token (classic)"
5. **Configuration**:
   - **Note**: "RDS Discovery Tool"
   - **Expiration**: Choose duration (30-90 days recommended)
   - **Scopes**: Check `repo` (full repository access)
6. **Copy token immediately** - you won't see it again

**Direct link**: https://github.com/settings/tokens

### **Push to GitHub**
```bash
# First push (handles merge conflicts)
git pull origin main --allow-unrelated-histories --no-rebase

# Resolve any conflicts by keeping local files
git checkout --ours .gitignore CONTRIBUTING.md LICENSE README.md pyproject.toml
git add .gitignore CONTRIBUTING.md LICENSE README.md pyproject.toml
git commit -m "Merge remote changes, keeping local RDS discovery tool files"

# Push to GitHub
git push -u origin main
# Username: bobtherdsman
# Password: [paste your personal access token]
```

### **Handling Merge Conflicts**
When pushing to an existing repository with different files:

1. **Pull with merge strategy**:
   ```bash
   git pull origin main --allow-unrelated-histories --no-rebase
   ```

2. **Resolve conflicts** (keep your local versions):
   ```bash
   git checkout --ours [conflicted-files]
   git add [conflicted-files]
   ```

3. **Commit merge**:
   ```bash
   git commit -m "Merge remote changes, keeping local RDS discovery tool files"
   ```

4. **Push successfully**:
   ```bash
   git push -u origin main
   ```

### **Authentication Notes**
- **Username**: Your GitHub username (`bobtherdsman`)
- **Password**: Your Personal Access Token (NOT your GitHub password)
- **Token Security**: Store token securely, never commit to code
- **Token Expiration**: Set appropriate expiration and renew as needed

### **Repository Structure**
```
strands-rds-discovery/
├── src/
│   └── rds_discovery.py          # Main tool
├── requirements.txt              # Dependencies
├── README.md                     # This documentation
├── RUN_GUIDE.md                 # Usage guide
├── real_servers.csv             # Server input template
└── RdsDiscovery_[timestamp].csv # Output files
```

## **🤝 Contributing**

### **Strands Integration**
This tool is designed for integration into the mainstream Strands tools ecosystem. The PowerShell-compatible output ensures seamless migration from existing PowerShell-based workflows.

### **Development**
```bash
# Setup development environment
git clone https://github.com/bobtherdsman/RDSMCP.git
cd strands-rds-discovery
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test with real server
python3 -c "from src.rds_discovery import strands_rds_discovery; ..."
```

### **Making Changes**
```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push origin main
# Use your personal access token when prompted
```

## **📞 Support**

### **Key Files**
- **src/rds_discovery.py** - Main Strands tool with PowerShell CSV output
- **RdsDiscovery_[timestamp].csv** - PowerShell-compatible assessment results
- **real_servers.csv** - Server input template

### **Community**
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Strands Community**: Integration with main Strands community channels

## **📜 License**

MIT License - see LICENSE file for details.

---

**Production-ready Strands tool with PowerShell-compatible CSV output and real SQL Server data collection!** 🚀
=======
<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents Tools
  </h1>

  <h2>
    A model-driven approach to building AI agents in just a few lines of code.
  </h2>

  <div align="center">
    <a href="https://github.com/strands-agents/tools/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/tools"/></a>
    <a href="https://github.com/strands-agents/tools/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/tools"/></a>
    <a href="https://github.com/strands-agents/tools/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/tools"/></a>
    <a href="https://github.com/strands-agents/tools/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/tools"/></a>
    <a href="https://pypi.org/project/strands-agents-tools/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents-tools"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents-tools"/></a>
  </div>

  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

Strands Agents Tools is a community-driven project that provides a powerful set of tools for your agents to use. It bridges the gap between large language models and practical applications by offering ready-to-use tools for file operations, system execution, API interactions, mathematical operations, and more.

## ✨ Features

- 📁 **File Operations** - Read, write, and edit files with syntax highlighting and intelligent modifications
- 🖥️ **Shell Integration** - Execute and interact with shell commands securely
- 🧠 **Memory** - Store user and agent memories across agent runs to provide personalized experiences with both Mem0 and Amazon Bedrock Knowledge Bases
- 🕸️ **Web Infrastructure** - Perform web searches, extract page content, and crawl websites with Tavily and Exa-powered tools
- 🌐 **HTTP Client** - Make API requests with comprehensive authentication support
- 💬 **Slack Client** - Real-time Slack events, message processing, and Slack API access
- 🐍 **Python Execution** - Run Python code snippets with state persistence, user confirmation for code execution, and safety features
- 🧮 **Mathematical Tools** - Perform advanced calculations with symbolic math capabilities
- ☁️ **AWS Integration** - Seamless access to AWS services
- 🖼️ **Image Processing** - Generate and process images for AI applications
- 🎥 **Video Processing** - Use models and agents to generate dynamic videos
- 🎙️ **Audio Output** - Enable models to generate audio and speak
- 🔄 **Environment Management** - Handle environment variables safely
- 📝 **Journaling** - Create and manage structured logs and journals
- ⏱️ **Task Scheduling** - Schedule and manage cron jobs
- 🧠 **Advanced Reasoning** - Tools for complex thinking and reasoning capabilities
- 🐝 **Swarm Intelligence** - Coordinate multiple AI agents for parallel problem solving with shared memory
- 🔌 **Dynamic MCP Client** - ⚠️ Dynamically connect to external MCP servers and load remote tools (use with caution - see security warnings)
- 🔄 **Multiple tools in Parallel**  - Call multiple other tools at the same time in parallel with Batch Tool
- 🔍 **Browser Tool** - Tool giving an agent access to perform automated actions on a browser (chromium)
- 📈 **Diagram** - Create AWS cloud diagrams, basic diagrams, or UML diagrams using python libraries
- 📰 **RSS Feed Manager** - Subscribe, fetch, and process RSS feeds with content filtering and persistent storage
- 🖱️ **Computer Tool** - Automate desktop actions including mouse movements, keyboard input, screenshots, and application management

## 📦 Installation

### Quick Install

```bash
pip install strands-agents-tools
```

To install the dependencies for optional tools:

```bash
pip install strands-agents-tools[mem0_memory, use_browser, rss, use_computer]
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/strands-agents/tools.git
cd tools

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Tools Overview

Below is a comprehensive table of all available tools, how to use them with an agent, and typical use cases:

| Tool | Agent Usage | Use Case |
|------|-------------|----------|
| a2a_client | `provider = A2AClientToolProvider(known_agent_urls=["http://localhost:9000"]); agent = Agent(tools=provider.tools)` | Discover and communicate with A2A-compliant agents, send messages between agents |
| file_read | `agent.tool.file_read(path="path/to/file.txt")` | Reading configuration files, parsing code files, loading datasets |
| file_write | `agent.tool.file_write(path="path/to/file.txt", content="file content")` | Writing results to files, creating new files, saving output data |
| editor | `agent.tool.editor(command="view", path="path/to/file.py")` | Advanced file operations like syntax highlighting, pattern replacement, and multi-file edits |
| shell* | `agent.tool.shell(command="ls -la")` | Executing shell commands, interacting with the operating system, running scripts |
| http_request | `agent.tool.http_request(method="GET", url="https://api.example.com/data")` | Making API calls, fetching web data, sending data to external services |
| tavily_search | `agent.tool.tavily_search(query="What is artificial intelligence?", search_depth="advanced")` | Real-time web search optimized for AI agents with a variety of custom parameters |
| tavily_extract | `agent.tool.tavily_extract(urls=["www.tavily.com"], extract_depth="advanced")` | Extract clean, structured content from web pages with advanced processing and noise removal |
| tavily_crawl | `agent.tool.tavily_crawl(url="www.tavily.com", max_depth=2, instructions="Find API docs")` | Crawl websites intelligently starting from a base URL with filtering and extraction |
| tavily_map | `agent.tool.tavily_map(url="www.tavily.com", max_depth=2, instructions="Find all pages")` | Map website structure and discover URLs starting from a base URL without content extraction |
| exa_search | `agent.tool.exa_search(query="Best project management tools", text=True)` | Intelligent web search with auto mode (default) that combines neural and keyword search for optimal results |
| exa_get_contents | `agent.tool.exa_get_contents(urls=["https://example.com/article"], text=True, summary={"query": "key points"})` | Extract full content and summaries from specific URLs with live crawling fallback |
| python_repl* | `agent.tool.python_repl(code="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())")` | Running Python code snippets, data analysis, executing complex logic with user confirmation for security |
| calculator | `agent.tool.calculator(expression="2 * sin(pi/4) + log(e**2)")` | Performing mathematical operations, symbolic math, equation solving |
| code_interpreter | `code_interpreter = AgentCoreCodeInterpreter(region="us-west-2"); agent = Agent(tools=[code_interpreter.code_interpreter])` | Execute code in isolated sandbox environments with multi-language support (Python, JavaScript, TypeScript), persistent sessions, and file operations |
| use_aws | `agent.tool.use_aws(service_name="s3", operation_name="list_buckets", parameters={}, region="us-west-2")` | Interacting with AWS services, cloud resource management |
| retrieve | `agent.tool.retrieve(text="What is STRANDS?")` | Retrieving information from Amazon Bedrock Knowledge Bases |
| nova_reels | `agent.tool.nova_reels(action="create", text="A cinematic shot of mountains", s3_bucket="my-bucket")` | Create high-quality videos using Amazon Bedrock Nova Reel with configurable parameters via environment variables |
| agent_core_memory | `agent.tool.agent_core_memory(action="record", content="Hello, I like vegetarian food")` | Store and retrieve memories with Amazon Bedrock Agent Core Memory service |
| mem0_memory | `agent.tool.mem0_memory(action="store", content="Remember I like to play tennis", user_id="alex")` | Store user and agent memories across agent runs to provide personalized experience |
| bright_data | `agent.tool.bright_data(action="scrape_as_markdown", url="https://example.com")` | Web scraping, search queries, screenshot capture, and structured data extraction from websites and different data feeds|
| memory | `agent.tool.memory(action="retrieve", query="product features")` | Store, retrieve, list, and manage documents in Amazon Bedrock Knowledge Bases with configurable parameters via environment variables |
| environment | `agent.tool.environment(action="list", prefix="AWS_")` | Managing environment variables, configuration management |
| generate_image_stability | `agent.tool.generate_image_stability(prompt="A tranquil pool")` | Creating images using Stability AI models |
| generate_image | `agent.tool.generate_image(prompt="A sunset over mountains")` | Creating AI-generated images for various applications |
| image_reader | `agent.tool.image_reader(image_path="path/to/image.jpg")` | Processing and reading image files for AI analysis |
| journal | `agent.tool.journal(action="write", content="Today's progress notes")` | Creating structured logs, maintaining documentation |
| think | `agent.tool.think(thought="Complex problem to analyze", cycle_count=3)` | Advanced reasoning, multi-step thinking processes |
| load_tool | `agent.tool.load_tool(path="path/to/custom_tool.py", name="custom_tool")` | Dynamically loading custom tools and extensions |
| swarm | `agent.tool.swarm(task="Analyze this problem", swarm_size=3, coordination_pattern="collaborative")` | Coordinating multiple AI agents to solve complex problems through collective intelligence |
| current_time | `agent.tool.current_time(timezone="US/Pacific")` | Get the current time in ISO 8601 format for a specified timezone |
| sleep | `agent.tool.sleep(seconds=5)` | Pause execution for the specified number of seconds, interruptible with SIGINT (Ctrl+C) |
| agent_graph | `agent.tool.agent_graph(agents=["agent1", "agent2"], connections=[{"from": "agent1", "to": "agent2"}])` | Create and visualize agent relationship graphs for complex multi-agent systems |
| cron* | `agent.tool.cron(action="schedule", name="task", schedule="0 * * * *", command="backup.sh")` | Schedule and manage recurring tasks with cron job syntax <br> **Does not work on Windows |
| slack | `agent.tool.slack(action="post_message", channel="general", text="Hello team!")` | Interact with Slack workspace for messaging and monitoring |
| speak | `agent.tool.speak(text="Operation completed successfully", style="green", mode="polly")` | Output status messages with rich formatting and optional text-to-speech |
| stop | `agent.tool.stop(message="Process terminated by user request")` | Gracefully terminate agent execution with custom message |
| handoff_to_user | `agent.tool.handoff_to_user(message="Please confirm action", breakout_of_loop=False)` | Hand off control to user for confirmation, input, or complete task handoff |
| use_llm | `agent.tool.use_llm(prompt="Analyze this data", system_prompt="You are a data analyst")` | Create nested AI loops with customized system prompts for specialized tasks |
| workflow | `agent.tool.workflow(action="create", name="data_pipeline", steps=[{"tool": "file_read"}, {"tool": "python_repl"}])` | Define, execute, and manage multi-step automated workflows |
| mcp_client | `agent.tool.mcp_client(action="connect", connection_id="my_server", transport="stdio", command="python", args=["server.py"])` | ⚠️ **SECURITY WARNING**: Dynamically connect to external MCP servers via stdio, sse, or streamable_http, list tools, and call remote tools. This can pose security risks as agents may connect to malicious servers. Use with caution in production. |
| batch| `agent.tool.batch(invocations=[{"name": "current_time", "arguments": {"timezone": "Europe/London"}}, {"name": "stop", "arguments": {}}])` | Call multiple other tools in parallel. |
| browser | `browser = LocalChromiumBrowser(); agent = Agent(tools=[browser.browser])` | Web scraping, automated testing, form filling, web automation tasks |
| diagram | `agent.tool.diagram(diagram_type="cloud", nodes=[{"id": "s3", "type": "S3"}], edges=[])` | Create AWS cloud architecture diagrams, network diagrams, graphs, and UML diagrams (all 14 types) |
| rss | `agent.tool.rss(action="subscribe", url="https://example.com/feed.xml", feed_id="tech_news")` | Manage RSS feeds: subscribe, fetch, read, search, and update content from various sources |
| use_computer | `agent.tool.use_computer(action="click", x=100, y=200, app_name="Chrome") ` | Desktop automation, GUI interaction, screen capture |
| search_video | `agent.tool.search_video(query="people discussing AI")` | Semantic video search using TwelveLabs' Marengo model |
| chat_video | `agent.tool.chat_video(prompt="What are the main topics?", video_id="video_123")` | Interactive video analysis using TwelveLabs' Pegasus model |

\* *These tools do not work on windows*

## 💻 Usage Examples

### File Operations

```python
from strands import Agent
from strands_tools import file_read, file_write, editor

agent = Agent(tools=[file_read, file_write, editor])

agent.tool.file_read(path="config.json")
agent.tool.file_write(path="output.txt", content="Hello, world!")
agent.tool.editor(command="view", path="script.py")
```

### Dynamic MCP Client Integration

⚠️ **SECURITY WARNING**: The Dynamic MCP Client allows agents to autonomously connect to external MCP servers and load remote tools at runtime. This poses significant security risks as agents can potentially connect to malicious servers and execute untrusted code. Use with extreme caution in production environments.

This tool is different from the static MCP server implementation in the Strands SDK (see [MCP Tools Documentation](https://github.com/strands-agents/docs/blob/main/docs/user-guide/concepts/tools/mcp-tools.md)) which uses pre-configured, trusted MCP servers.

```python
from strands import Agent
from strands_tools import mcp_client

agent = Agent(tools=[mcp_client])

# Connect to a custom MCP server via stdio
agent.tool.mcp_client(
    action="connect",
    connection_id="my_tools",
    transport="stdio",
    command="python",
    args=["my_mcp_server.py"]
)

# List available tools on the server
tools = agent.tool.mcp_client(
    action="list_tools",
    connection_id="my_tools"
)

# Call a tool from the MCP server
result = agent.tool.mcp_client(
    action="call_tool",
    connection_id="my_tools",
    tool_name="calculate",
    tool_args={"x": 10, "y": 20}
)

# Connect to a SSE-based server
agent.tool.mcp_client(
    action="connect",
    connection_id="web_server",
    transport="sse",
    server_url="http://localhost:8080/sse"
)

# Connect to a streamable HTTP server
agent.tool.mcp_client(
    action="connect",
    connection_id="http_server",
    transport="streamable_http",
    server_url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer token"},
    timeout=60
)

# Load MCP tools into agent's registry for direct access
# ⚠️ WARNING: This loads external tools directly into the agent
agent.tool.mcp_client(
    action="load_tools",
    connection_id="my_tools"
)
# Now you can call MCP tools directly as: agent.tool.calculate(x=10, y=20)
```

### Shell Commands

*Note: `shell` does not work on Windows.*

```python
from strands import Agent
from strands_tools import shell

agent = Agent(tools=[shell])

# Execute a single command
result = agent.tool.shell(command="ls -la")

# Execute a sequence of commands
results = agent.tool.shell(command=["mkdir -p test_dir", "cd test_dir", "touch test.txt"])

# Execute commands with error handling
agent.tool.shell(command="risky-command", ignore_errors=True)
```

### HTTP Requests

```python
from strands import Agent
from strands_tools import http_request

agent = Agent(tools=[http_request])

# Make a simple GET request
response = agent.tool.http_request(
    method="GET",
    url="https://api.example.com/data"
)

# POST request with authentication
response = agent.tool.http_request(
    method="POST",
    url="https://api.example.com/resource",
    headers={"Content-Type": "application/json"},
    body=json.dumps({"key": "value"}),
    auth_type="Bearer",
    auth_token="your_token_here"
)

# Convert HTML webpages to markdown for better readability
response = agent.tool.http_request(
    method="GET",
    url="https://example.com/article",
    convert_to_markdown=True
)
```

### Tavily Search, Extract, Crawl, and Map

```python
from strands import Agent
from strands_tools.tavily import (
    tavily_search, tavily_extract, tavily_crawl, tavily_map
)

# For async usage, call the corresponding *_async function with await.
# Synchronous usage 
agent = Agent(tools=[tavily_search, tavily_extract, tavily_crawl, tavily_map])

# Real-time web search
result = agent.tool.tavily_search(
    query="Latest developments in renewable energy",
    search_depth="advanced",
    topic="news",
    max_results=10,
    include_raw_content=True
)

# Extract content from multiple URLs
result = agent.tool.tavily_extract(
    urls=["www.tavily.com", "www.apple.com"],
    extract_depth="advanced",
    format="markdown"
)

# Advanced crawl with instructions and filtering
result = agent.tool.tavily_crawl(
    url="www.tavily.com",
    max_depth=2,
    limit=50,
    instructions="Find all API documentation and developer guides",
    extract_depth="advanced",
    include_images=True
)

# Basic website mapping
result = agent.tool.tavily_map(url="www.tavily.com")

```

### Exa Search and Contents

```python
from strands import Agent
from strands_tools.exa import exa_search, exa_get_contents

agent = Agent(tools=[exa_search, exa_get_contents])

# Basic search (auto mode is default and recommended)
result = agent.tool.exa_search(
    query="Best project management software",
    text=True
)

# Company-specific search when needed
result = agent.tool.exa_search(
    query="Anthropic AI safety research",
    category="company",
    include_domains=["anthropic.com"],
    num_results=5,
    summary={"query": "key research areas and findings"}
)

# News search with date filtering
result = agent.tool.exa_search(
    query="AI regulation policy updates",
    category="news",
    start_published_date="2024-01-01T00:00:00.000Z",
    text=True
)

# Get detailed content from specific URLs
result = agent.tool.exa_get_contents(
    urls=[
        "https://example.com/blog-post",
        "https://github.com/microsoft/semantic-kernel"
    ],
    text={"maxCharacters": 5000, "includeHtmlTags": False},
    summary={
        "query": "main points and practical applications"
    },
    subpages=2,
    extras={"links": 5, "imageLinks": 2}
)

# Structured summary with JSON schema
result = agent.tool.exa_get_contents(
    urls=["https://example.com/article"],
    summary={
        "query": "main findings and recommendations",
        "schema": {
            "type": "object",
            "properties": {
                "main_points": {"type": "string", "description": "Key points from the article"},
                "recommendations": {"type": "string", "description": "Suggested actions or advice"},
                "conclusion": {"type": "string", "description": "Overall conclusion"},
                "relevance": {"type": "string", "description": "Why this matters"}
            },
            "required": ["main_points", "conclusion"]
        }
    }
)

```

### Python Code Execution

*Note: `python_repl` does not work on Windows.*

```python
from strands import Agent
from strands_tools import python_repl

agent = Agent(tools=[python_repl])

# Execute Python code with state persistence
result = agent.tool.python_repl(code="""
import pandas as pd

# Load and process data
data = pd.read_csv('data.csv')
processed = data.groupby('category').mean()

processed.head()
""")
```

### Code Interpreter

```python
from strands import Agent
from strands_tools.code_interpreter import AgentCoreCodeInterpreter

# Create the code interpreter tool
bedrock_agent_core_code_interpreter = AgentCoreCodeInterpreter(region="us-west-2")
agent = Agent(tools=[bedrock_agent_core_code_interpreter.code_interpreter])

# Create a session
agent.tool.code_interpreter({
    "action": {
        "type": "initSession",
        "description": "Data analysis session",
        "session_name": "analysis-session"
    }
})

# Execute Python code
agent.tool.code_interpreter({
    "action": {
        "type": "executeCode",
        "session_name": "analysis-session",
        "code": "print('Hello from sandbox!')",
        "language": "python"
    }
})
```

### Swarm Intelligence

```python
from strands import Agent
from strands_tools import swarm

agent = Agent(tools=[swarm])

# Create a collaborative swarm of agents to tackle a complex problem
result = agent.tool.swarm(
    task="Generate creative solutions for reducing plastic waste in urban areas",
    swarm_size=5,
    coordination_pattern="collaborative"
)

# Create a competitive swarm for diverse solution generation
result = agent.tool.swarm(
    task="Design an innovative product for smart home automation",
    swarm_size=3,
    coordination_pattern="competitive"
)

# Hybrid approach combining collaboration and competition
result = agent.tool.swarm(
    task="Develop marketing strategies for a new sustainable fashion brand",
    swarm_size=4,
    coordination_pattern="hybrid"
)
```

### Use AWS

```python
from strands import Agent
from strands_tools import use_aws

agent = Agent(tools=[use_aws])

# List S3 buckets
result = agent.tool.use_aws(
    service_name="s3",
    operation_name="list_buckets",
    parameters={},
    region="us-east-1",
    label="List all S3 buckets"
)

# Get the contents of a specific S3 bucket
result = agent.tool.use_aws(
    service_name="s3",
    operation_name="list_objects_v2",
    parameters={"Bucket": "example-bucket"},  # Replace with your actual bucket name
    region="us-east-1",
    label="List objects in a specific S3 bucket"
)

# Get the list of EC2 subnets
result = agent.tool.use_aws(
    service_name="ec2",
    operation_name="describe_subnets",
    parameters={},
    region="us-east-1",
    label="List all subnets"
)
```

### Batch Tool

```python
import os
import sys

from strands import Agent
from strands_tools import batch, http_request, use_aws

# Example usage of the batch with http_request and use_aws tools
agent = Agent(tools=[batch, http_request, use_aws])

result = agent.tool.batch(
    invocations=[
        {"name": "http_request", "arguments": {"method": "GET", "url": "https://api.ipify.org?format=json"}},
        {
            "name": "use_aws",
            "arguments": {
                "service_name": "s3",
                "operation_name": "list_buckets",
                "parameters": {},
                "region": "us-east-1",
                "label": "List S3 Buckets"
            }
        },
    ]
)
```

### Video Tools

```python
from strands import Agent
from strands_tools import search_video, chat_video

agent = Agent(tools=[search_video, chat_video])

# Search for video content using natural language
result = agent.tool.search_video(
    query="people discussing AI technology",
    threshold="high",
    group_by="video",
    page_limit=5
)

# Chat with existing video (no index_id needed)
result = agent.tool.chat_video(
    prompt="What are the main topics discussed in this video?",
    video_id="existing-video-id"
)

# Chat with new video file (index_id required for upload)
result = agent.tool.chat_video(
    prompt="Describe what happens in this video",
    video_path="/path/to/video.mp4",
    index_id="your-index-id"  # or set TWELVELABS_PEGASUS_INDEX_ID env var
)
```

### AgentCore Memory
```python
from strands import Agent
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider


provider = AgentCoreMemoryToolProvider(
    memory_id="memory-123abc",  # Required
    actor_id="user-456",        # Required
    session_id="session-789",   # Required
    namespace="default",        # Required
    region="us-west-2"          # Optional, defaults to us-west-2
)

agent = Agent(tools=provider.tools)

# Create a new memory
result = agent.tool.agent_core_memory(
    action="record",
    content="I am allergic to shellfish"
)

# Search for relevant memories
result = agent.tool.agent_core_memory(
    action="retrieve",
    query="user preferences"
)

# List all memories
result = agent.tool.agent_core_memory(
    action="list"
)

# Get a specific memory by ID
result = agent.tool.agent_core_memory(
    action="get",
    memory_record_id="mr-12345"
)
```

### Browser
```python
from strands import Agent
from strands_tools.browser import LocalChromiumBrowser

# Create browser tool
browser = LocalChromiumBrowser()
agent = Agent(tools=[browser.browser])

# Simple navigation
result = agent.tool.browser({
    "action": {
        "type": "navigate",
        "url": "https://example.com"
    }
})

# Initialize a session first
result = agent.tool.browser({
    "action": {
        "type": "initSession",
        "session_name": "main-session",
        "description": "Web automation session"
    }
})
```

### Handoff to User

```python
from strands import Agent
from strands_tools import handoff_to_user

agent = Agent(tools=[handoff_to_user])

# Request user confirmation and continue
response = agent.tool.handoff_to_user(
    message="I need your approval to proceed with deleting these files. Type 'yes' to confirm.",
    breakout_of_loop=False
)

# Complete handoff to user (stops agent execution)
agent.tool.handoff_to_user(
    message="Task completed. Please review the results and take any necessary follow-up actions.",
    breakout_of_loop=True
)
```

### A2A Client

```python
from strands import Agent
from strands_tools.a2a_client import A2AClientToolProvider

# Initialize the A2A client provider with known agent URLs
provider = A2AClientToolProvider(known_agent_urls=["http://localhost:9000"])
agent = Agent(tools=provider.tools)

# Use natural language to interact with A2A agents
response = agent("discover available agents and send a greeting message")

# The agent will automatically use the available tools:
# - discover_agent(url) to find agents
# - list_discovered_agents() to see all discovered agents
# - send_message(message_text, target_agent_url) to communicate
```

### Diagram

```python
from strands import Agent
from strands_tools import diagram

agent = Agent(tools=[diagram])

# Create an AWS cloud architecture diagram
result = agent.tool.diagram(
    diagram_type="cloud",
    nodes=[
        {"id": "users", "type": "Users", "label": "End Users"},
        {"id": "cloudfront", "type": "CloudFront", "label": "CDN"},
        {"id": "s3", "type": "S3", "label": "Static Assets"},
        {"id": "api", "type": "APIGateway", "label": "API Gateway"},
        {"id": "lambda", "type": "Lambda", "label": "Backend Service"}
    ],
    edges=[
        {"from": "users", "to": "cloudfront"},
        {"from": "cloudfront", "to": "s3"},
        {"from": "users", "to": "api"},
        {"from": "api", "to": "lambda"}
    ],
    title="Web Application Architecture"
)

# Create a UML class diagram
result = agent.tool.diagram(
    diagram_type="class",
    elements=[
        {
            "name": "User",
            "attributes": ["+id: int", "-name: string", "#email: string"],
            "methods": ["+login(): bool", "+logout(): void"]
        },
        {
            "name": "Order",
            "attributes": ["+id: int", "-items: List", "-total: float"],
            "methods": ["+addItem(item): void", "+calculateTotal(): float"]
        }
    ],
    relationships=[
        {"from": "User", "to": "Order", "type": "association", "multiplicity": "1..*"}
    ],
    title="E-commerce Domain Model"
)
```

### RSS Feed Management

```python
from strands import Agent
from strands_tools import rss

agent = Agent(tools=[rss])

# Subscribe to a feed
result = agent.tool.rss(
    action="subscribe",
    url="https://news.example.com/rss/technology"
)

# List all subscribed feeds
feeds = agent.tool.rss(action="list")

# Read entries from a specific feed
entries = agent.tool.rss(
    action="read",
    feed_id="news_example_com_technology",
    max_entries=5,
    include_content=True
)

# Search across all feeds
search_results = agent.tool.rss(
    action="search",
    query="machine learning",
    max_entries=10
)

# Fetch feed content without subscribing
latest_news = agent.tool.rss(
    action="fetch",
    url="https://blog.example.org/feed",
    max_entries=3
)
```

### Use Computer

```python
from strands import Agent
from strands_tools import use_computer

agent = Agent(tools=[use_computer])

# Find mouse position
result = agent.tool.use_computer(action="mouse_position")

# Automate adding text
result = agent.tool.use_computer(action="type", text="Hello, world!", app_name="Notepad")

# Analyze current computer screen
result = agent.tool.use_computer(action="analyze_screen")

result = agent.tool.use_computer(action="open_app", app_name="Calculator")
result = agent.tool.use_computer(action="close_app", app_name="Calendar")

result = agent.tool.use_computer(
    action="hotkey",
    hotkey_str="command+ctrl+f",  # For macOS
    app_name="Chrome"
)
```

## 🌍 Environment Variables Configuration

Agents Tools provides extensive customization through environment variables. This allows you to configure tool behavior without modifying code, making it ideal for different environments (development, testing, production).

### Global Environment Variables

These variables affect multiple tools:

| Environment Variable | Description | Default | Affected Tools |
|----------------------|-------------|---------|---------------|
| BYPASS_TOOL_CONSENT | Bypass consent for tool invocation, set to "true" to enable | false | All tools that require consent (e.g. shell, file_write, python_repl) |
| STRANDS_TOOL_CONSOLE_MODE | Enable rich UI for tools, set to "enabled" to enable | disabled | All tools that have optional rich UI |
| AWS_REGION | Default AWS region for AWS operations | us-west-2 | use_aws, retrieve, generate_image, memory, nova_reels |
| AWS_PROFILE | AWS profile name to use from ~/.aws/credentials | default | use_aws, retrieve |
| LOG_LEVEL | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO | All tools |

### Tool-Specific Environment Variables

#### Calculator Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| CALCULATOR_MODE | Default calculation mode | evaluate |
| CALCULATOR_PRECISION | Number of decimal places for results | 10 |
| CALCULATOR_SCIENTIFIC | Whether to use scientific notation for numbers | False |
| CALCULATOR_FORCE_NUMERIC | Force numeric evaluation of symbolic expressions | False |
| CALCULATOR_FORCE_SCIENTIFIC_THRESHOLD | Threshold for automatic scientific notation | 1e21 |
| CALCULATOR_DERIVE_ORDER | Default order for derivatives | 1 |
| CALCULATOR_SERIES_POINT | Default point for series expansion | 0 |
| CALCULATOR_SERIES_ORDER | Default order for series expansion | 5 |

#### Current Time Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| DEFAULT_TIMEZONE | Default timezone for current_time tool | UTC |

#### Sleep Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| MAX_SLEEP_SECONDS | Maximum allowed sleep duration in seconds | 300 |

#### Tavily Search, Extract, Crawl, and Map Tools

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| TAVILY_API_KEY | Tavily API key (required for all Tavily functionality) | None |
- Visit https://www.tavily.com/ to create a free account and API key.

#### Exa Search and Contents Tools

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| EXA_API_KEY | Exa API key (required for all Exa functionality) | None |
- Visit https://dashboard.exa.ai/api-keys to create a free account and API key.

#### Mem0 Memory Tool

The Mem0 Memory Tool supports three different backend configurations:

1. **Mem0 Platform**:
   - Uses the Mem0 Platform API for memory management
   - Requires a Mem0 API key

2. **OpenSearch** (Recommended for AWS environments):
   - Uses OpenSearch as the vector store backend
   - Requires AWS credentials and OpenSearch configuration

3. **FAISS** (Default for local development):
   - Uses FAISS as the local vector store backend
   - Requires faiss-cpu package for local vector storage

4. **Neptune Analytics** (Optional Graph backend for search enhancement):
   - Uses Neptune Analytics as the graph store backend to enhance memory recall.
   - Requires AWS credentials and Neptune Analytics configuration
   ```
   # Configure your Neptune Analytics graph ID in the .env file:
   export NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER=sample-graph-id
   
   # Configure your Neptune Analytics graph ID in Python code:
   import os
   os.environ['NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER'] = "g-sample-graph-id"
   
   ```

| Environment Variable | Description | Default | Required For |
|----------------------|-------------|---------|--------------|
| MEM0_API_KEY | Mem0 Platform API key | None | Mem0 Platform |
| OPENSEARCH_HOST | OpenSearch Host URL | None | OpenSearch |
| AWS_REGION | AWS Region for OpenSearch | us-west-2 | OpenSearch |
| NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER | Neptune Analytics Graph Identifier | None | Neptune Analytics |
| DEV | Enable development mode (bypasses confirmations) | false | All modes |
| MEM0_LLM_PROVIDER | LLM provider for memory processing | aws_bedrock | All modes |
| MEM0_LLM_MODEL | LLM model for memory processing | anthropic.claude-3-5-haiku-20241022-v1:0 | All modes |
| MEM0_LLM_TEMPERATURE | LLM temperature (0.0-2.0) | 0.1 | All modes |
| MEM0_LLM_MAX_TOKENS | LLM maximum tokens | 2000 | All modes |
| MEM0_EMBEDDER_PROVIDER | Embedder provider for vector embeddings | aws_bedrock | All modes |
| MEM0_EMBEDDER_MODEL | Embedder model for vector embeddings | amazon.titan-embed-text-v2:0 | All modes |


**Note**:
- If `MEM0_API_KEY` is set, the tool will use the Mem0 Platform
- If `OPENSEARCH_HOST` is set, the tool will use OpenSearch
- If neither is set, the tool will default to FAISS (requires `faiss-cpu` package)
- If `NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER` is set, the tool will configure Neptune Analytics as graph store to enhance memory search
- LLM configuration applies to all backend modes and allows customization of the language model used for memory processing

#### Bright Data Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| BRIGHTDATA_API_KEY | Bright Data API Key | None |
| BRIGHTDATA_ZONE | Bright Data Web Unlocker Zone | web_unlocker1 |

#### Memory Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| MEMORY_DEFAULT_MAX_RESULTS | Default maximum results for list operations | 50 |
| MEMORY_DEFAULT_MIN_SCORE | Default minimum relevance score for filtering results | 0.4 |

#### Nova Reels Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| NOVA_REEL_DEFAULT_SEED | Default seed for video generation | 0 |
| NOVA_REEL_DEFAULT_FPS | Default frames per second for generated videos | 24 |
| NOVA_REEL_DEFAULT_DIMENSION | Default video resolution in WIDTHxHEIGHT format | 1280x720 |
| NOVA_REEL_DEFAULT_MAX_RESULTS | Default maximum number of jobs to return for list action | 10 |

#### Python REPL Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| PYTHON_REPL_BINARY_MAX_LEN | Maximum length for binary content before truncation | 100 |
| PYTHON_REPL_INTERACTIVE | Whether to enable interactive PTY mode | None |
| PYTHON_REPL_RESET_STATE | Whether to reset the REPL state before execution | None |

#### Shell Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| SHELL_DEFAULT_TIMEOUT | Default timeout in seconds for shell commands | 900 |

#### Slack Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| SLACK_DEFAULT_EVENT_COUNT | Default number of events to retrieve | 42 |
| STRANDS_SLACK_AUTO_REPLY | Enable automatic replies to messages | false |
| STRANDS_SLACK_LISTEN_ONLY_TAG | Only process messages containing this tag | None |

#### Speak Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| SPEAK_DEFAULT_STYLE | Default style for status messages | green |
| SPEAK_DEFAULT_MODE | Default speech mode (fast/polly) | fast |
| SPEAK_DEFAULT_VOICE_ID | Default Polly voice ID | Joanna |
| SPEAK_DEFAULT_OUTPUT_PATH | Default audio output path | speech_output.mp3 |
| SPEAK_DEFAULT_PLAY_AUDIO | Whether to play audio by default | True |

#### Editor Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| EDITOR_DIR_TREE_MAX_DEPTH | Maximum depth for directory tree visualization | 2 |
| EDITOR_DEFAULT_STYLE | Default style for output panels | default |
| EDITOR_DEFAULT_LANGUAGE | Default language for syntax highlighting | python |

#### Environment Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| ENV_VARS_MASKED_DEFAULT | Default setting for masking sensitive values | true |

#### Dynamic MCP Client Tool

| Environment Variable | Description | Default | 
|----------------------|-------------|---------|
| STRANDS_MCP_TIMEOUT | Default timeout in seconds for MCP operations | 30.0 |

#### File Read Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| FILE_READ_RECURSIVE_DEFAULT | Default setting for recursive file searching | true |
| FILE_READ_CONTEXT_LINES_DEFAULT | Default number of context lines around search matches | 2 |
| FILE_READ_START_LINE_DEFAULT | Default starting line number for lines mode | 0 |
| FILE_READ_CHUNK_OFFSET_DEFAULT | Default byte offset for chunk mode | 0 |
| FILE_READ_DIFF_TYPE_DEFAULT | Default diff type for file comparisons | unified |
| FILE_READ_USE_GIT_DEFAULT | Default setting for using git in time machine mode | true |
| FILE_READ_NUM_REVISIONS_DEFAULT | Default number of revisions to show in time machine mode | 5 |

#### Browser Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| STRANDS_DEFAULT_WAIT_TIME | Default setting for wait time with actions | 1 |
| STRANDS_BROWSER_MAX_RETRIES | Default number of retries to perform when an action fails | 3 |
| STRANDS_BROWSER_RETRY_DELAY | Default retry delay time for retry mechanisms | 1 |
| STRANDS_BROWSER_SCREENSHOTS_DIR | Default directory where screenshots will be saved | screenshots |
| STRANDS_BROWSER_USER_DATA_DIR | Default directory where data for reloading a browser instance is stored | ~/.browser_automation |
| STRANDS_BROWSER_HEADLESS | Default headless setting for launching browsers | false |
| STRANDS_BROWSER_WIDTH | Default width of the browser | 1280 |
| STRANDS_BROWSER_HEIGHT | Default height of the browser | 800 |

#### RSS Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| STRANDS_RSS_MAX_ENTRIES | Default setting for maximum number of entries per feed | 100 |
| STRANDS_RSS_UPDATE_INTERVAL | Default amount of time between updating rss feeds in minutes | 60 |
| STRANDS_RSS_STORAGE_PATH | Default storage path where rss feeds are stored locally | strands_rss_feeds (this may vary based on your system) |

#### Video Tools

| Environment Variable | Description | Default | 
|----------------------|-------------|---------|
| TWELVELABS_API_KEY | TwelveLabs API key for video analysis | None |
| TWELVELABS_MARENGO_INDEX_ID | Default index ID for search_video tool | None |
| TWELVELABS_PEGASUS_INDEX_ID | Default index ID for chat_video tool | None |


## Contributing ❤️

This is a community-driven project, powered by passionate developers like you.
We enthusiastically welcome contributions from everyone,
regardless of experience level—your unique perspective is valuable to us!

### How to Get Started?

1. **Find your first opportunity**: If you're new to the project, explore our labeled "good first issues" for beginner-friendly tasks.
2. **Understand our workflow**: Review our [Contributing Guide](CONTRIBUTING.md)  to learn about our development setup, coding standards, and pull request process.
3. **Make your impact**: Contributions come in many forms—fixing bugs, enhancing documentation, improving performance, adding features, writing tests, or refining the user experience.
4. **Submit your work**: When you're ready, submit a well-documented pull request, and our maintainers will provide feedback to help get your changes merged.

Your questions, insights, and ideas are always welcome!

Together, we're building something meaningful that impacts real users. We look forward to collaborating with you!

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
>>>>>>> b65dd11eb92e513a76ff4a37ed170aefaa664d41
