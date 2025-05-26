<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-auto.svg" alt="Strands Agents" width="55px" height="105px">
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

Strands Agents Tools provides a powerful set of tools for your agents to use. It bridges the gap between large language models and practical applications by offering ready-to-use tools for file operations, system execution, API interactions, mathematical operations, and more.

## ✨ Features

- 📁 **File Operations** - Read, write, and edit files with syntax highlighting and intelligent modifications
- 🖥️ **Shell Integration** - Execute and interact with shell commands securely
- 🧠 **Memory** - Store user and agent memories across agent runs to provide personalized experiences with both Mem0 and Amazon Bedrock Knowledge Bases
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

## 📦 Installation

### Quick Install

```bash
pip install strands-agents-tools
```

To install the dependencies for optional tools:

```bash
pip install strands-agents-tools[mem0_memory]
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/strands-agents/tools.git
cd tools

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Tools Overview

Below is a comprehensive table of all available tools, how to use them with an agent, and typical use cases:

| Tool | Agent Usage | Use Case |
|------|-------------|----------|
| file_read | `agent.tool.file_read(path="path/to/file.txt")` | Reading configuration files, parsing code files, loading datasets |
| file_write | `agent.tool.file_write(path="path/to/file.txt", content="file content")` | Writing results to files, creating new files, saving output data |
| editor | `agent.tool.editor(command="view", path="path/to/file.py")` | Advanced file operations like syntax highlighting, pattern replacement, and multi-file edits |
| shell | `agent.tool.shell(command="ls -la")` | Executing shell commands, interacting with the operating system, running scripts |
| http_request | `agent.tool.http_request(method="GET", url="https://api.example.com/data")` | Making API calls, fetching web data, sending data to external services |
| python_repl | `agent.tool.python_repl(code="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())")` | Running Python code snippets, data analysis, executing complex logic with user confirmation for security |
| calculator | `agent.tool.calculator(expression="2 * sin(pi/4) + log(e**2)")` | Performing mathematical operations, symbolic math, equation solving |
| use_aws | `agent.tool.use_aws(service_name="s3", operation_name="list_buckets", parameters={}, region="us-west-2")` | Interacting with AWS services, cloud resource management |
| retrieve | `agent.tool.retrieve(text="What is STRANDS?")` | Retrieving information from Amazon Bedrock Knowledge Bases |
| nova_reels | `agent.tool.nova_reels(action="create", text="A cinematic shot of mountains", s3_bucket="my-bucket")` | Create high-quality videos using Amazon Bedrock Nova Reel with configurable parameters via environment variables |
| mem0_memory | `agent.tool.mem0_memory(action="store", content="Remember I like to tennis", user_id="alex")` | Store user and agent memories across agent runs to provide personalized experience |
| memory | `agent.tool.memory(action="retrieve", query="product features")` | Store, retrieve, list, and manage documents in Amazon Bedrock Knowledge Bases with configurable parameters via environment variables |
| environment | `agent.tool.environment(action="list", prefix="AWS_")` | Managing environment variables, configuration management |
| generate_image | `agent.tool.generate_image(prompt="A sunset over mountains")` | Creating AI-generated images for various applications |
| image_reader | `agent.tool.image_reader(image_path="path/to/image.jpg")` | Processing and reading image files for AI analysis |
| journal | `agent.tool.journal(action="write", content="Today's progress notes")` | Creating structured logs, maintaining documentation |
| think | `agent.tool.think(thought="Complex problem to analyze", cycle_count=3)` | Advanced reasoning, multi-step thinking processes |
| load_tool | `agent.tool.load_tool(path="path/to/custom_tool.py", name="custom_tool")` | Dynamically loading custom tools and extensions |
| swarm | `agent.tool.swarm(task="Analyze this problem", swarm_size=3, coordination_pattern="collaborative")` | Coordinating multiple AI agents to solve complex problems through collective intelligence |
| current_time | `agent.tool.current_time(timezone="US/Pacific")` | Get the current time in ISO 8601 format for a specified timezone |
| agent_graph | `agent.tool.agent_graph(agents=["agent1", "agent2"], connections=[{"from": "agent1", "to": "agent2"}])` | Create and visualize agent relationship graphs for complex multi-agent systems |
| cron | `agent.tool.cron(action="schedule", name="task", schedule="0 * * * *", command="backup.sh")` | Schedule and manage recurring tasks with cron job syntax |
| slack | `agent.tool.slack(action="post_message", channel="general", text="Hello team!")` | Interact with Slack workspace for messaging and monitoring |
| speak | `agent.tool.speak(message="Operation completed successfully", style="green", mode="polly")` | Output status messages with rich formatting and optional text-to-speech |
| stop | `agent.tool.stop(message="Process terminated by user request")` | Gracefully terminate agent execution with custom message |
| use_llm | `agent.tool.use_llm(prompt="Analyze this data", system_prompt="You are a data analyst")` | Create nested AI loops with customized system prompts for specialized tasks |
| workflow | `agent.tool.workflow(action="create", name="data_pipeline", steps=[{"tool": "file_read"}, {"tool": "python_repl"}])` | Define, execute, and manage multi-step automated workflows |

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

### Shell Commands

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
```

### Python Code Execution

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

#### Mem0 Memory Tool

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| OPENSEARCH_HOST | OpenSearch Host URL | None |

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

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## ⚠️ Preview Status

Strands Agents is currently in public preview. During this period:
- APIs may change as we refine the SDK
- We welcome feedback and contributions
