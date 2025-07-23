# LlamaAgent Master Program

## Complete AI Agent System with Dynamic Task Planning and Subagent Spawning

The LlamaAgent Master Program is a production-ready, enterprise-grade AI agent orchestration system that provides:

- **Dynamic Task Planning**: Automatically decomposes complex tasks into manageable subtasks
- **Intelligent Agent Spawning**: Creates specialized agents on-demand based on task requirements
- **OpenAI Agents SDK Integration**: Full compatibility with OpenAI's agent ecosystem
- **Hierarchical Agent Management**: Coordinate teams of agents with parent-child relationships
- **Real-time Monitoring**: WebSocket-based progress tracking and system monitoring
- **Resource Management**: Smart allocation and tracking of computational resources

## Features

### Target Dynamic Task Planning
- AI-powered task decomposition
- Dependency resolution and critical path analysis
- Parallel execution optimization
- Automatic resource allocation

### Agent Intelligent Agent Spawning
- Role-based agent creation (Coordinator, Researcher, Analyzer, Executor, etc.)
- On-demand spawning based on task requirements
- Hierarchical team structures
- Agent lifecycle management

###  OpenAI Integration
- Native OpenAI Agents SDK support
- Budget tracking and cost management
- Tool execution framework
- Conversation history management

### Results Real-time Monitoring
- WebSocket progress updates
- Live hierarchy visualization
- Resource usage tracking
- Performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llamaagent.git
cd llamaagent

# Install dependencies
pip install -e .

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"  # Optional, for OpenAI integration
export DATABASE_URL="postgresql://user:pass@localhost/llamaagent"  # Optional, for persistence
```

## Quick Start

### 1. Start the API Server

```bash
python llamaagent_master_program.py server --port 8000
```

### 2. Execute a Task via CLI

```bash
# Simple task execution
python llamaagent_master_program.py execute "Build a web scraper for e-commerce products"

# With OpenAI integration
python llamaagent_master_program.py execute "Analyze customer sentiment from reviews" \
  --openai-key "your-api-key" \
  --max-agents 15
```

### 3. Run Interactive Demo

```bash
python llamaagent_master_program.py demo
```

### 4. Monitor System in Real-time

```bash
python llamaagent_master_program.py monitor
```

## API Usage

### Create a Master Task

```python
import requests

# Create a complex task with auto-decomposition
response = requests.post("http://localhost:8000/api/v1/tasks", json={
    "task_description": "Create a data pipeline that extracts data from APIs, transforms it, and loads into a database",
    "auto_decompose": True,
    "auto_spawn": True,
    "max_agents": 10,
    "enable_openai": True,
    "openai_api_key": "your-api-key",
    "priority": "high"
})

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Subtasks: {result['total_subtasks']}")
print(f"Spawned Agents: {result['spawned_agents']}")
```

### Monitor Progress via WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log(`Task ${update.task_id}: ${update.status} (${update.progress}%)`);
};
```

### Get System Status

```python
# Get comprehensive system status
response = requests.get("http://localhost:8000/api/v1/status")
status = response.json()

print(f"Active Agents: {status['active_agents']}")
print(f"Running Tasks: {status['running_tasks']}")
print(f"Resource Usage: {status['resource_usage']}")
```

### View Agent Hierarchy

```python
# Get hierarchical view of all agents
response = requests.get("http://localhost:8000/api/v1/hierarchy")
hierarchy = response.json()

# Returns nested structure showing agent relationships
print(hierarchy['visualization'])  # Text-based tree view
```

## Architecture

### Core Components

1. **MasterOrchestrator**: Central coordinator managing all operations
2. **TaskPlanner**: Decomposes tasks and manages dependencies
3. **AgentSpawner**: Creates and manages agent lifecycles
4. **AgentHierarchy**: Maintains parent-child relationships
5. **ResourceMonitor**: Tracks and allocates system resources
6. **OpenAIAgentsManager**: Integrates with OpenAI Agents SDK

### Task Execution Flow

```
User Request
    ↓
Task Planning (AI-powered decomposition)
    ↓
Dependency Analysis
    ↓
Agent Spawning (Role-based assignment)
    ↓
Parallel Execution
    ↓
Progress Monitoring
    ↓
Result Synthesis
```

## Advanced Usage

### Custom Task Decomposition

```python
from llamaagent.planning.task_planner import TaskDecomposer, Task

# Create custom decomposer
decomposer = TaskDecomposer()

# Define custom task type
task = Task(
    name="Custom Analysis",
    task_type="custom_analysis",
    metadata={"domain": "finance"}
)

# Decompose into subtasks
subtasks = decomposer.decompose(task)
```

### Manual Agent Spawning

```python
# Spawn a specialized agent
response = requests.post("http://localhost:8000/api/v1/agents/spawn", json={
    "task": "Research latest AI developments",
    "role": "researcher",
    "tools": ["web_search", "file_reader"],
    "auto_plan": True
})
```

### Team-based Execution

```python
# The system automatically creates agent teams for complex tasks
# Example: Data science project team

response = requests.post("http://localhost:8000/api/v1/tasks", json={
    "task_description": "Develop a machine learning model for customer churn prediction",
    "auto_decompose": True,
    "auto_spawn": True,
    "max_agents": 20,  # Will create specialized team
    "metadata": {
        "team_structure": "hierarchical",
        "require_roles": ["coordinator", "researcher", "analyzer", "executor"]
    }
})
```

## Configuration

### Environment Variables

```bash
# Core settings
LLAMAAGENT_MAX_AGENTS=100
LLAMAAGENT_MAX_CONCURRENT_TASKS=50
LLAMAAGENT_DEFAULT_TIMEOUT=300

# OpenAI integration
OPENAI_API_KEY=your-api-key
LLAMAAGENT_DEFAULT_MODEL=gpt-4o-mini

# Resource limits
LLAMAAGENT_AGENT_MEMORY_MB=512
LLAMAAGENT_SYSTEM_MEMORY_MB=4096

# Features
LLAMAAGENT_ENABLE_AUTO_SPAWN=true
LLAMAAGENT_ENABLE_DYNAMIC_PLANNING=true
LLAMAAGENT_ENABLE_OPENAI_INTEGRATION=true
```

### System Configuration

Edit `MASTER_CONFIG` in the main program:

```python
MASTER_CONFIG = {
    "max_agents": 100,
    "max_concurrent_tasks": 50,
    "default_timeout": 300.0,
    "enable_auto_spawn": True,
    "enable_dynamic_planning": True,
    "enable_openai_integration": True,
    "default_model": "gpt-4o-mini",
    "agent_memory_mb": 512,
    "system_memory_mb": 4096,
}
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "llamaagent_master_program.py", "server", "--host", "0.0.0.0"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaagent-master
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamaagent-master
  template:
    metadata:
      labels:
        app: llamaagent-master
    spec:
      containers:
      - name: llamaagent
        image: llamaagent-master:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## Performance Optimization

### Resource Management

- **Memory Limits**: Each agent has configurable memory limits
- **API Call Budgets**: Track and limit external API usage
- **Concurrent Execution**: Parallel task execution with thread pools
- **Auto-scaling**: Dynamic agent spawning based on load

### Best Practices

1. **Task Granularity**: Break down tasks appropriately for parallel execution
2. **Agent Specialization**: Use role-based agents for specific task types
3. **Resource Monitoring**: Monitor resource usage via the `/api/v1/status` endpoint
4. **Caching**: Enable result caching for repetitive tasks
5. **Batch Processing**: Group similar tasks for efficient execution

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH=/path/to/llamaagent:$PYTHONPATH
   ```

2. **OpenAI Integration Issues**
   ```bash
   # Verify API key
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

3. **Resource Exhaustion**
   - Monitor via `/api/v1/status`
   - Adjust `max_agents` and memory limits
   - Enable agent cleanup

### Debug Mode

```bash
# Run with verbose logging
python llamaagent_master_program.py server --verbose

# Enable debug endpoints
export LLAMAAGENT_DEBUG=true
```

## Examples

### Example 1: Web Scraping Pipeline

```python
# Create a complete web scraping pipeline
task = """
Create a web scraping system that:
1. Scrapes product data from multiple e-commerce sites
2. Cleans and normalizes the data
3. Stores in a database
4. Generates daily reports
"""

response = requests.post("http://localhost:8000/api/v1/tasks", json={
    "task_description": task,
    "auto_decompose": True,
    "auto_spawn": True,
    "max_agents": 15
})
```

### Example 2: Data Analysis Project

```python
# Complex data analysis with multiple stages
task = """
Analyze customer behavior data:
1. Extract data from multiple sources (database, APIs, files)
2. Clean and preprocess the data
3. Perform exploratory data analysis
4. Build predictive models
5. Create visualization dashboard
6. Generate executive summary
"""

response = requests.post("http://localhost:8000/api/v1/tasks", json={
    "task_description": task,
    "priority": "critical",
    "metadata": {
        "require_specialized_agents": True,
        "enable_ml_tools": True
    }
})
```

### Example 3: Software Development Task

```python
# Automated software development
task = """
Build a REST API for user management:
1. Design database schema
2. Implement user models
3. Create CRUD endpoints
4. Add authentication
5. Write unit tests
6. Generate API documentation
7. Set up CI/CD pipeline
"""

response = requests.post("http://localhost:8000/api/v1/tasks", json={
    "task_description": task,
    "metadata": {
        "technology_stack": ["python", "fastapi", "postgresql"],
        "testing_required": True
    }
})
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Nik Jois**
Email: nikjois@llamasearch.ai

---

For more information and updates, visit the [LlamaAgent documentation](https://docs.llamaagent.ai).
