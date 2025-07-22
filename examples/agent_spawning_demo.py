"""Demonstration of agent spawning and sub-agent management capabilities."""

import asyncio
import logging
from typing import Dict, Any

from llamaagent.agents.base import AgentConfig, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.spawning import (
    AgentSpawner,
    AgentPool,
    SpawnConfig,
    PoolConfig,
    AgentRelationship,
    MessageBus,
    MessageType,
)
from llamaagent.orchestrator import (
    AgentOrchestrator,
    WorkflowDefinition,
    WorkflowStep,
    OrchestrationStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_spawning():
    """Demonstrate basic agent spawning."""
    print("\n=== Basic Agent Spawning Demo ===")
    
    spawner = AgentSpawner()
    
    # Spawn a single agent
    print("\n1. Spawning a single agent...")
    result = await spawner.spawn_agent(
        task="Analyze market trends for technology sector",
        config=SpawnConfig(
            agent_config=AgentConfig(
                name="market_analyst",
                role=AgentRole.ANALYZER,
                description="Market analysis specialist",
            )
        )
    )
    
    if result.success:
        print(f" Spawned agent: {result.agent_id}")
        print(f"  Spawn time: {result.spawn_time:.2f}s")
        
        # Execute a task with the spawned agent
        agent_response = await result.agent.execute(
            "What are the top 3 technology trends for 2024?"
        )
        print(f"  Agent response: {agent_response.content[:100]}...")
    
    # Spawn a team
    print("\n2. Spawning a team of agents...")
    team_results = await spawner.spawn_team(
        task="Develop a comprehensive business strategy",
        team_size=3,
        roles=[AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.PLANNER],
    )
    
    print(f" Spawned team with {len(team_results)} agents:")
    for name, result in team_results.items():
        if result.success:
            print(f"  - {name}: {result.agent_id}")
    
    # Show hierarchy statistics
    stats = spawner.get_stats()
    print(f"\nHierarchy Statistics:")
    print(f"  Total spawned: {stats['total_spawned']}")
    print(f"  Active agents: {stats['hierarchy']['active_agents']}")
    print(f"  Max depth: {stats['hierarchy']['max_depth']}")
    
    # Clean up
    for result in team_results.values():
        if result.success:
            await spawner.terminate_agent(result.agent_id)


async def demo_agent_pool():
    """Demonstrate agent pool functionality."""
    print("\n=== Agent Pool Demo ===")
    
    # Create pool configuration
    pool_config = PoolConfig(
        min_agents=2,
        max_agents=5,
        initial_agents=3,
        auto_scale=True,
        strategy="least_loaded",
    )
    
    pool = AgentPool(config=pool_config)
    
    # Start the pool
    print("\n1. Starting agent pool...")
    await pool.start()
    print(f" Pool started with {pool.config.initial_agents} agents")
    
    # Submit tasks
    print("\n2. Submitting tasks to pool...")
    task_ids = []
    
    for i in range(5):
        task_id = await pool.submit_task(
            task=f"Process data batch {i+1}",
            priority=5 - i,  # Higher priority for earlier tasks
        )
        task_ids.append(task_id)
        print(f"  Submitted task {task_id}")
    
    # Wait for results
    print("\n3. Waiting for results...")
    for task_id in task_ids:
        try:
            result = await pool.get_result(task_id, timeout=10.0)
            if result:
                print(f"  Task {task_id}: {result.content[:50]}...")
        except TimeoutError:
            print(f"  Task {task_id}: Timed out")
    
    # Show pool statistics
    stats = pool.get_pool_stats()
    print(f"\nPool Statistics:")
    print(f"  Total agents: {stats.total_agents}")
    print(f"  Tasks completed: {stats.total_tasks_completed}")
    print(f"  Average task time: {stats.average_task_time:.2f}s")
    print(f"  Pool utilization: {stats.pool_utilization:.1%}")
    
    # Stop the pool
    await pool.stop()
    print("\n Pool stopped")


async def demo_communication():
    """Demonstrate inter-agent communication."""
    print("\n=== Agent Communication Demo ===")
    
    # Create message bus
    bus = MessageBus()
    spawner = AgentSpawner()
    
    # Spawn communicating agents
    print("\n1. Spawning communicating agents...")
    
    # Spawn coordinator
    coordinator_result = await spawner.spawn_agent(
        task="Coordinate research project",
        config=SpawnConfig(
            agent_config=AgentConfig(
                name="coordinator",
                role=AgentRole.COORDINATOR,
            )
        )
    )
    
    # Spawn workers
    worker_results = []
    for i in range(2):
        result = await spawner.spawn_agent(
            task=f"Research subtopic {i+1}",
            config=SpawnConfig(
                agent_config=AgentConfig(
                    name=f"worker_{i}",
                    role=AgentRole.RESEARCHER,
                ),
                parent_id=coordinator_result.agent_id,
            )
        )
        worker_results.append(result)
    
    print(" Spawned coordinator and 2 workers")
    
    # Create channels
    from llamaagent.spawning import AgentChannel
    
    channels = {}
    for agent_id in [coordinator_result.agent_id] + [r.agent_id for r in worker_results]:
        channel = AgentChannel(agent_id, bus)
        channels[agent_id] = channel
        
        # Subscribe to relevant message types
        await channel.subscribe({
            MessageType.TASK_REQUEST,
            MessageType.TASK_RESPONSE,
            MessageType.COORDINATION,
        })
    
    print("\n2. Demonstrating message exchange...")
    
    # Coordinator sends task to workers
    coordinator_channel = channels[coordinator_result.agent_id]
    
    # Send task request
    from llamaagent.spawning import Message
    
    for i, worker_result in enumerate(worker_results):
        message = Message(
            type=MessageType.TASK_REQUEST,
            sender=coordinator_result.agent_id,
            recipient=worker_result.agent_id,
            content=f"Please research aspect {i+1} of the topic",
            requires_response=True,
        )
        
        await coordinator_channel.send(message)
        print(f"  Coordinator -> Worker {i}: Task request sent")
    
    # Workers receive and respond
    for i, worker_result in enumerate(worker_results):
        worker_channel = channels[worker_result.agent_id]
        
        # Receive message
        received = await worker_channel.receive(timeout=1.0)
        if received:
            print(f"  Worker {i} received: {received.content}")
            
            # Send response
            await worker_channel.respond(
                received,
                f"Research on aspect {i+1} completed"
            )
    
    print("\n Communication demonstration complete")
    
    # Clean up
    await spawner.terminate_agent(coordinator_result.agent_id, cascade=True)


async def demo_orchestrator_integration():
    """Demonstrate orchestrator with spawning capabilities."""
    print("\n=== Orchestrator Integration Demo ===")
    
    # Create orchestrator with all features enabled
    orchestrator = AgentOrchestrator(
        enable_spawning=True,
        enable_pool=True,
        enable_communication=True,
    )
    
    # Define a dynamic workflow
    print("\n1. Creating dynamic workflow...")
    workflow = WorkflowDefinition(
        workflow_id="research_project",
        name="Research Project Workflow",
        description="Complete research project with dynamic agent spawning",
        steps=[
            WorkflowStep(
                step_id="research",
                agent_name="dynamic_researcher",  # Will be spawned
                task="Research the state of AI in healthcare",
            ),
            WorkflowStep(
                step_id="analyze",
                agent_name="dynamic_analyzer",  # Will be spawned
                task="Analyze the research findings and identify key trends",
                dependencies=["research"],
            ),
            WorkflowStep(
                step_id="report",
                agent_name="dynamic_writer",  # Will be spawned
                task="Write a comprehensive report on the findings",
                dependencies=["analyze"],
            ),
        ],
        strategy=OrchestrationStrategy.DYNAMIC,
    )
    
    orchestrator.register_workflow(workflow)
    print(" Workflow registered")
    
    # Execute workflow
    print("\n2. Executing workflow with dynamic spawning...")
    result = await orchestrator.execute_workflow("research_project")
    
    print(f"\n Workflow completed:")
    print(f"  Success: {result.success}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    print(f"  Steps completed: {len(result.results)}")
    
    # Show spawning statistics
    spawn_stats = orchestrator.get_spawning_stats()
    if spawn_stats:
        print(f"\nSpawning Statistics:")
        print(f"  Total agents spawned: {spawn_stats['total_spawned']}")
        print(f"  Active spawns: {spawn_stats['active_spawns']}")
    
    # Define a pool-based workflow
    print("\n3. Creating pool-based workflow...")
    pool_workflow = WorkflowDefinition(
        workflow_id="batch_processing",
        name="Batch Processing Workflow",
        description="Process multiple items using agent pool",
        steps=[
            WorkflowStep(
                step_id=f"process_{i}",
                agent_name="pool_worker",
                task=f"Process data item {i}",
                metadata={"priority": 10 - i},
            )
            for i in range(5)
        ],
        strategy=OrchestrationStrategy.POOL_BASED,
    )
    
    orchestrator.register_workflow(pool_workflow)
    
    # Start pool
    if orchestrator.agent_pool:
        await orchestrator.agent_pool.start()
        print(" Agent pool started")
    
    # Execute pool-based workflow
    print("\n4. Executing pool-based workflow...")
    pool_result = await orchestrator.execute_workflow("batch_processing")
    
    print(f"\n Pool workflow completed:")
    print(f"  Success: {pool_result.success}")
    print(f"  Items processed: {len(pool_result.results)}")
    
    # Show pool statistics
    pool_stats = orchestrator.get_pool_stats()
    if pool_stats:
        print(f"\nPool Statistics:")
        print(f"  Total agents: {pool_stats['total_agents']}")
        print(f"  Tasks completed: {pool_stats['total_tasks_completed']}")
        print(f"  Pool utilization: {pool_stats['pool_utilization']:.1%}")
    
    # Stop pool
    if orchestrator.agent_pool:
        await orchestrator.agent_pool.stop()
        print("\n Agent pool stopped")


async def main():
    """Run all demonstrations."""
    print("=== LlamaAgent Spawning System Demonstration ===")
    print("This demo showcases the autonomous agent spawning capabilities")
    
    try:
        # Run demonstrations
        await demo_basic_spawning()
        await demo_agent_pool()
        await demo_communication()
        await demo_orchestrator_integration()
        
        print("\n=== All demonstrations completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())