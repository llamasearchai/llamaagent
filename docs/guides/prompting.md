# Advanced Prompting Guide

This guide covers advanced prompting techniques available in LlamaAgent, including DSPy integration, chain-of-thought reasoning, and custom strategies.

## Table of Contents

1. [Introduction to Prompting](#introduction-to-prompting)
2. [Built-in Strategies](#built-in-strategies)
3. [DSPy Integration](#dspy-integration)
4. [Chain of Thought Prompting](#chain-of-thought-prompting)
5. [Compound Prompting](#compound-prompting)
6. [Custom Strategies](#custom-strategies)
7. [Best Practices](#best-practices)

## Introduction to Prompting

Prompting is the art and science of instructing language models to produce desired outputs. LlamaAgent provides a comprehensive framework for implementing state-of-the-art prompting techniques.

### Basic Usage

```python
from llamaagent import Agent
from llamaagent.prompting import ChainOfThoughtStrategy

agent = Agent(
    name="ReasoningAgent",
    llm=llm,
    prompting_strategy=ChainOfThoughtStrategy()
)
```

## Built-in Strategies

### 1. Chain of Thought (CoT)

Break down complex reasoning into steps:

```python
from llamaagent.prompting import ChainOfThoughtStrategy

cot = ChainOfThoughtStrategy(
    style="detailed",  # or "concise"
    include_confidence=True
)

agent = Agent(
    prompting_strategy=cot,
    system_prompt="Think step by step before answering."
)

# Example usage
response = await agent.run(
    "If a train travels 120 miles in 2 hours, and then 180 miles "
    "in 3 hours, what is its average speed for the entire journey?"
)
# Output includes step-by-step reasoning
```

### 2. Tree of Thoughts (ToT)

Explore multiple reasoning paths:

```python
from llamaagent.prompting import TreeOfThoughtsStrategy

tot = TreeOfThoughtsStrategy(
    num_thoughts=3,  # Number of initial thoughts
    num_steps=3,     # Depth of reasoning
    value_threshold=0.7
)

agent = Agent(prompting_strategy=tot)

# Complex problem solving
response = await agent.run(
    "Design a sustainable city for 1 million people"
)
# Explores multiple design approaches
```

### 3. Graph of Thoughts (GoT)

Non-linear reasoning with connections:

```python
from llamaagent.prompting import GraphOfThoughtsStrategy

got = GraphOfThoughtsStrategy(
    max_nodes=10,
    edge_threshold=0.6,
    enable_backtracking=True
)

# Useful for complex analysis
response = await agent.run(
    "Analyze the causes and effects of climate change"
)
```

### 4. Least-to-Most Prompting

Decompose problems from simple to complex:

```python
from llamaagent.prompting import LeastToMostStrategy

ltm = LeastToMostStrategy(
    max_decomposition_depth=5,
    auto_decompose=True
)

# Excellent for educational explanations
response = await agent.run(
    "Explain quantum computing to a beginner"
)
```

### 5. Self-Consistency

Multiple reasoning paths with voting:

```python
from llamaagent.prompting import SelfConsistencyStrategy

sc = SelfConsistencyStrategy(
    num_samples=5,
    temperature_range=(0.5, 0.9),
    aggregation="majority_vote"  # or "weighted", "confidence"
)

# Improves accuracy on reasoning tasks
response = await agent.run(
    "Is the following statement logically valid? ..."
)
```

## DSPy Integration

LlamaAgent integrates with DSPy for automatic prompt optimization:

### Basic DSPy Usage

```python
from llamaagent.prompting import DSPyOptimizer, DSPySignature

# Define task signature
signature = DSPySignature(
    name="QuestionAnswering",
    description="Answer questions based on context",
    input_fields={
        "context": "The context to use",
        "question": "The question to answer"
    },
    output_fields={
        "answer": "The answer to the question",
        "confidence": "Confidence score (0-1)"
    }
)

# Create optimizer
optimizer = DSPyOptimizer(
    signature=signature,
    num_demonstrations=10,
    optimization_metric="f1_score"
)

# Optimize prompts with training data
training_data = [
    {
        "context": "The Earth is round.",
        "question": "What shape is Earth?",
        "answer": "Round",
        "confidence": 1.0
    },
    # ... more examples
]

optimized_module = await optimizer.optimize(
    training_data=training_data,
    validation_data=validation_data
)

# Use optimized module
agent = Agent(
    prompting_strategy=optimized_module
)
```

### Advanced DSPy Features

```python
# Compound DSPy modules
from llamaagent.prompting import DSPyChain

chain = DSPyChain([
    DSPyModule("Decompose", decompose_signature),
    DSPyModule("Solve", solve_signature),
    DSPyModule("Combine", combine_signature)
])

# Automatic few-shot learning
chain.bootstrap(examples=training_examples)

# Adaptive optimization
optimizer = DSPyOptimizer(
    signature=signature,
    optimizer_class="MIPRO",  # or "BootstrapFewShot", "COPRO"
    teleprompter_params={
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 16
    }
)
```

## Chain of Thought Prompting

### Zero-Shot CoT

```python
from llamaagent.prompting import ZeroShotCoT

strategy = ZeroShotCoT(
    trigger="Let's think step by step.",
    format="numbered"  # or "bullet", "narrative"
)

response = await agent.run(
    "What is the probability of rolling two sixes with fair dice?"
)
```

### Few-Shot CoT

```python
from llamaagent.prompting import FewShotCoT

examples = [
    {
        "question": "If it takes 5 machines 5 minutes to make 5 widgets...",
        "reasoning": "Step 1: 5 machines make 5 widgets in 5 minutes...",
        "answer": "5 minutes"
    }
]

strategy = FewShotCoT(
    examples=examples,
    num_examples=3,
    example_selection="semantic"  # or "random", "diverse"
)
```

### Auto-CoT

Automatically generate reasoning examples:

```python
from llamaagent.prompting import AutoCoT

strategy = AutoCoT(
    num_clusters=5,
    questions_per_cluster=2,
    diversity_threshold=0.7
)

# Automatically creates diverse examples
await strategy.generate_examples(task_description)
```

## Compound Prompting

Combine multiple prompting techniques:

### Compound Strategy Builder

```python
from llamaagent.prompting import CompoundStrategy

compound = CompoundStrategy.builder()
    .add_stage("decompose", LeastToMostStrategy())
    .add_stage("reason", ChainOfThoughtStrategy())
    .add_stage("verify", SelfConsistencyStrategy())
    .add_stage("refine", CritiqueStrategy())
    .build()

agent = Agent(prompting_strategy=compound)
```

### Conditional Strategies

```python
from llamaagent.prompting import ConditionalStrategy

strategy = ConditionalStrategy({
    "math": ChainOfThoughtStrategy(),
    "creative": TreeOfThoughtsStrategy(),
    "factual": DirectStrategy(),
    "default": ReasoningStrategy()
})

# Automatically selects strategy based on task
strategy.set_classifier(task_classifier_model)
```

### Pipeline Strategies

```python
from llamaagent.prompting import PromptPipeline

pipeline = PromptPipeline([
    ("extract", ExtractionPrompt()),
    ("analyze", AnalysisPrompt()),
    ("synthesize", SynthesisPrompt())
])

# Each stage processes the output of the previous
result = await pipeline.run(input_data)
```

## Custom Strategies

Create your own prompting strategies:

### Basic Custom Strategy

```python
from llamaagent.prompting import PromptingStrategy

class SocraticStrategy(PromptingStrategy):
    """Guide reasoning through questions"""
    
    def format_prompt(self, task: str, context: dict) -> str:
        questions = self.generate_questions(task)
        
        prompt = f"""Answer by exploring these questions:
        
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(questions))}

Task: {task}
Context: {context}

Explore each question before providing your final answer."""
        
        return prompt
    
    def generate_questions(self, task: str) -> List[str]:
        # Generate guiding questions based on task
        return [
            "What are we trying to solve?",
            "What information do we have?",
            "What assumptions are we making?",
            "What are the possible approaches?",
            "What are the trade-offs?"
        ]

# Use custom strategy
agent = Agent(prompting_strategy=SocraticStrategy())
```

### Advanced Custom Strategy

```python
class AdaptiveReasoningStrategy(PromptingStrategy):
    """Adapts reasoning approach based on task complexity"""
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.strategies = {
            "simple": DirectStrategy(),
            "moderate": ChainOfThoughtStrategy(),
            "complex": TreeOfThoughtsStrategy()
        }
    
    async def format_prompt(self, task: str, context: dict) -> str:
        # Analyze task complexity
        complexity = await self.complexity_analyzer.analyze(task)
        
        # Select appropriate strategy
        strategy = self.strategies[complexity.level]
        
        # Add complexity-aware instructions
        base_prompt = await strategy.format_prompt(task, context)
        
        return f"""Task Complexity: {complexity.level}
Reasoning Approach: {strategy.__class__.__name__}

{base_prompt}

Additional Considerations:
- Time estimate: {complexity.estimated_time}
- Key challenges: {', '.join(complexity.challenges)}
- Suggested tools: {', '.join(complexity.suggested_tools)}
"""
```

### Meta-Learning Strategy

```python
class MetaLearningStrategy(PromptingStrategy):
    """Learns from previous interactions"""
    
    def __init__(self, memory_store):
        self.memory = memory_store
        self.performance_tracker = PerformanceTracker()
    
    async def format_prompt(self, task: str, context: dict) -> str:
        # Retrieve similar past tasks
        similar_tasks = await self.memory.search_similar(task, k=5)
        
        # Analyze what worked well
        successful_approaches = self.analyze_successes(similar_tasks)
        
        # Build adaptive prompt
        prompt = f"""Based on previous experience with similar tasks:

Successful Approaches:
{self.format_approaches(successful_approaches)}

Current Task: {task}

Apply the most relevant approach, adapting as needed."""
        
        return prompt
    
    async def post_process(self, response: str, feedback: dict):
        """Learn from the interaction"""
        await self.performance_tracker.record(
            task=task,
            response=response,
            feedback=feedback
        )
        
        # Update strategy based on performance
        if feedback.get("success", False):
            await self.reinforce_approach()
```

## Best Practices

### 1. Strategy Selection

Choose strategies based on task type:

```python
# Mathematical/Logical tasks
math_agent = Agent(
    prompting_strategy=ChainOfThoughtStrategy(style="detailed")
)

# Creative tasks
creative_agent = Agent(
    prompting_strategy=TreeOfThoughtsStrategy(num_thoughts=5)
)

# Factual queries
factual_agent = Agent(
    prompting_strategy=DirectStrategy(include_sources=True)
)

# Complex analysis
analysis_agent = Agent(
    prompting_strategy=CompoundStrategy([
        DecompositionStrategy(),
        ParallelAnalysisStrategy(),
        SynthesisStrategy()
    ])
)
```

### 2. Prompt Engineering Tips

```python
# Use clear structure
structured_prompt = PromptTemplate("""
# Task
{task_description}

# Context
{context}

# Requirements
{requirements}

# Output Format
{output_format}
""")

# Include examples when helpful
few_shot_prompt = FewShotTemplate(
    examples=load_examples("domain_specific.json"),
    example_separator="\n---\n"
)

# Be specific about constraints
constrained_prompt = ConstrainedPrompt(
    task=task,
    constraints=[
        "Use only information from the provided context",
        "Cite sources for all claims",
        "Limit response to 200 words"
    ]
)
```

### 3. Performance Optimization

```python
# Cache prompting results
from llamaagent.cache import PromptCache

cached_strategy = CachedStrategy(
    base_strategy=ChainOfThoughtStrategy(),
    cache=PromptCache(ttl=3600)
)

# Batch similar prompts
batch_optimizer = BatchPromptOptimizer(
    strategy=strategy,
    batch_size=10,
    similarity_threshold=0.9
)

# Profile prompt performance
from llamaagent.profiling import PromptProfiler

profiler = PromptProfiler()
with profiler.profile("complex_reasoning"):
    response = await agent.run(task)

print(profiler.get_report())
```

### 4. Error Handling

```python
# Robust prompting with fallbacks
robust_strategy = RobustStrategy(
    primary=ComplexStrategy(),
    fallbacks=[
        SimplerStrategy(),
        BasicStrategy()
    ],
    error_handlers={
        "timeout": lambda: "The task is taking too long...",
        "parse_error": lambda: "Retrying with clearer format..."
    }
)

# Validation and retry
validated_strategy = ValidatedStrategy(
    base_strategy=strategy,
    validators=[
        OutputFormatValidator(),
        FactualityValidator(),
        CoherenceValidator()
    ],
    max_retries=3
)
```

### 5. Monitoring and Improvement

```python
# Track prompt effectiveness
from llamaagent.monitoring import PromptMonitor

monitor = PromptMonitor()
monitor.track(
    strategy_name="chain_of_thought_v2",
    task_type="reasoning",
    success_criteria=["accuracy", "clarity", "speed"]
)

# A/B testing strategies
from llamaagent.testing import PromptABTest

ab_test = PromptABTest(
    strategy_a=ChainOfThoughtStrategy(),
    strategy_b=TreeOfThoughtsStrategy(),
    metric="task_success_rate"
)

results = await ab_test.run(test_tasks, num_trials=100)
print(f"Best strategy: {results.winner}")

# Continuous improvement
from llamaagent.optimization import PromptEvolution

evolver = PromptEvolution(
    population_size=10,
    mutation_rate=0.1,
    fitness_function=task_performance_score
)

evolved_strategy = await evolver.evolve(
    initial_strategies=[strategy1, strategy2],
    generations=20
)
```

## Advanced Examples

### Research Paper Analysis

```python
research_strategy = CompoundStrategy.builder()
    .add_stage("extract", ExtractionStrategy(
        targets=["methodology", "results", "conclusions"]
    ))
    .add_stage("critique", CriticalAnalysisStrategy(
        criteria=["validity", "novelty", "impact"]
    ))
    .add_stage("synthesize", SynthesisStrategy(
        format="structured_summary"
    ))
    .build()

agent = Agent(
    prompting_strategy=research_strategy,
    tools=[PDFReader(), CitationChecker(), StatisticalAnalyzer()]
)
```

### Code Generation

```python
code_gen_strategy = CodeGenerationStrategy(
    language="python",
    style_guide="PEP8",
    include_tests=True,
    include_docs=True
)

# With iterative refinement
refined_strategy = IterativeRefinementStrategy(
    base_strategy=code_gen_strategy,
    refinement_steps=[
        "optimize_performance",
        "improve_readability",
        "add_error_handling",
        "enhance_documentation"
    ]
)
```

### Multi-Modal Reasoning

```python
multimodal_strategy = MultiModalStrategy(
    text_strategy=ChainOfThoughtStrategy(),
    image_strategy=VisualReasoningStrategy(),
    fusion_method="late_fusion"
)

agent = Agent(
    prompting_strategy=multimodal_strategy,
    capabilities=["text", "vision"]
)
```

## Conclusion

LlamaAgent's prompting framework provides:

1. **Flexibility**: From simple to complex strategies
2. **Performance**: Optimized implementations
3. **Extensibility**: Easy to create custom strategies
4. **Integration**: Works with DSPy and other frameworks
5. **Production-Ready**: Monitoring, caching, and optimization

Choose the right strategy for your use case and iterate based on performance metrics.