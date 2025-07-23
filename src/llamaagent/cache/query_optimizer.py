"""
Query Optimizer Module

Provides advanced query optimization capabilities for LLM caching:
- Query analysis and pattern recognition
- Cost-based optimization
- Batch processing optimization
- Parallel execution planning
- Cache-first strategies

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Query optimization strategies"""

    NONE = "none"
    BATCH = "batch"
    CACHE_FIRST = "cache_first"
    PARALLEL = "parallel"
    COST_BASED = "cost_based"
    ADAPTIVE = "adaptive"


@dataclass
class QueryPlan:
    """Execution plan for query optimization"""

    strategy: OptimizationStrategy
    estimated_cost: float
    estimated_time_ms: float
    parallelism: int = 1
    cache_hits_expected: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStats:
    """Statistics for query execution"""

    query_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tokens_used: int = 0
    cost: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None


class QueryOptimizer:
    """Advanced query optimization system"""

    def __init__(
        self,
        default_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
        batch_size: int = 10,
        batch_timeout_ms: int = 1000,
        enable_statistics: bool = True,
        cost_per_token: float = 0.0001,
    ):
        self.default_strategy = default_strategy
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_statistics = enable_statistics
        self.cost_per_token = cost_per_token

        # Query batching
        self.pending_queries: Dict[str, List[Tuple[str, asyncio.Future]]] = defaultdict(
            list
        )

        # Statistics tracking
        self.query_stats: List[QueryStats] = []
        self.pattern_cache: Dict[str, float] = {}
        self.execution_history: Dict[str, List[float]] = defaultdict(list)

        # Optimization rules
        self.optimization_rules: List[Callable] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default optimization rules"""
        # Rule: Batch similar queries
        self.optimization_rules.append(self._batch_similar_queries_rule)

        # Rule: Cache frequently accessed patterns
        self.optimization_rules.append(self._cache_frequent_patterns_rule)

        # Rule: Parallelize independent queries
        self.optimization_rules.append(self._parallelize_independent_queries_rule)

    async def optimize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Optimize a single query"""
        if self.default_strategy == OptimizationStrategy.NONE:
            return QueryPlan(
                strategy=OptimizationStrategy.NONE,
                estimated_cost=self._estimate_cost(query),
                estimated_time_ms=self._estimate_time(query),
            )

        if self.default_strategy == OptimizationStrategy.ADAPTIVE:
            return await self._adaptive_optimize(query, context)
        elif self.default_strategy == OptimizationStrategy.BATCH:
            return await self._batch_optimize(query, context)
        elif self.default_strategy == OptimizationStrategy.CACHE_FIRST:
            return await self._cache_first_optimize(query, context)
        elif self.default_strategy == OptimizationStrategy.PARALLEL:
            return await self._parallel_optimize(query, context)
        else:
            # Default to adaptive
            return await self._adaptive_optimize(query, context)

    async def optimize_batch(
        self, queries: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[QueryPlan]:
        """Optimize a batch of queries"""
        # Analyze query relationships
        groups = self._group_queries(queries)
        plans: List[QueryPlan] = []

        for group in groups:
            if len(group) > 1:
                # Create batch plan
                plan = QueryPlan(
                    strategy=OptimizationStrategy.BATCH,
                    estimated_cost=sum(self._estimate_cost(q) for q in group)
                    * 0.8,  # 20% batch discount
                    estimated_time_ms=max(self._estimate_time(q) for q in group),
                    parallelism=len(group),
                )
                plans.extend([plan] * len(group))
            else:
                # Single query
                plan = await self.optimize_query(group[0], context)
                plans.append(plan)

        return plans

    async def execute_query(
        self,
        query: str,
        executor: Callable[[str], Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute query with optimization"""
        # Get optimization plan
        plan = await self.optimize_query(query, context)

        # Execute based on strategy
        if plan.strategy == OptimizationStrategy.BATCH:
            return await self._execute_with_batching(query, executor, context)
        elif plan.strategy == OptimizationStrategy.PARALLEL:
            return await self._execute_with_parallelism(query, executor, context)
        elif plan.strategy == OptimizationStrategy.CACHE_FIRST:
            return await self._execute_with_caching(query, executor, context)
        else:
            # Direct execution
            return await executor(query)

    async def _adaptive_optimize(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Adaptive optimization based on query characteristics"""
        # Analyze query
        characteristics = self._analyze_query(query, context)

        # Apply optimization rules
        for rule in self.optimization_rules:
            plan = rule(query, characteristics)
            if plan:
                return plan

        # Default plan
        return QueryPlan(
            strategy=OptimizationStrategy.NONE,
            estimated_cost=self._estimate_cost(query),
            estimated_time_ms=self._estimate_time(query),
        )

    async def _batch_optimize(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Optimize for batching"""
        return QueryPlan(
            strategy=OptimizationStrategy.BATCH,
            estimated_cost=self._estimate_cost(query) * 0.8,
            estimated_time_ms=self._estimate_time(query),
            parallelism=self.batch_size,
        )

    async def _cache_first_optimize(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Optimize with cache-first approach"""
        pattern = self._extract_pattern(query)
        cache_hit_prob = self._estimate_cache_hit_probability(pattern)

        return QueryPlan(
            strategy=OptimizationStrategy.CACHE_FIRST,
            estimated_cost=self._estimate_cost(query) * (1 - cache_hit_prob),
            estimated_time_ms=self._estimate_time(query) * (1 - cache_hit_prob * 0.9),
            cache_hits_expected=int(cache_hit_prob * 10),
        )

    async def _parallel_optimize(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Optimize for parallel execution"""
        # Check if query can be decomposed
        sub_queries = self._decompose_query(query)

        if len(sub_queries) > 1:
            return QueryPlan(
                strategy=OptimizationStrategy.PARALLEL,
                estimated_cost=sum(self._estimate_cost(q) for q in sub_queries),
                estimated_time_ms=max(self._estimate_time(q) for q in sub_queries),
                parallelism=len(sub_queries),
            )

        # Cannot parallelize
        return await self._adaptive_optimize(query, context)

    async def _cost_based_optimize(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Cost-based optimization"""
        # Generate candidate plans
        candidates: List[QueryPlan] = []

        # Plan 1: Direct execution
        direct_plan = QueryPlan(
            strategy=OptimizationStrategy.NONE,
            estimated_cost=self._estimate_cost(query),
            estimated_time_ms=self._estimate_time(query),
        )
        candidates.append(direct_plan)

        # Plan 2: With caching
        cache_plan = await self._cache_first_optimize(query, context)
        candidates.append(cache_plan)

        # Plan 3: With parallelism
        parallel_plan = await self._parallel_optimize(query, context)
        candidates.append(parallel_plan)

        # Select best plan based on cost
        best_plan = min(candidates, key=lambda p: p.estimated_cost)
        return best_plan

    def _analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze query characteristics"""
        characteristics = {
            "length": len(query),
            "complexity": self._estimate_complexity(query),
            "pattern": self._extract_pattern(query),
            "is_batchable": self._is_batchable(query),
            "is_decomposable": self._is_decomposable(query),
            "frequency": self._get_query_frequency(query),
            "sub_query_count": len(self._decompose_query(query)),
        }

        if context:
            characteristics.update(context)

        return characteristics

    def _batch_similar_queries_rule(
        self, query: str, characteristics: Dict[str, Any]
    ) -> Optional[QueryPlan]:
        """Rule: Batch similar queries together"""
        if (
            characteristics.get("is_batchable")
            and characteristics.get("frequency", 0) > 10
        ):
            return QueryPlan(
                strategy=OptimizationStrategy.BATCH,
                estimated_cost=self._estimate_cost(query) * 0.8,
                estimated_time_ms=self._estimate_time(query),
                parallelism=self.batch_size,
            )
        return None

    def _cache_frequent_patterns_rule(
        self, query: str, characteristics: Dict[str, Any]
    ) -> Optional[QueryPlan]:
        """Rule: Cache frequently accessed patterns"""
        pattern = characteristics.get("pattern")
        if pattern and pattern in self.pattern_cache:
            avg_time = self.pattern_cache[pattern]
            if avg_time > 1000:  # If average time > 1 second
                return QueryPlan(
                    strategy=OptimizationStrategy.CACHE_FIRST,
                    estimated_cost=self._estimate_cost(query) * 0.3,
                    estimated_time_ms=avg_time * 0.1,
                    cache_hits_expected=8,
                )
        return None

    def _parallelize_independent_queries_rule(
        self, query: str, characteristics: Dict[str, Any]
    ) -> Optional[QueryPlan]:
        """Rule: Parallelize independent sub-queries"""
        if (
            characteristics.get("is_decomposable")
            and characteristics.get("sub_query_count", 0) > 1
        ):
            sub_queries = self._decompose_query(query)
            return QueryPlan(
                strategy=OptimizationStrategy.PARALLEL,
                estimated_cost=sum(self._estimate_cost(q) for q in sub_queries),
                estimated_time_ms=max(self._estimate_time(q) for q in sub_queries),
                parallelism=len(sub_queries),
            )
        return None

    async def _execute_with_batching(
        self,
        query: str,
        executor: Callable[[str], Any],
        context: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Any:
        """Execute query with batching"""
        timeout = timeout_ms or self.batch_timeout_ms

        # Create future for result
        future = asyncio.Future()

        # Add to pending queries
        query_type = self._extract_pattern(query)
        self.pending_queries[query_type].append((query, future))

        # If batch is full or timeout, execute batch
        if len(self.pending_queries[query_type]) >= self.batch_size:
            await self._flush_batch(query_type, executor)
        else:
            # Set timeout to flush batch
            asyncio.create_task(
                self._flush_after_timeout(query_type, executor, timeout)
            )

        return await future

    async def _execute_with_parallelism(
        self,
        query: str,
        executor: Callable[[str], Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute query with parallelism"""
        sub_queries = self._decompose_query(query)

        if len(sub_queries) <= 1:
            return await executor(query)

        # Execute sub-queries in parallel
        tasks = [executor(sub_query) for sub_query in sub_queries]
        results = await asyncio.gather(*tasks)

        # Combine results
        return self._combine_results(results)

    async def _execute_with_caching(
        self,
        query: str,
        executor: Callable[[str], Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute query with caching"""
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_result = await self._get_from_cache(cache_key)

        if cached_result is not None:
            return cached_result

        # Execute and cache
        result = await executor(query)
        await self._store_in_cache(cache_key, result)

        return result

    # Utility methods

    def _estimate_cost(self, query: str) -> float:
        """Estimate cost of query execution"""
        # Simple heuristic based on query length
        return len(query.split()) * self.cost_per_token

    def _estimate_time(self, query: str) -> float:
        """Estimate execution time in milliseconds"""
        # Simple heuristic based on query complexity
        base_time = 100  # Base 100ms
        complexity_factor = self._estimate_complexity(query)
        return base_time * complexity_factor

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity"""
        # Simple heuristic
        words = query.split()
        return 1.0 + (len(words) / 100.0)

    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query for caching"""
        # Simple pattern extraction
        words = query.lower().split()
        # Keep only first 5 words as pattern
        pattern_words = words[:5]
        return " ".join(pattern_words)

    def _estimate_cache_hit_probability(self, pattern: str) -> float:
        """Estimate cache hit probability for pattern"""
        if pattern in self.pattern_cache:
            return min(0.9, self.pattern_cache[pattern] / 1000.0)
        return 0.1

    def _is_batchable(self, query: str) -> bool:
        """Check if query can be batched"""
        # Simple heuristic - short queries are more batchable
        return len(query.split()) < 20

    def _is_decomposable(self, query: str) -> bool:
        """Check if query can be decomposed"""
        # Simple heuristic - check for "and", "or", etc.
        keywords = ["and", "or", "also", "furthermore", "additionally"]
        return any(keyword in query.lower() for keyword in keywords)

    def _get_query_frequency(self, query: str) -> int:
        """Get frequency of similar queries"""
        pattern = self._extract_pattern(query)
        return len(self.execution_history.get(pattern, []))

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose query into sub-queries"""
        # Simple decomposition based on conjunctions
        separators = [" and ", " or ", " also ", " furthermore ", " additionally "]

        sub_queries = [query]
        for separator in separators:
            new_sub_queries = []
            for sub_query in sub_queries:
                new_sub_queries.extend(sub_query.split(separator))
            sub_queries = new_sub_queries

        return [q.strip() for q in sub_queries if q.strip()]

    def _group_queries(self, queries: List[str]) -> List[List[str]]:
        """Group similar queries together"""
        groups = defaultdict(list)

        for query in queries:
            pattern = self._extract_pattern(query)
            groups[pattern].append(query)

        return list(groups.values())

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache"""
        # Placeholder - implement actual cache logic
        return None

    async def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache"""
        # Placeholder - implement actual cache logic
        pass

    def _combine_results(self, results: List[Any]) -> Any:
        """Combine parallel execution results"""
        # Simple combination - join if strings
        if all(isinstance(r, str) for r in results):
            return " ".join(results)
        return results

    async def _flush_batch(
        self, query_type: str, executor: Callable[[str], Any]
    ) -> None:
        """Flush batch of queries"""
        batch = self.pending_queries[query_type]
        if not batch:
            return

        # Clear pending queries
        self.pending_queries[query_type] = []

        # Execute batch
        queries = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        try:
            # Execute all queries
            results = await asyncio.gather(*[executor(q) for q in queries])

            # Set results on futures
            for future, result in zip(futures, results, strict=False):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            # Set exception on all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _flush_after_timeout(
        self, query_type: str, executor: Callable[[str], Any], timeout_ms: int
    ) -> None:
        """Flush batch after timeout"""
        await asyncio.sleep(timeout_ms / 1000.0)
        await self._flush_batch(query_type, executor)

    def record_execution(
        self, query: str, duration_ms: float, cost: float, cache_hit: bool = False
    ) -> None:
        """Record query execution statistics"""
        if not self.enable_statistics:
            return

        pattern = self._extract_pattern(query)
        self.execution_history[pattern].append(duration_ms)

        # Update pattern cache
        if pattern not in self.pattern_cache:
            self.pattern_cache[pattern] = duration_ms
        else:
            # Moving average
            self.pattern_cache[pattern] = (
                self.pattern_cache[pattern] + duration_ms
            ) / 2

        # Record stats
        stats = QueryStats(
            query_id=self._get_cache_key(query),
            start_time=time.time(),
            duration_ms=duration_ms,
            cost=cost,
            cache_hit=cache_hit,
        )
        self.query_stats.append(stats)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.query_stats:
            return {}

        total_queries = len(self.query_stats)
        successful_queries = sum(1 for s in self.query_stats if s.error is None)
        cache_hits = sum(1 for s in self.query_stats if s.cache_hit)

        total_time = sum(s.duration_ms or 0 for s in self.query_stats)
        total_cost = sum(s.cost for s in self.query_stats)
        total_tokens = sum(s.tokens_used for s in self.query_stats)

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": (
                successful_queries / total_queries if total_queries > 0 else 0
            ),
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_queries if total_queries > 0 else 0,
            "total_time_ms": total_time,
            "average_time_ms": total_time / total_queries if total_queries > 0 else 0,
            "total_cost": total_cost,
            "average_cost": total_cost / total_queries if total_queries > 0 else 0,
            "total_tokens": total_tokens,
            "patterns_cached": len(self.pattern_cache),
            "optimization_rules": len(self.optimization_rules),
        }

    def clear_statistics(self) -> None:
        """Clear all statistics"""
        self.query_stats.clear()
        self.execution_history.clear()
        self.pattern_cache.clear()

    def add_optimization_rule(
        self, rule: Callable[[str, Dict[str, Any]], Optional[QueryPlan]]
    ) -> None:
        """Add custom optimization rule"""
        self.optimization_rules.append(rule)

    def remove_optimization_rule(
        self, rule: Callable[[str, Dict[str, Any]], Optional[QueryPlan]]
    ) -> None:
        """Remove optimization rule"""
        if rule in self.optimization_rules:
            self.optimization_rules.remove(rule)
