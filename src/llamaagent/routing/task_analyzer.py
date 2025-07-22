"""
Task analyzer for understanding coding task characteristics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken


class TaskType(Enum):
    """Types of coding tasks."""

    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    MIGRATION = "migration"
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"  # < 10 lines, simple logic
    SIMPLE = "simple"  # 10-50 lines, straightforward
    MODERATE = "moderate"  # 50-200 lines, some complexity
    COMPLEX = "complex"  # 200-500 lines, significant complexity
    VERY_COMPLEX = "very_complex"  # > 500 lines, high complexity


@dataclass
class TaskCharacteristics:
    """Characteristics of a coding task."""

    task_type: TaskType
    complexity: TaskComplexity
    languages: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    estimated_tokens: int = 0
    has_tests: bool = False
    requires_external_apis: bool = False
    requires_database: bool = False
    requires_ui: bool = False
    security_sensitive: bool = False
    performance_critical: bool = False
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskAnalyzer:
    """Analyzes coding tasks to determine their characteristics."""

    def __init__(self):
        # Task type indicators
        self.task_type_patterns = {
            TaskType.DEBUGGING: [
                r"\b(debug|fix|error|bug|issue|problem|crash|exception|traceback)\b",
                r"\b(not working|doesn't work|broken|failing)\b",
                r"\b(investigate|diagnose|troubleshoot)\b",
            ],
            TaskType.REFACTORING: [
                r"\b(refactor|restructure|reorganize|clean up|improve|optimize)\b",
                r"\b(rename|extract|inline|move|split|merge)\b",
                r"\b(design pattern|SOLID|DRY|KISS)\b",
            ],
            TaskType.CODE_GENERATION: [
                r"\b(create|generate|implement|build|write|add|new)\b",
                r"\b(function|class|method|component|module|feature)\b",
                r"\b(from scratch|boilerplate|scaffold)\b",
            ],
            TaskType.CODE_REVIEW: [
                r"\b(review|analyze|audit|inspect|evaluate)\b",
                r"\b(code quality|best practices|standards|conventions)\b",
                r"\b(feedback|suggestions|improvements)\b",
            ],
            TaskType.DOCUMENTATION: [
                r"\b(document|documentation|docs|README|docstring|comment)\b",
                r"\b(explain|describe|annotate|clarify)\b",
                r"\b(API reference|guide|tutorial)\b",
            ],
            TaskType.TESTING: [
                r"\b(test|testing|unit test|integration test|e2e|coverage)\b",
                r"\b(jest|pytest|mocha|jasmine|unittest)\b",
                r"\b(mock|stub|fixture|assertion)\b",
            ],
            TaskType.OPTIMIZATION: [
                r"\b(optimize|performance|speed up|faster|efficient)\b",
                r"\b(memory|CPU|latency|throughput|bottleneck)\b",
                r"\b(profile|benchmark|measure)\b",
            ],
            TaskType.ARCHITECTURE: [
                r"\b(architecture|design|structure|pattern|system)\b",
                r"\b(microservice|monolith|serverless|distributed)\b",
                r"\b(scalability|reliability|maintainability)\b",
            ],
            TaskType.MIGRATION: [
                r"\b(migrate|upgrade|port|convert|transition)\b",
                r"\b(legacy|deprecated|old version|new version)\b",
                r"\b(compatibility|breaking change)\b",
            ],
        }

        # Language detection patterns
        self.language_patterns = {
            "python": [
                r"\.py\b",
                r"\bpython\b",
                r"\bpip\b",
                r"\bdjango\b",
                r"\bflask\b",
                r"\bpandas\b",
            ],
            "javascript": [
                r"\.js\b",
                r"\bjavascript\b",
                r"\bnpm\b",
                r"\bnode\b",
                r"\breact\b",
                r"\bvue\b",
            ],
            "typescript": [
                r"\.ts\b",
                r"\btypescript\b",
                r"\btsconfig\b",
                r"\binterface\b",
                r"\btype\b",
            ],
            "java": [
                r"\.java\b",
                r"\bjava\b",
                r"\bmaven\b",
                r"\bgradle\b",
                r"\bspring\b",
            ],
            "cpp": [r"\.cpp\b", r"\.cc\b", r"\.cxx\b", r"\bc\+\+\b", r"\bcpp\b"],
            "csharp": [
                r"\.cs\b",
                r"\bc#\b",
                r"\bcsharp\b",
                r"\.net\b",
                r"\basync Task\b",
            ],
            "rust": [r"\.rs\b", r"\brust\b", r"\bcargo\b", r"\bcrate\b", r"\bfn\b.*->"],
            "go": [
                r"\.go\b",
                r"\bgolang\b",
                r"\bgo\b",
                r"\bfunc\b",
                r"\bpackage main\b",
            ],
            "ruby": [
                r"\.rb\b",
                r"\bruby\b",
                r"\brails\b",
                r"\bgem\b",
                r"\bdef\b.*end\b",
            ],
            "php": [r"\.php\b", r"\bphp\b", r"\blaravel\b", r"\bcomposer\b", r"<\?php"],
            "swift": [
                r"\.swift\b",
                r"\bswift\b",
                r"\bios\b",
                r"\bxcode\b",
                r"\bfunc\b",
            ],
            "kotlin": [r"\.kt\b", r"\bkotlin\b", r"\bandroid\b", r"\bfun\b"],
            "r": [r"\.r\b", r"\br language\b", r"\bggplot\b", r"\bdplyr\b"],
            "sql": [
                r"\.sql\b",
                r"\bSELECT\b",
                r"\bINSERT\b",
                r"\bUPDATE\b",
                r"\bDELETE\b",
            ],
            "html": [r"\.html?\b", r"<html", r"<body", r"<div", r"<span"],
            "css": [
                r"\.css\b",
                r"\bstylesheet\b",
                r"{\s*[a-z-]+:",
                r"\.class\b",
                r"#id\b",
            ],
        }

        # Framework detection patterns
        self.framework_patterns = {
            "django": [r"\bdjango\b", r"\bmodels\.Model\b", r"\bviews\.py\b"],
            "flask": [r"\bflask\b", r"\b@app\.route\b", r"\bBlueprint\b"],
            "react": [r"\breact\b", r"\buseState\b", r"\buseEffect\b", r"\bjsx\b"],
            "vue": [r"\bvue\b", r"\.vue\b", r"\bv-if\b", r"\bv-for\b"],
            "angular": [r"\bangular\b", r"\b@Component\b", r"\bngOnInit\b"],
            "spring": [r"\bspring\b", r"\b@RestController\b", r"\b@Service\b"],
            "express": [r"\bexpress\b", r"\bapp\.get\b", r"\bapp\.post\b"],
            "fastapi": [r"\bfastapi\b", r"\b@app\.get\b", r"\bpydantic\b"],
            "tensorflow": [r"\btensorflow\b", r"\btf\.\b", r"\bkeras\b"],
            "pytorch": [r"\bpytorch\b", r"\btorch\b", r"\bnn\.Module\b"],
        }

        # Complexity indicators
        self.complexity_indicators = {
            "algorithms": [r"\b(algorithm|recursion|dynamic programming|graph|tree)\b"],
            "concurrency": [r"\b(async|await|thread|concurrent|parallel|mutex|lock)\b"],
            "distributed": [r"\b(distributed|microservice|kafka|rabbitmq|redis)\b"],
            "ml": [r"\b(machine learning|neural network|model|training|dataset)\b"],
            "security": [
                r"\b(security|encryption|authentication|authorization|oauth)\b"
            ],
        }

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None

    async def analyze(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskCharacteristics:
        """
        Analyze a task to determine its characteristics.

        Args:
            task: The task description
            context: Additional context

        Returns:
            TaskCharacteristics object
        """
        task_lower = task.lower()

        # Determine task type
        task_type, type_confidence = self._determine_task_type(task_lower)
        # Determine complexity
        complexity = self._determine_complexity(task, context)
        # Detect languages
        languages = self._detect_languages(task_lower, context)
        # Detect frameworks
        frameworks = self._detect_frameworks(task_lower, context)
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(task)
        # Extract keywords
        keywords = self._extract_keywords(task_lower)
        # Analyze requirements
        has_tests = self._check_pattern(task_lower, [r"\b(test|testing|coverage)\b"])
        requires_external_apis = self._check_pattern(
            task_lower, [r"\b(API|REST|GraphQL|webhook)\b"]
        )
        requires_database = self._check_pattern(
            task_lower, [r"\b(database|DB|SQL|NoSQL|MongoDB|PostgreSQL)\b"]
        )
        requires_ui = self._check_pattern(
            task_lower, [r"\b(UI|frontend|interface|component|widget)\b"]
        )
        security_sensitive = self._check_pattern(
            task_lower, self.complexity_indicators["security"]
        )
        performance_critical = self._check_pattern(
            task_lower, [r"\b(performance|speed|latency|optimization)\b"]
        )

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            type_confidence,
            bool(languages),
            bool(frameworks),
            len(keywords),
        )

        return TaskCharacteristics(
            task_type=task_type,
            complexity=complexity,
            languages=languages,
            frameworks=frameworks,
            estimated_tokens=estimated_tokens,
            has_tests=has_tests,
            requires_external_apis=requires_external_apis,
            requires_database=requires_database,
            requires_ui=requires_ui,
            security_sensitive=security_sensitive,
            performance_critical=performance_critical,
            keywords=keywords,
            confidence=confidence,
            metadata=context or {},
        )

    def _determine_task_type(self, task_lower: str) -> Tuple[TaskType, float]:
        """Determine the task type based on patterns."""
        scores = {}

        for task_type, patterns in self.task_type_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, task_lower))
            if score > 0:
                scores[task_type] = score

        if not scores:
            return TaskType.UNKNOWN, 0.0

        # Get type with highest score
        best_type = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_type] / total_score if total_score > 0 else 0.0

        return best_type, confidence

    def _determine_complexity(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> TaskComplexity:
        """Determine task complexity."""
        # Check context for explicit complexity hints
        if context:
            if "lines_of_code" in context:
                loc = context["lines_of_code"]
                if loc < 10:
                    return TaskComplexity.TRIVIAL
                elif loc < 50:
                    return TaskComplexity.SIMPLE
                elif loc < 200:
                    return TaskComplexity.MODERATE
                elif loc < 500:
                    return TaskComplexity.COMPLEX
                else:
                    return TaskComplexity.VERY_COMPLEX

        # Analyze task description
        task_lower = task.lower()

        # Count complexity indicators
        complexity_score = 0
        for category, patterns in self.complexity_indicators.items():
            if self._check_pattern(task_lower, patterns):
                complexity_score += 1

        # Check for multiple requirements
        if task.count("\n") > 5 or len(task.split()) > 200:
            complexity_score += 1

        # Map score to complexity
        if complexity_score == 0:
            if len(task.split()) < 20:
                return TaskComplexity.TRIVIAL
            else:
                return TaskComplexity.SIMPLE
        elif complexity_score == 1:
            return TaskComplexity.MODERATE
        elif complexity_score == 2:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX

    def _detect_languages(
        self,
        task_lower: str,
        context: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """Detect programming languages mentioned in the task."""
        languages = set()

        # Check context for explicit language
        if context and "language" in context:
            languages.add(context["language"].lower())
        # Pattern matching
        for language, patterns in self.language_patterns.items():
            if self._check_pattern(task_lower, patterns):
                languages.add(language)
        return languages

    def _detect_frameworks(
        self,
        task_lower: str,
        context: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """Detect frameworks mentioned in the task."""
        frameworks = set()

        # Check context
        if context and "frameworks" in context:
            if isinstance(context["frameworks"], list):
                frameworks.update(f.lower() for f in context["frameworks"])
            else:
                frameworks.add(context["frameworks"].lower())
        # Pattern matching
        for framework, patterns in self.framework_patterns.items():
            if self._check_pattern(task_lower, patterns):
                frameworks.add(framework)
        return frameworks

    def _estimate_tokens(self, task: str) -> int:
        """Estimate the number of tokens in the task."""
        if self.encoding:
            return len(self.encoding.encode(task))
        else:
            # Rough estimate: 4 characters per token
            return len(task) // 4

    def _extract_keywords(self, task_lower: str) -> List[str]:
        """Extract important keywords from the task."""
        # Common programming keywords to look for
        keyword_patterns = [
            r"\b(function|class|method|variable|constant)\b",
            r"\b(array|list|dict|map|set|queue|stack)\b",
            r"\b(loop|iteration|recursion|condition)\b",
            r"\b(input|output|file|stream|buffer)\b",
            r"\b(error|exception|validation|sanitization)\b",
            r"\b(async|sync|callback|promise|observable)\b",
            r"\b(REST|GraphQL|WebSocket|gRPC)\b",
            r"\b(JSON|XML|YAML|CSV)\b",
        ]

        keywords = []
        for pattern in keyword_patterns:
            matches = re.findall(pattern, task_lower)
            keywords.extend(matches)
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    def _check_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if any pattern matches the text."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _calculate_confidence(
        self,
        type_confidence: float,
        has_languages: bool,
        has_frameworks: bool,
        keyword_count: int,
    ) -> float:
        """Calculate overall analysis confidence."""
        confidence = type_confidence * 0.4

        if has_languages:
            confidence += 0.2
        if has_frameworks:
            confidence += 0.2
        if keyword_count > 3:
            confidence += 0.2

        return min(confidence, 1.0)
