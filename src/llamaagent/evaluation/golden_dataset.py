"""
Golden Dataset Management System

Provides comprehensive dataset creation, validation, and management capabilities
for evaluation benchmarks with quality assurance and synthetic data generation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)
class DatasetType(str, Enum):
    """Types of evaluation datasets"""
    
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    MULTIMODAL = "multimodal"
    CODING = "coding"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    CREATIVE = "creative"
    SAFETY = "safety"
    INSTRUCTION_FOLLOWING = "instruction_following"


class DifficultyLevel(str, Enum):
    """Difficulty levels for evaluation tasks"""
    
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class DatasetSample:
    """Single sample in a golden dataset"""
    
    id: str
    input: Union[str, Dict[str, Any]]
    expected_output: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    quality_score: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSample":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            input=data["input"],
            expected_output=data["expected_output"],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            difficulty=DifficultyLevel(data.get("difficulty", DifficultyLevel.MEDIUM.value)),
            quality_score=data.get("quality_score", 1.0),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "input": self.input,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
            "tags": self.tags,
            "difficulty": self.difficulty.value,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DatasetMetrics:
    """Metrics for dataset quality and characteristics"""
    
    total_samples: int
    difficulty_distribution: Dict[str, int]
    tag_distribution: Dict[str, int]
    quality_score_stats: Dict[str, float]
    diversity_score: float
    completeness_score: float
    consistency_score: float
    overall_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_samples": self.total_samples,
            "difficulty_distribution": self.difficulty_distribution,
            "tag_distribution": self.tag_distribution,
            "quality_score_stats": self.quality_score_stats,
            "diversity_score": self.diversity_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "overall_quality": self.overall_quality
        }


@dataclass
class DataQualityReport:
    """Comprehensive dataset quality report"""
    
    dataset_name: str
    metrics: DatasetMetrics
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dataset_name": self.dataset_name,
            "metrics": self.metrics.to_dict(),
            "issues": self.issues,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat()
        }


class GoldenDatasetManager:
    """Manages creation, validation, and storage of golden datasets"""
    
    def __init__(self, storage_path: str = "./datasets"):
        """Initialize dataset manager"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        # In-memory dataset storage
        self.datasets: Dict[str, List[DatasetSample]] = {}
        self.dataset_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def create_dataset(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        description: str = "",
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        config: Optional[Dict[str, Any]] = None)
    ) -> None:
        """Create a new dataset"""
        config = config or {}
        
        self.datasets[dataset_name] = []
        self.dataset_metadata[dataset_name] = {
            "name": dataset_name,
            "type": dataset_type.value,
            "description": description,
            "difficulty": difficulty.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config
        }
        
        logger.info(f"Created dataset: {dataset_name}")
    async def add_sample(
        self,
        dataset_name: str,
        input_data: Union[str, Dict[str, Any]],
        expected_output: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        difficulty: Optional[DifficultyLevel] = None,
        quality_score: float = 1.0
    ) -> str:
        """Add a sample to the dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        # Generate unique ID
        sample_id = self._generate_sample_id(dataset_name, len(self.datasets[dataset_name])
        sample = DatasetSample(
            id=sample_id,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata or {},
            tags=tags or [],
            difficulty=difficulty or DifficultyLevel.MEDIUM,
            quality_score=quality_score
        )
        self.datasets[dataset_name].append(sample)
        logger.debug(f"Added sample {sample_id} to dataset {dataset_name}")
        return sample_id
    
    def _generate_sample_id(self, dataset_name: str, index: int) -> str:
        """Generate unique sample ID"""
        return f"{dataset_name}_{index:06d}_{int(datetime.now().timestamp()}"
    
    async def load_dataset(self, dataset_name: str) -> None:
        """Load dataset from storage"""
        dataset_path = self.storage_path / f"{dataset_name}.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Load samples
        samples = []
        for sample_data in data.get("samples", []):
            sample = DatasetSample.from_dict(sample_data)
            samples.append(sample)
        self.datasets[dataset_name] = samples
        self.dataset_metadata[dataset_name] = data.get("metadata", {})
        logger.info(f"Loaded dataset {dataset_name} with {len(samples)} samples")
    
    async def save_dataset(self, dataset_name: str) -> None:
        """Save dataset to storage"""
        dataset_path = self.storage_path / f"{dataset_name}.json"
        
        export_data = await self.export_dataset(dataset_name)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Saved dataset {dataset_name} to {dataset_path}")
    async def export_dataset(
        self,
        dataset_name: str,
        format: str = "json",
        filter_config: Optional[Dict[str, Any]] = None)
    ) -> Dict[str, Any]:
        """Export dataset in specified format"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        samples = self.datasets[dataset_name]
        
        # Apply filters if specified
        if filter_config:
            samples = self._filter_samples(samples, filter_config)
        export_data = {
            "metadata": self.dataset_metadata.get(dataset_name, {}),
            "samples": [sample.to_dict() for sample in samples],
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(samples)
        }
        
        return export_data
    
    def _filter_samples(
        self, samples: List[DatasetSample], filter_config: Dict[str, Any]
    ) -> List[DatasetSample]:
        """Filter samples based on configuration"""
        filtered = samples
        
        # Filter by difficulty
        if "difficulty" in filter_config:
            target_difficulty = filter_config["difficulty"]
            filtered = [s for s in filtered if s.difficulty.value == target_difficulty]
        
        # Filter by tags
        if "tags" in filter_config:
            required_tags = set(filter_config["tags"])
            filtered = [s for s in filtered if required_tags.issubset(set(s.tags)]
        
        # Filter by quality score
        if "min_quality" in filter_config:
            min_quality = filter_config["min_quality"]
            filtered = [s for s in filtered if s.quality_score >= min_quality]
        
        return filtered
    
    async def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        samples = self.datasets[dataset_name]
        metrics = await self._calculate_dataset_metrics(samples)
        return {
            "name": dataset_name,
            "total_samples": len(samples),
            "metrics": metrics.to_dict(),
            "metadata": self.dataset_metadata.get(dataset_name, {})
        }
    
    async def _calculate_dataset_metrics(self, samples: List[DatasetSample]) -> DatasetMetrics:
        """Calculate comprehensive dataset metrics"""
        if not samples:
            return DatasetMetrics(
                total_samples=0,
                difficulty_distribution={},
                tag_distribution={},
                quality_score_stats={},
                diversity_score=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                overall_quality=0.0
            )
        # Basic counts
        total_samples = len(samples)
        # Difficulty distribution
        difficulty_dist = defaultdict(int)
        for sample in samples:
            difficulty_dist[sample.difficulty.value] += 1
        
        # Tag distribution
        tag_dist = defaultdict(int)
        for sample in samples:
            for tag in sample.tags:
                tag_dist[tag] += 1
        
        # Quality score statistics
        quality_scores = [sample.quality_score for sample in samples]
        quality_stats = {
            "mean": np.mean(quality_scores),
            "std": np.std(quality_scores),
            "min": np.min(quality_scores),
            "max": np.max(quality_scores),
            "median": np.median(quality_scores)
        }
        
        # Calculate diversity score
        diversity_score = await self._calculate_diversity_score(samples)
        # Calculate completeness score
        completeness_score = await self._calculate_completeness_score(samples)
        # Calculate consistency score
        consistency_score = await self._calculate_consistency_score(samples)
        # Overall quality
        overall_quality = (diversity_score + completeness_score + consistency_score) / 3
        
        return DatasetMetrics(
            total_samples=total_samples,
            difficulty_distribution=dict(difficulty_dist),
            tag_distribution=dict(tag_dist),
            quality_score_stats=quality_stats,
            diversity_score=diversity_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality
        )
    
    async def _calculate_diversity_score(self, samples: List[DatasetSample]) -> float:
        """Calculate diversity score based on input/output variety"""
        if len(samples) < 2:
            return 1.0
        
        # Simple diversity metric based on unique inputs/outputs
        unique_inputs = set()
        unique_outputs = set()
        
        for sample in samples:
            input_str = str(sample.input) if isinstance(sample.input, (str, int, float) else str(sample.input)
            output_str = str(sample.expected_output) if isinstance(sample.expected_output, (str, int, float) else str(sample.expected_output)
            unique_inputs.add(input_str)
            unique_outputs.add(output_str)
        input_diversity = len(unique_inputs) / len(samples)
        output_diversity = len(unique_outputs) / len(samples)
        return (input_diversity + output_diversity) / 2
    
    async def _calculate_completeness_score(self, samples: List[DatasetSample]) -> float:
        """Calculate completeness score based on required fields"""
        if not samples:
            return 0.0
        
        complete_samples = 0
        for sample in samples:
            # Check if sample has all required fields
            has_input = sample.input is not None and str(sample.input).strip() != ""
            has_output = sample.expected_output is not None and str(sample.expected_output).strip() != ""
            has_metadata = bool(sample.metadata)
            if has_input and has_output and has_metadata:
                complete_samples += 1
        
        return complete_samples / len(samples)
    async def _calculate_consistency_score(self, samples: List[DatasetSample]) -> float:
        """Calculate consistency score based on format and structure"""
        if len(samples) < 2:
            return 1.0
        
        # Check input type consistency
        input_types = [type(sample.input).__name__ for sample in samples]
        type_consistency = len(set(input_types) / len(input_types)
        # Check structure consistency for dict inputs
        dict_inputs = [sample.input for sample in samples if isinstance(sample.input, dict)]
        structure_consistency = 1.0
        
        if dict_inputs:
            key_sets = [set(d.keys() for d in dict_inputs]
            if key_sets:
                common_keys = set.intersection(*key_sets)
                avg_keys = np.mean([len(keys) for keys in key_sets])
                structure_consistency = len(common_keys) / max(1, avg_keys)
        return (type_consistency + structure_consistency) / 2
    
    async def validate_dataset(self, dataset_name: str) -> DataQualityReport:
        """Comprehensive dataset validation"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        samples = self.datasets[dataset_name]
        issues: List[Dict[str, Any]] = []
        recommendations: List[str] = []
        
        # Calculate metrics
        metrics = await self._calculate_dataset_metrics(samples)
        # Quality checks
        if metrics.total_samples < 100:
            issues.append({
                "type": "size",
                "severity": "warning",
                "message": f"Dataset has only {metrics.total_samples} samples"
            })
            recommendations.append()
                "Add more samples to improve statistical significance"
            )
        
        if metrics.diversity_score < 0.7:
            issues.append({
                "type": "diversity",
                "severity": "warning",
                "message": f"Low diversity score: {metrics.diversity_score:.2f}"
            })
            recommendations.append("Increase input/output diversity across samples")
        if metrics.consistency_score < 0.8:
            issues.append({
                "type": "consistency",
                "severity": "warning",
                "message": f"Low consistency score: {metrics.consistency_score:.2f}"
            })
            recommendations.append("Review samples for annotation consistency")
        # Difficulty distribution check
        diff_counts = metrics.difficulty_distribution
        if len(diff_counts) < 3:
            issues.append({
                "type": "difficulty",
                "severity": "info",
                "message": "Limited difficulty level distribution"
            })
            recommendations.append("Add samples across different difficulty levels")
        return DataQualityReport(
            dataset_name=dataset_name,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    async def generate_synthetic_samples(
        self,
        dataset_name: str,
        count: int,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate synthetic samples for the dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        config = generation_config or {}
        samples = self.datasets[dataset_name]
        
        if not samples:
            logger.warning(f"No existing samples in {dataset_name} for pattern analysis")
            return []
        
        # Analyze existing patterns
        patterns = await self._analyze_dataset_patterns(samples)
        generated_ids: List[str] = []
        
        for i in range(count):
            try:
                # Generate a new sample based on patterns
                new_sample = await self._generate_sample_from_patterns(
                    dataset_name, patterns, config
                )
                # Add to dataset
                sample_id = await self.add_sample(
                    dataset_name,
                    new_sample["input"],
                    new_sample["expected_output"],
                    new_sample.get("metadata", {}),
                    new_sample.get("tags", []),
                    DifficultyLevel(new_sample.get("difficulty", DifficultyLevel.MEDIUM.value)
                )
                
                generated_ids.append(sample_id)
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                continue
        
        logger.info(
            f"Generated {len(generated_ids)} synthetic samples for {dataset_name}"
        )
        return generated_ids
    
    async def _analyze_dataset_patterns(self, samples: List[DatasetSample]) -> Dict[str, Any]:
        """Analyze patterns in existing dataset for synthetic generation"""
        patterns = {
            "input_patterns": [],
            "output_patterns": [],
            "difficulty_patterns": defaultdict(int),
            "tag_patterns": defaultdict(int),
            "structure_patterns": {}
        }
        
        for sample in samples:
            # Analyze input patterns
            if isinstance(sample.input, str):
                patterns["input_patterns"].append({
                    "type": "text",
                    "length": len(sample.input),
                    "words": len(sample.input.split()
                })
            elif isinstance(sample.input, dict):
                patterns["input_patterns"].append({
                    "type": "dict",
                    "keys": list(sample.input.keys(),
                    "structure": type(sample.input).__name__
                })
            
            # Analyze output patterns
            if isinstance(sample.expected_output, str):
                patterns["output_patterns"].append({
                    "type": "text",
                    "length": len(sample.expected_output),
                    "words": len(sample.expected_output.split()
                })
            elif isinstance(sample.expected_output, dict):
                patterns["output_patterns"].append({
                    "type": "dict",
                    "keys": list(sample.expected_output.keys(),
                    "structure": type(sample.expected_output).__name__
                })
            
            # Difficulty and tag patterns
            patterns["difficulty_patterns"][sample.difficulty.value] += 1
            for tag in sample.tags:
                patterns["tag_patterns"][tag] += 1
        
        return patterns
    
    async def _generate_sample_from_patterns(
        self,
        dataset_name: str,
        patterns: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a new sample based on existing patterns"""
        # This would typically use an LLM to generate new samples
        # For now, we'll create a simplified template-based approach
        
        # Select difficulty level
        difficulty_levels = list(patterns["difficulty_patterns"].keys()
        if not difficulty_levels:
            difficulty_levels = ["medium"]
        
        difficulty = random.choice(difficulty_levels)
        # Select tags
        available_tags = list(patterns["tag_patterns"].keys())
        selected_tags = (
            random.sample(available_tags, min(3, len(available_tags)))
            if available_tags
            else []
        )
        
        # Generate sample (simplified - would use LLM in practice)
        sample = {
            "input": f"Sample input for {dataset_name} with difficulty {difficulty}",
            "expected_output": f"Expected output corresponding to the input for {difficulty} level task",
            "metadata": {
                "generated": True,
                "source": "synthetic",
                "generation_config": config
            },
            "tags": selected_tags,
            "difficulty": difficulty,
            "quality_score": 0.8  # Slightly lower for synthetic samples
        }
        
        return sample