"""
Premium API endpoints for llamaagent.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class PremiumRequest(BaseModel):
    """Base request model for premium endpoints."""

    subscription_key: str = Field(..., description="Premium subscription key")
    user_id: str = Field(..., description="User identifier")
    priority: int = Field(1, description="Request priority (1-10)")


class APIResponse(BaseModel):
    """Standard API response model."""

    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    message: str = Field(default="", description="Response message")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Response metadata"
    )


class DatasetCreateRequest(BaseModel):
    """Request to create a new golden dataset."""

    name: str = Field(..., description="Dataset name")
    description: str = Field(default="", description="Dataset description")
    tags: Optional[List[str]] = Field(default=None, description="Dataset tags")


class DatasetSampleRequest(BaseModel):
    """Request to add a sample to dataset."""

    dataset_name: str = Field(..., description="Target dataset name")
    input_data: Any = Field(..., description="Input data for the sample")
    expected_output: Any = Field(..., description="Expected output for the sample")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Sample metadata"
    )


class BenchmarkCreateRequest(BaseModel):
    """Request to create benchmark from dataset."""

    benchmark_id: str = Field(..., description="Unique benchmark ID")
    dataset_name: str = Field(..., description="Source dataset name")
    description: str = Field(default="", description="Benchmark description")


class BenchmarkRunRequest(BaseModel):
    """Request to run a benchmark."""

    benchmark_id: str = Field(..., description="Benchmark to run")
    model_name: str = Field(..., description="Model to test")
    include_examples: bool = Field(default=True, description="Include examples")
    sample_limit: Optional[int] = Field(
        default=None, description="Limit number of samples"
    )


def verify_subscription(subscription_key: str) -> Dict[str, Any]:
    """Verify subscription key (mock implementation)."""
    return {
        "valid": True,
        "plan": "premium",
        "features": [
            "advanced_models",
            "enhanced_code_gen",
            "data_analysis",
            "premium_chat",
        ],
    }


def get_subscription_info(request: PremiumRequest) -> Dict[str, Any]:
    """Get subscription information."""
    return verify_subscription(request.subscription_key)


@router.post("/datasets/create", response_model=APIResponse)
async def create_dataset(request: DatasetCreateRequest):
    """Create a new golden dataset."""
    try:
        # Mock implementation
        return APIResponse(
            success=True,
            message=f"Dataset '{request.name}' created successfully",
            data={"dataset_id": request.name},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/add-sample", response_model=APIResponse)
async def add_dataset_sample(request: DatasetSampleRequest):
    """Add a sample to an existing dataset."""
    try:
        # Mock implementation
        return APIResponse(
            success=True,
            message=f"Sample added to dataset '{request.dataset_name}'",
            data={"sample_id": "sample_123"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks/create-from-dataset", response_model=APIResponse)
async def create_benchmark_from_dataset(request: BenchmarkCreateRequest):
    """Create a benchmark from an existing golden dataset."""
    try:
        # Mock implementation
        return APIResponse(
            success=True,
            message=f"Benchmark '{request.benchmark_id}' created from dataset '{request.dataset_name}'",
            data={"benchmark_id": request.benchmark_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks/run", response_model=APIResponse)
async def run_benchmark(request: BenchmarkRunRequest):
    """Run a benchmark."""
    try:
        # Mock implementation
        return APIResponse(
            success=True,
            message=f"Running benchmark '{request.benchmark_id}' for model '{request.model_name}' in background",
            data={"task_id": "task_123"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/premium-features", response_model=APIResponse)
async def premium_features_health():
    """Health check for all premium features."""
    try:
        health_status = {
            "dataset_manager": "healthy",
            "benchmark_engine": "healthy",
            "knowledge_generator": "healthy",
            "model_comparison": "healthy",
        }
        return APIResponse(
            success=True,
            message="Premium features health check completed",
            data=health_status,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
