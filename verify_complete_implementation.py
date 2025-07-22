#!/usr/bin/env python3
"""
Complete Implementation Verification Script

Tests all major components of the LlamaAgent system to ensure
complete functionality and integration.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verification results
verification_results: Dict[str, Dict[str, Any]] = {}


def verify_import(module_name: str, component_name: Optional[str] = None) -> bool:
    """Verify that a module can be imported successfully"""
    try:
        module = importlib.import_module(module_name)
        if component_name:
            getattr(module, component_name)
        logger.info(f"PASS Successfully imported {module_name}" + (f".{component_name}" if component_name else ""))
        return True
    except Exception as e:
        logger.error(f"FAIL Failed to import {module_name}" + (f".{component_name}" if component_name else "") + f": {e}")
        return False


def verify_file_exists(file_path: str) -> bool:
    """Verify that a file exists"""
    path = Path(file_path)
    if path.exists():
        logger.info(f"PASS File exists: {file_path}")
        return True
    else:
        logger.error(f"FAIL File missing: {file_path}")
        return False


async def verify_spre_generator() -> Dict[str, Any]:
    """Verify SPREGenerator functionality"""
    logger.info("Verifying SPREGenerator...")
    results: Dict[str, list] = {"passed": [], "failed": []}
    
    try:
        # Test imports
        from src.llamaagent.data_generation.spre import (
            SPREGenerator, SPREItem, DataType, ValidationStatus
        )
        results["passed"].append("SPREGenerator imports")
        
        # Test dataclass creation - using proper content structure
        item = SPREItem(
            id="test-1",
            content={"text": "Test content"},  # Proper Dict[str, Any] format
            data_type=DataType.TEXT,
            validation_status=ValidationStatus.VALID,
            quality_score=0.8
        )
        results["passed"].append("SPREItem creation")
        logger.debug(f"Created test item: {item.id}")
        
        # Test generator initialization
        generator = SPREGenerator()
        results["passed"].append("SPREGenerator initialization")
        
        # Test dataset generation (mock mode)
        try:
            dataset = await generator.generate_dataset(
                name="test_dataset",
                count=2,
                data_type=DataType.TEXT,
                topic="test"
            )
            
            if dataset and hasattr(dataset, 'items') and dataset.items:
                results["passed"].append("Dataset generation")
            else:
                results["failed"].append("Dataset generation - no items")
        except Exception as gen_error:
            results["failed"].append(f"Dataset generation error: {gen_error}")
            
    except Exception as e:
        results["failed"].append(f"SPREGenerator error: {e}")
        logger.error(f"SPREGenerator verification failed: {e}")
    
    return results


async def verify_api_components() -> Dict[str, Any]:
    """Verify FastAPI components"""
    logger.info("Verifying API components...")
    results = {"passed": [], "failed": []}
    
    try:
        # Test FastAPI app import
        from src.llamaagent.api.complete_api import app
        results["passed"].append("FastAPI app import")
        
        # Test Pydantic models
        from src.llamaagent.api.complete_api import (
            SPREGenerationRequest, AgentCreationRequest, HealthResponse
        )
        results["passed"].append("Pydantic models import")
        
        # Test model creation
        request = SPREGenerationRequest(
            name="test",
            count=5,
            data_type="text"
        )
        results["passed"].append("Request model creation")
        
    except Exception as e:
        results["failed"].append(f"API components error: {e}")
        logger.error(f"API verification failed: {e}")
    
    return results


async def verify_openai_integration() -> Dict[str, Any]:
    """Verify OpenAI integration"""
    logger.info("Verifying OpenAI integration...")
    results = {"passed": [], "failed": []}
    
    try:
        # Test OpenAI integration imports
        from src.llamaagent.integration.openai_agents import (
            OpenAIAgentAdapter, OpenAIAgentsIntegration, OpenAIIntegrationConfig
        )
        results["passed"].append("OpenAI integration imports")
        
        # Test configuration
        config = OpenAIIntegrationConfig(
            api_key="test-key",
            model="gpt-4o-mini"
        )
        results["passed"].append("OpenAI config creation")
        
        # Test adapter initialization
        adapter = OpenAIAgentAdapter(config)
        results["passed"].append("OpenAI adapter initialization")
        
    except Exception as e:
        results["failed"].append(f"OpenAI integration error: {e}")
        logger.error(f"OpenAI integration verification failed: {e}")
    
    return results


async def verify_agent_system() -> Dict[str, Any]:
    """Verify agent system"""
    logger.info("Verifying agent system...")
    results = {"passed": [], "failed": []}
    
    try:
        # Test agent imports
        from src.llamaagent.agents.base import AgentConfig, AgentRole
        results["passed"].append("Agent base imports")
        
        # Test config creation
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.GENERALIST,
            llm_provider="mock"
        )
        results["passed"].append("AgentConfig creation")
        
        # Test agent creation if ReactAgent is available
        try:
            from src.llamaagent.agents.react import ReactAgent
            agent = ReactAgent(config)
            results["passed"].append("ReactAgent creation")
        except ImportError:
            results["failed"].append("ReactAgent not available")
            
    except Exception as e:
        results["failed"].append(f"Agent system error: {e}")
        logger.error(f"Agent system verification failed: {e}")
    
    return results


async def verify_tools_system() -> Dict[str, Any]:
    """Verify tools system"""
    logger.info("Verifying tools system...")
    results = {"passed": [], "failed": []}
    
    try:
        # Test tools imports
        from src.llamaagent.tools import ToolRegistry
        results["passed"].append("Tools imports")
        
        # Test registry
        registry = ToolRegistry()
        results["passed"].append("ToolRegistry creation")
        
        # Test base tool
        from src.llamaagent.tools.base import BaseTool
        results["passed"].append("BaseTool import")
        
    except Exception as e:
        results["failed"].append(f"Tools system error: {e}")
        logger.error(f"Tools system verification failed: {e}")
    
    return results


def verify_docker_files() -> Dict[str, Any]:
    """Verify Docker configuration files"""
    logger.info("Verifying Docker files...")
    results = {"passed": [], "failed": []}
    
    docker_files = [
        "Dockerfile.complete",
        "docker-compose.complete.yml",
        "docker/entrypoint.sh"
    ]
    
    for file_path in docker_files:
        if verify_file_exists(file_path):
            results["passed"].append(f"Docker file: {file_path}")
        else:
            results["failed"].append(f"Missing Docker file: {file_path}")
    
    return results


def verify_test_files() -> Dict[str, Any]:
    """Verify test files"""
    logger.info("Verifying test files...")
    results = {"passed": [], "failed": []}
    
    test_files = [
        "tests/test_comprehensive_functionality.py",
        "tests/conftest.py"
    ]
    
    for file_path in test_files:
        if verify_file_exists(file_path):
            results["passed"].append(f"Test file: {file_path}")
        else:
            results["failed"].append(f"Missing test file: {file_path}")
    
    return results


def verify_documentation() -> Dict[str, Any]:
    """Verify documentation files"""
    logger.info("Verifying documentation...")
    results = {"passed": [], "failed": []}
    
    doc_files = [
        "README_COMPREHENSIVE.md",
        "COMPLETE_IMPLEMENTATION_REPORT.md",
        "build_comprehensive.py"
    ]
    
    for file_path in doc_files:
        if verify_file_exists(file_path):
            results["passed"].append(f"Documentation: {file_path}")
        else:
            results["failed"].append(f"Missing documentation: {file_path}")
    
    return results


async def run_verification_suite() -> Tuple[int, int]:
    """Run complete verification suite"""
    logger.info("Starting LlamaAgent Complete Implementation Verification...")
    logger.info("=" * 60)
    
    verification_tasks = [
        ("SPREGenerator", verify_spre_generator()),
        ("API Components", verify_api_components()),
        ("OpenAI Integration", verify_openai_integration()),
        ("Agent System", verify_agent_system()),
        ("Tools System", verify_tools_system()),
        ("Docker Files", verify_docker_files()),
        ("Test Files", verify_test_files()),
        ("Documentation", verify_documentation())
    ]
    
    total_passed = 0
    total_failed = 0
    
    for component_name, verification_task in verification_tasks:
        logger.info(f"\n--- Verifying {component_name} ---")
        
        if asyncio.iscoroutine(verification_task):
            results = await verification_task
        else:
            results = verification_task
            
        verification_results[component_name] = results
        
        passed = len(results["passed"])
        failed = len(results["failed"])
        total_passed += passed
        total_failed += failed
        
        logger.info(f"{component_name}: {passed} passed, {failed} failed")
        
        if results["failed"]:
            for failure in results["failed"]:
                logger.error(f"  FAIL {failure}")
    
    return total_passed, total_failed


def generate_verification_report(total_passed: int, total_failed: int) -> None:
    """Generate verification report"""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    total_checks = total_passed + total_failed
    success_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    logger.info(f"Total Checks: {total_checks}")
    logger.info(f"Passed: {total_passed}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    # Component breakdown
    logger.info("\nComponent Breakdown:")
    for component, results in verification_results.items():
        passed = len(results["passed"])
        failed = len(results["failed"])
        total = passed + failed
        rate = (passed / total * 100) if total > 0 else 0
        logger.info(f"  {component}: {passed}/{total} ({rate:.1f}%)")
    
    # Overall status
    if total_failed == 0:
        logger.info("\nSUCCESS ALL VERIFICATIONS PASSED!")
        logger.info("LlamaAgent Complete Implementation is ready!")
    else:
        logger.info(f"\nWARNING:  {total_failed} verifications failed")
        logger.info("Please check the failed components above")
    
    logger.info("=" * 60)


async def main():
    """Main verification function"""
    start_time = time.time()
    
    try:
        total_passed, total_failed = await run_verification_suite()
        generate_verification_report(total_passed, total_failed)
        
        execution_time = time.time() - start_time
        logger.info(f"\nVerification completed in {execution_time:.2f} seconds")
        
        # Exit with appropriate code
        sys.exit(0 if total_failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Verification suite failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 