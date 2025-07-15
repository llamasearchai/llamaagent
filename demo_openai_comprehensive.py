"""
Comprehensive OpenAI Integration Demonstration

This script demonstrates all OpenAI model types and APIs integrated in llamaagent:
- Reasoning models (o-series)
- Flagship chat models
- Cost-optimized models
- Image generation models
- Text-to-speech models
- Transcription models
- Embeddings models
- Moderation models
- FastAPI endpoints
- Budget tracking
- Error handling

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our comprehensive OpenAI integration
from src.llamaagent.integration.openai_comprehensive import (
    OpenAIComprehensiveConfig,
    create_comprehensive_openai_integration,
)
from src.llamaagent.tools.openai_tools import create_all_openai_tools

# Import FastAPI client for endpoint testing


class OpenAIComprehensiveDemo:
    """Comprehensive demonstration of OpenAI integration."""
    
    def __init__(self, budget_limit: float = 50.0):
        self.budget_limit = budget_limit
        self.integration = None
        self.tools = {}
        self.results = []
        self.start_time = datetime.now(timezone.utc)
        
        # Check for API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. Some demos will be skipped.")
    
    async def initialize(self):
        """Initialize the comprehensive integration."""
        logger.info("Initializing OpenAI Comprehensive Integration")
        
        if not self.api_key:
            logger.error("FAIL OpenAI API key required")
            return False
        
        try:
            # Create comprehensive integration
            config = OpenAIComprehensiveConfig(
                api_key=self.api_key,
                budget_limit=self.budget_limit,
                enable_usage_tracking=True,
                enable_cost_warnings=True
            )
            
            self.integration = create_comprehensive_openai_integration(
                api_key=self.api_key,
                budget_limit=self.budget_limit
            )
            
            # Create all tools
            all_tools = create_all_openai_tools(self.integration)
            for tool in all_tools:
                self.tools[tool.name] = tool
            
            logger.info(f"PASS Integration initialized with ${self.budget_limit} budget")
            logger.info(f"🛠️  Available tools: {list(self.tools.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"FAIL Initialization failed: {e}")
            return False
    
    async def demo_health_check(self):
        """Demo health check functionality."""
        logger.info("\n🏥 Demo: Health Check")
        logger.info("-" * 40)
        
        try:
            health = await self.integration.health_check()
            
            logger.info(f"API Accessible: {health['api_accessible']}")
            logger.info(f"Available Models: {health.get('available_models_count', 'Unknown')}")
            
            # Check specific model types
            model_availability = health.get("model_types_available", {})
            for model_type, info in model_availability.items():
                availability_ratio = info.get("availability_ratio", 0)
                logger.info(f"{model_type}: {availability_ratio:.1%} available")
            
            self.results.append({
                "demo": "health_check",
                "success": health["api_accessible"],
                "data": health
            })
            
        except Exception as e:
            logger.error(f"FAIL Health check failed: {e}")
            self.results.append({
                "demo": "health_check",
                "success": False,
                "error": str(e)
            })
    
    async def demo_reasoning_models(self):
        """Demo reasoning models (o-series)."""
        logger.info("\nINTELLIGENCE Demo: Reasoning Models (o-series)")
        logger.info("-" * 40)
        
        try:
            reasoning_tool = self.tools["openai_reasoning"]
            
            problems = [
                "Solve this step by step: If a train travels 60 mph for 2.5 hours, how far does it go?",
                "What is the sum of the first 10 prime numbers?",
                "If I have 100 books and I read 3 books per week, how many weeks to read all books?"
            ]
            
            for i, problem in enumerate(problems, 1):
                logger.info(f"Problem {i}: {problem}")
                
                result = await reasoning_tool.aexecute(
                    problem=problem,
                    model="o3-mini",  # Use most cost-effective reasoning model
                    temperature=0.1
                )
                
                if result["success"]:
                    logger.info(f"PASS Solution: {result['response'][:200]}...")
                    logger.info(f"Tokens used: {result.get('usage', {}).get('total_tokens', 'Unknown')}")
                else:
                    logger.error(f"FAIL Failed: {result.get('error')}")
                
                self.results.append({
                    "demo": f"reasoning_problem_{i}",
                    "success": result["success"],
                    "data": result
                })
                
                # Brief pause between requests
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"FAIL Reasoning demo failed: {e}")
            self.results.append({
                "demo": "reasoning_models",
                "success": False,
                "error": str(e)
            })
    
    async def demo_chat_models(self):
        """Demo flagship and cost-optimized chat models."""
        logger.info("\n💬 Demo: Chat Models")
        logger.info("-" * 40)
        
        try:
            models_to_test = [
                ("gpt-4o-mini", "Cost-optimized"),
                ("gpt-4o", "Flagship"),
            ]
            
            test_prompt = "Explain artificial intelligence in exactly 50 words."
            
            for model, category in models_to_test:
                logger.info(f"Testing {category} model: {model}")
                
                result = await self.integration.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": test_prompt}
                    ],
                    model=model,
                    max_tokens=100
                )
                
                response_text = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                
                logger.info(f"PASS Response: {response_text}")
                logger.info(f"Tokens: {usage.get('total_tokens', 'Unknown')}")
                
                self.results.append({
                    "demo": f"chat_{model}",
                    "success": True,
                    "data": {
                        "model": model,
                        "response": response_text,
                        "usage": usage
                    }
                })
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"FAIL Chat models demo failed: {e}")
            self.results.append({
                "demo": "chat_models",
                "success": False,
                "error": str(e)
            })
    
    async def demo_image_generation(self):
        """Demo image generation with DALL-E."""
        logger.info("\nENHANCED Demo: Image Generation")
        logger.info("-" * 40)
        
        try:
            image_tool = self.tools["openai_image_generation"]
            
            prompts = [
                "A serene mountain landscape at sunset",
                "A cute robot reading a book in a library",
            ]
            
            for i, prompt in enumerate(prompts, 1):
                logger.info(f"Generating image {i}: {prompt}")
                
                result = await image_tool.aexecute(
                    prompt=prompt,
                    model="dall-e-3",
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                
                if result["success"]:
                    images = result["images"]
                    logger.info(f"PASS Generated {len(images)} image(s)")
                    for j, image in enumerate(images):
                        logger.info(f"  Image {j+1} URL: {image.get('url', 'N/A')}")
                        if image.get('revised_prompt'):
                            logger.info(f"  Revised prompt: {image['revised_prompt']}")
                else:
                    logger.error(f"FAIL Failed: {result.get('error')}")
                
                self.results.append({
                    "demo": f"image_generation_{i}",
                    "success": result["success"],
                    "data": result
                })
                
                await asyncio.sleep(2)  # Longer pause for image generation
            
        except Exception as e:
            logger.error(f"FAIL Image generation demo failed: {e}")
            self.results.append({
                "demo": "image_generation",
                "success": False,
                "error": str(e)
            })
    
    async def demo_text_to_speech(self):
        """Demo text-to-speech conversion."""
        logger.info("\n🔊 Demo: Text-to-Speech")
        logger.info("-" * 40)
        
        try:
            tts_tool = self.tools["openai_text_to_speech"]
            
            texts = [
                "Hello, this is a demonstration of OpenAI's text-to-speech technology.",
                "The weather is beautiful today!",
            ]
            
            voices = ["alloy", "echo"]
            
            for i, (text, voice) in enumerate(zip(texts, voices, strict=False), 1):
                logger.info(f"Converting text {i} with voice '{voice}': {text}")
                
                result = await tts_tool.aexecute(
                    text=text,
                    model="tts-1",
                    voice=voice,
                    output_format="mp3",
                    speed=1.0
                )
                
                if result["success"]:
                    audio_path = result["audio_path"]
                    audio_size = result["audio_size_bytes"]
                    logger.info(f"PASS Generated audio: {audio_path}")
                    logger.info(f"Audio size: {audio_size} bytes")
                else:
                    logger.error(f"FAIL Failed: {result.get('error')}")
                
                self.results.append({
                    "demo": f"text_to_speech_{i}",
                    "success": result["success"],
                    "data": result
                })
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"FAIL Text-to-speech demo failed: {e}")
            self.results.append({
                "demo": "text_to_speech",
                "success": False,
                "error": str(e)
            })
    
    async def demo_embeddings(self):
        """Demo text embeddings."""
        logger.info("\n🔢 Demo: Text Embeddings")
        logger.info("-" * 40)
        
        try:
            embeddings_tool = self.tools["openai_embeddings"]
            
            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Dogs are loyal and friendly animals.",
                "Python is a popular programming language.",
                "The ocean is vast and mysterious.",
            ]
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            result = await embeddings_tool.aexecute(
                texts=texts,
                model="text-embedding-3-small",  # More cost-effective
                dimensions=1536
            )
            
            if result["success"]:
                embeddings = result["embeddings"]
                logger.info(f"PASS Generated {len(embeddings)} embeddings")
                
                for i, embedding_data in enumerate(embeddings):
                    embedding = embedding_data["embedding"]
                    logger.info(f"  Text {i+1}: {len(embedding)} dimensions")
                    logger.info(f"    First 5 values: {embedding[:5]}")
                
                usage = result.get("usage", {})
                logger.info(f"Tokens used: {usage.get('total_tokens', 'Unknown')}")
                
                # Demonstrate similarity calculation
                if len(embeddings) >= 2:
                    import numpy as np
                    emb1 = np.array(embeddings[0]["embedding"])
                    emb2 = np.array(embeddings[1]["embedding"])
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    logger.info(f"Similarity between text 1 and 2: {similarity:.3f}")
            else:
                logger.error(f"FAIL Failed: {result.get('error')}")
            
            self.results.append({
                "demo": "embeddings",
                "success": result["success"],
                "data": result
            })
            
        except Exception as e:
            logger.error(f"FAIL Embeddings demo failed: {e}")
            self.results.append({
                "demo": "embeddings",
                "success": False,
                "error": str(e)
            })
    
    async def demo_content_moderation(self):
        """Demo content moderation."""
        logger.info("\nSecurity Demo: Content Moderation")
        logger.info("-" * 40)
        
        try:
            moderation_tool = self.tools["openai_moderation"]
            
            test_contents = [
                "This is a normal, safe message about technology.",
                "I love learning about artificial intelligence and programming.",
                "The weather is nice today. I hope everyone has a great day!",
            ]
            
            logger.info(f"Moderating {len(test_contents)} pieces of content")
            
            result = await moderation_tool.aexecute(
                content=test_contents,
                model="text-moderation-latest"
            )
            
            if result["success"]:
                moderation_results = result["moderation_results"]
                logger.info(f"PASS Moderated {len(moderation_results)} items")
                
                for i, mod_result in enumerate(moderation_results):
                    flagged = mod_result["flagged"]
                    categories = mod_result["categories"]
                    
                    status = "🚫 FLAGGED" if flagged else "PASS SAFE"
                    logger.info(f"  Content {i+1}: {status}")
                    
                    if flagged:
                        flagged_categories = [cat for cat, is_flagged in categories.items() if is_flagged]
                        logger.info(f"    Flagged categories: {flagged_categories}")
            else:
                logger.error(f"FAIL Failed: {result.get('error')}")
            
            self.results.append({
                "demo": "content_moderation",
                "success": result["success"],
                "data": result
            })
            
        except Exception as e:
            logger.error(f"FAIL Content moderation demo failed: {e}")
            self.results.append({
                "demo": "content_moderation",
                "success": False,
                "error": str(e)
            })
    
    async def demo_budget_tracking(self):
        """Demo budget tracking and usage monitoring."""
        logger.info("\n💰 Demo: Budget Tracking")
        logger.info("-" * 40)
        
        try:
            # Get current budget status
            budget_status = self.integration.get_budget_status()
            usage_summary = self.integration.get_usage_summary()
            
            logger.info("Budget Status:")
            logger.info(f"  Budget Limit: ${budget_status['budget_limit']:.2f}")
            logger.info(f"  Total Cost: ${budget_status['total_cost']:.4f}")
            logger.info(f"  Remaining: ${budget_status['remaining_budget']:.4f}")
            logger.info(f"  Utilization: {budget_status['budget_utilization_percent']:.1f}%")
            
            logger.info("\nUsage Summary:")
            logger.info(f"  Total Requests: {usage_summary['total_requests']}")
            logger.info(f"  Runtime: {usage_summary['runtime_seconds']:.1f}s")
            logger.info(f"  Cost per Request: ${usage_summary['cost_per_request']:.4f}")
            
            # Usage by model
            usage_by_model = usage_summary.get('usage_by_model', {})
            if usage_by_model:
                logger.info("\nUsage by Model:")
                for model, stats in usage_by_model.items():
                    logger.info(f"  {model}:")
                    logger.info(f"    Requests: {stats['request_count']}")
                    logger.info(f"    Cost: ${stats['total_cost']:.4f}")
                    logger.info(f"    Tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")
            
            self.results.append({
                "demo": "budget_tracking",
                "success": True,
                "data": {
                    "budget_status": budget_status,
                    "usage_summary": usage_summary
                }
            })
            
        except Exception as e:
            logger.error(f"FAIL Budget tracking demo failed: {e}")
            self.results.append({
                "demo": "budget_tracking",
                "success": False,
                "error": str(e)
            })
    
    async def demo_comprehensive_tool(self):
        """Demo the comprehensive tool that provides access to all APIs."""
        logger.info("\nFIXING Demo: Comprehensive Tool")
        logger.info("-" * 40)
        
        try:
            comprehensive_tool = self.tools["openai_comprehensive"]
            
            # Test different operations
            operations = [
                {
                    "operation": "chat",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "model": "gpt-4o-mini",
                    "max_tokens": 50
                },
                {
                    "operation": "embeddings",
                    "texts": ["Test embedding"],
                    "model": "text-embedding-3-small"
                },
                {
                    "operation": "moderation",
                    "content": ["This is safe content"],
                    "model": "text-moderation-latest"
                }
            ]
            
            for i, op_params in enumerate(operations, 1):
                operation = op_params.pop("operation")
                logger.info(f"Operation {i}: {operation}")
                
                result = await comprehensive_tool.aexecute(operation, **op_params)
                
                if result["success"]:
                    logger.info(f"PASS {operation} successful")
                else:
                    logger.error(f"FAIL {operation} failed: {result.get('error')}")
                
                self.results.append({
                    "demo": f"comprehensive_tool_{operation}",
                    "success": result["success"],
                    "data": result
                })
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"FAIL Comprehensive tool demo failed: {e}")
            self.results.append({
                "demo": "comprehensive_tool",
                "success": False,
                "error": str(e)
            })
    
    async def demo_error_handling(self):
        """Demo error handling and edge cases."""
        logger.info("\n⚠️ Demo: Error Handling")
        logger.info("-" * 40)
        
        try:
            # Test invalid model
            logger.info("Testing invalid model...")
            try:
                await self.integration.chat_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    model="invalid-model-name"
                )
                logger.info("FAIL Should have failed with invalid model")
            except Exception as e:
                logger.info(f"PASS Correctly caught error: {type(e).__name__}")
            
            # Test budget exceeded (simulate)
            logger.info("Testing budget tracking...")
            original_cost = self.integration.usage_tracker.total_cost
            remaining = self.integration.usage_tracker.get_remaining_budget()
            logger.info(f"Remaining budget: ${remaining:.4f}")
            
            # Test empty input
            logger.info("Testing empty input...")
            try:
                await self.integration.chat_completion(messages=[])
                logger.info("FAIL Should have failed with empty messages")
            except Exception as e:
                logger.info(f"PASS Correctly caught error: {type(e).__name__}")
            
            self.results.append({
                "demo": "error_handling",
                "success": True,
                "data": {"message": "Error handling tests completed"}
            })
            
        except Exception as e:
            logger.error(f"FAIL Error handling demo failed: {e}")
            self.results.append({
                "demo": "error_handling",
                "success": False,
                "error": str(e)
            })
    
    def generate_report(self):
        """Generate comprehensive demonstration report."""
        logger.info("\nRESULTS Demo Report")
        logger.info("=" * 50)
        
        total_demos = len(self.results)
        successful_demos = sum(1 for r in self.results if r["success"])
        failed_demos = total_demos - successful_demos
        
        runtime = datetime.now(timezone.utc) - self.start_time
        
        logger.info(f"Total Demonstrations: {total_demos}")
        logger.info(f"Successful: {successful_demos}")
        logger.info(f"Failed: {failed_demos}")
        logger.info(f"Success Rate: {(successful_demos/total_demos)*100:.1f}%")
        logger.info(f"Total Runtime: {runtime.total_seconds():.1f} seconds")
        
        if self.integration:
            final_budget = self.integration.get_budget_status()
            logger.info("\nFinal Budget Status:")
            logger.info(f"  Total Cost: ${final_budget['total_cost']:.4f}")
            logger.info(f"  Remaining: ${final_budget['remaining_budget']:.4f}")
        
        # Group results by category
        logger.info("\nDetailed Results:")
        for result in self.results:
            status = "PASS" if result["success"] else "FAIL"
            demo_name = result["demo"]
            logger.info(f"  {status} {demo_name}")
            if not result["success"] and "error" in result:
                logger.info(f"    Error: {result['error']}")
        
        # Save report to file
        report_data = {
            "summary": {
                "total_demos": total_demos,
                "successful_demos": successful_demos,
                "failed_demos": failed_demos,
                "success_rate": (successful_demos/total_demos)*100,
                "runtime_seconds": runtime.total_seconds(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "budget_status": self.integration.get_budget_status() if self.integration else None,
            "results": self.results
        }
        
        report_path = f"openai_comprehensive_demo_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\n📄 Report saved to: {report_path}")
        
        return report_data
    
    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        logger.info("Starting OpenAI Comprehensive Integration Demo")
        logger.info("=" * 60)
        
        if not await self.initialize():
            return
        
        # Run all demos in sequence
        demos = [
            self.demo_health_check(),
            self.demo_reasoning_models(),
            self.demo_chat_models(),
            self.demo_image_generation(),
            self.demo_text_to_speech(),
            self.demo_embeddings(),
            self.demo_content_moderation(),
            self.demo_comprehensive_tool(),
            self.demo_budget_tracking(),
            self.demo_error_handling(),
        ]
        
        for demo in demos:
            try:
                await demo
            except Exception as e:
                logger.error(f"Demo failed: {e}")
                continue
        
        # Generate final report
        return self.generate_report()


async def main():
    """Main demonstration function."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("FAIL Please set OPENAI_API_KEY environment variable")
        logger.info("INSIGHT Export your OpenAI API key:")
        logger.info("   export OPENAI_API_KEY='${OPENAI_API_KEY}'")
        return
    
    # Run demo with reasonable budget limit
    demo = OpenAIComprehensiveDemo(budget_limit=10.0)  # $10 limit for demo
    
    try:
        report = await demo.run_all_demos()
        
        logger.info("\nSUCCESS Demo completed successfully!")
        logger.info(f"💰 Total cost: ${report['budget_status']['total_cost']:.4f}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"FAIL Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 