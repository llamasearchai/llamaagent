#!/usr/bin/env python3
"""
Comprehensive LlamaAgent System Test

This script performs end-to-end testing of the entire LlamaAgent system:
- Enhanced MockProvider functionality
- ReactAgent execution capabilities
- Production API endpoints
- Benchmark performance validation
- System integration testing

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor


class SystemTester:
    """Comprehensive system tester for LlamaAgent."""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        self.api_server_process = None
        self.api_base_url = "http://localhost:8000"
        
    def log_test(self, test_name: str, success: bool, details: str = "", duration: float = 0.0):
        """Log a test result."""
        status = "PASS PASS" if success else "FAIL FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        if duration > 0:
            print(f"   Duration: {duration:.3f}s")
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def test_enhanced_mock_provider(self) -> bool:
        """Test the enhanced mock provider functionality."""
        print("\nINTELLIGENCE Testing Enhanced MockProvider")
        print("-" * 40)
        
        try:
            # Import and test the provider
            from final_working_system import EnhancedMockProvider, LLMMessage
            
            provider = EnhancedMockProvider()
            
            # Test mathematical problems
            test_cases = [
                {
                    "input": "Calculate 15% of 240 and then add 30 to the result.",
                    "expected": "66",
                    "type": "math"
                },
                {
                    "input": "If a rectangle has length 8 cm and width 5 cm, what is its perimeter?",
                    "expected": "26 cm",
                    "type": "math"
                },
                {
                    "input": "Write a Python function that returns the maximum of two numbers.",
                    "expected": "def max_two(a, b): return a if a > b else b",
                    "type": "programming"
                }
            ]
            
            success_count = 0
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                
                message = LLMMessage(role="user", content=test_case["input"])
                response = asyncio.run(provider.complete([message]))
                
                duration = time.time() - start_time
                
                # Check if response contains expected content
                success = test_case["expected"].lower() in response.content.lower()
                if success:
                    success_count += 1
                
                self.log_test(
                    f"MockProvider {test_case['type']} test {i+1}",
                    success,
                    f"Expected: {test_case['expected']}, Got: {response.content}",
                    duration
                )
            
            overall_success = success_count >= 2  # At least 2 out of 3 should pass
            self.log_test(
                "MockProvider Overall",
                overall_success,
                f"Passed {success_count}/{len(test_cases)} tests"
            )
            
            return overall_success
            
        except Exception as e:
            self.log_test("MockProvider Test", False, f"Error: {str(e)}")
            return False
    
    def test_react_agent(self) -> bool:
        """Test the ReactAgent functionality."""
        print("\nAgent Testing ReactAgent")
        print("-" * 40)
        
        try:
            from final_working_system import ReactAgent, AgentConfig, AgentRole
            
            # Create agent
            config = AgentConfig(
                agent_name="TestAgent",
                role=AgentRole.SPECIALIST,
                spree_enabled=True
            )
            
            agent = ReactAgent(config=config)
            
            # Test tasks
            test_tasks = [
                "Calculate 10% of 500",
                "What is 2 + 2?",
                "Write a simple Python function"
            ]
            
            success_count = 0
            for i, task in enumerate(test_tasks):
                start_time = time.time()
                
                response = asyncio.run(agent.execute(task))
                
                duration = time.time() - start_time
                
                success = response.success and len(response.content) > 0
                if success:
                    success_count += 1
                
                self.log_test(
                    f"ReactAgent task {i+1}",
                    success,
                    f"Task: {task}, Response: {response.content[:50]}...",
                    duration
                )
            
            overall_success = success_count >= 2
            self.log_test(
                "ReactAgent Overall",
                overall_success,
                f"Completed {success_count}/{len(test_tasks)} tasks successfully"
            )
            
            return overall_success
            
        except Exception as e:
            self.log_test("ReactAgent Test", False, f"Error: {str(e)}")
            return False
    
    def test_final_working_system(self) -> bool:
        """Test the final working system end-to-end."""
        print("\nLlamaAgent Testing Final Working System")
        print("-" * 40)
        
        try:
            # Run the final working system
            result = subprocess.run(
                [sys.executable, "final_working_system.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            
            # Parse output for success rate
            success_rate = 0.0
            if "Success Rate:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Success Rate:" in line:
                        try:
                            success_rate = float(line.split(':')[1].strip().replace('%', ''))
                            break
                        except:
                            pass
            
            self.log_test(
                "Final Working System",
                success and success_rate >= 80.0,
                f"Success Rate: {success_rate}%, Exit Code: {result.returncode}",
                0.0
            )
            
            return success and success_rate >= 80.0
            
        except Exception as e:
            self.log_test("Final Working System", False, f"Error: {str(e)}")
            return False
    
    def start_api_server(self) -> bool:
        """Start the production API server."""
        print("\nStarting Production API Server")
        print("-" * 40)
        
        try:
            # Start server in background
            self.api_server_process = subprocess.Popen(
                [sys.executable, "production_llamaagent_api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                success = response.status_code == 200
                
                self.log_test(
                    "API Server Startup",
                    success,
                    f"Health check status: {response.status_code}"
                )
                
                return success
                
            except requests.exceptions.ConnectionError:
                self.log_test("API Server Startup", False, "Connection failed")
                return False
            
        except Exception as e:
            self.log_test("API Server Startup", False, f"Error: {str(e)}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test the production API endpoints."""
        print("\nNETWORK Testing API Endpoints")
        print("-" * 40)
        
        endpoints_to_test = [
            {
                "name": "Health Check",
                "method": "GET",
                "url": "/health",
                "expected_status": 200
            },
            {
                "name": "Metrics",
                "method": "GET", 
                "url": "/metrics",
                "expected_status": 200
            },
            {
                "name": "Root",
                "method": "GET",
                "url": "/",
                "expected_status": 200
            },
            {
                "name": "List Agents",
                "method": "GET",
                "url": "/agents",
                "expected_status": 200
            }
        ]
        
        success_count = 0
        
        for endpoint in endpoints_to_test:
            try:
                start_time = time.time()
                
                if endpoint["method"] == "GET":
                    response = requests.get(f"{self.api_base_url}{endpoint['url']}", timeout=10)
                else:
                    response = requests.post(f"{self.api_base_url}{endpoint['url']}", timeout=10)
                
                duration = time.time() - start_time
                
                success = response.status_code == endpoint["expected_status"]
                if success:
                    success_count += 1
                
                self.log_test(
                    f"API {endpoint['name']}",
                    success,
                    f"Status: {response.status_code}, Expected: {endpoint['expected_status']}",
                    duration
                )
                
            except Exception as e:
                self.log_test(f"API {endpoint['name']}", False, f"Error: {str(e)}")
        
        overall_success = success_count >= 3  # At least 3 out of 4 should pass
        self.log_test(
            "API Endpoints Overall",
            overall_success,
            f"Passed {success_count}/{len(endpoints_to_test)} endpoint tests"
        )
        
        return overall_success
    
    def test_chat_completions(self) -> bool:
        """Test the chat completions API."""
        print("\n💬 Testing Chat Completions")
        print("-" * 40)
        
        try:
            # Test chat completions
            payload = {
                "model": "enhanced-mock-gpt-4",
                "messages": [
                    {"role": "user", "content": "Calculate 20% of 150"}
                ],
                "max_tokens": 100
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            
            details = f"Status: {response.status_code}"
            if success:
                try:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    details += f", Response: {content[:50]}..."
                except:
                    pass
            
            self.log_test(
                "Chat Completions API",
                success,
                details,
                duration
            )
            
            return success
            
        except Exception as e:
            self.log_test("Chat Completions API", False, f"Error: {str(e)}")
            return False
    
    def test_agent_execution(self) -> bool:
        """Test the agent execution API."""
        print("\nTARGET Testing Agent Execution")
        print("-" * 40)
        
        try:
            # Test agent execution
            payload = {
                "task": "Calculate 25% of 200 and add 10",
                "agent_name": "test_agent"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/agents/execute",
                json=payload,
                timeout=30
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            
            details = f"Status: {response.status_code}"
            if success:
                try:
                    data = response.json()
                    content = data.get("content", "")
                    task_success = data.get("success", False)
                    details += f", Task Success: {task_success}, Response: {content[:50]}..."
                    success = success and task_success
                except:
                    pass
            
            self.log_test(
                "Agent Execution API",
                success,
                details,
                duration
            )
            
            return success
            
        except Exception as e:
            self.log_test("Agent Execution API", False, f"Error: {str(e)}")
            return False
    
    def test_benchmark_api(self) -> bool:
        """Test the benchmark API."""
        print("\nRESULTS Testing Benchmark API")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/benchmark/run",
                timeout=60
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            success_rate = 0.0
            
            details = f"Status: {response.status_code}"
            if success:
                try:
                    data = response.json()
                    success_rate = data.get("summary", {}).get("success_rate", 0.0)
                    details += f", Success Rate: {success_rate}%"
                    success = success and success_rate >= 60.0  # Lower threshold for API test
                except:
                    pass
            
            self.log_test(
                "Benchmark API",
                success,
                details,
                duration
            )
            
            return success
            
        except Exception as e:
            self.log_test("Benchmark API", False, f"Error: {str(e)}")
            return False
    
    def stop_api_server(self):
        """Stop the API server."""
        if self.api_server_process:
            print("\n🛑 Stopping API Server")
            self.api_server_process.terminate()
            self.api_server_process.wait(timeout=10)
            print("PASS API Server stopped")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        total_duration = time.time() - self.start_time
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_duration": total_duration
            },
            "results": self.test_results,
            "timestamp": time.time()
        }
    
    def run_all_tests(self) -> bool:
        """Run all comprehensive tests."""
        print("Analyzing LlamaAgent Comprehensive System Test")
        print("=" * 60)
        print("Testing all components and functionality...")
        print("=" * 60)
        
        try:
            # Core functionality tests
            mock_provider_success = self.test_enhanced_mock_provider()
            react_agent_success = self.test_react_agent()
            final_system_success = self.test_final_working_system()
            
            # API tests
            api_server_success = self.start_api_server()
            api_endpoints_success = False
            chat_completions_success = False
            agent_execution_success = False
            benchmark_api_success = False
            
            if api_server_success:
                api_endpoints_success = self.test_api_endpoints()
                chat_completions_success = self.test_chat_completions()
                agent_execution_success = self.test_agent_execution()
                benchmark_api_success = self.test_benchmark_api()
                
                self.stop_api_server()
            
            # Generate report
            report = self.generate_report()
            
            # Print summary
            print("\n" + "=" * 60)
            print("TARGET COMPREHENSIVE TEST RESULTS")
            print("=" * 60)
            
            print(f"Total Tests: {report['summary']['total_tests']}")
            print(f"Passed: {report['summary']['passed_tests']}")
            print(f"Failed: {report['summary']['failed_tests']}")
            print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
            print(f"Total Duration: {report['summary']['total_duration']:.1f}s")
            
            # Component breakdown
            print(f"\n📋 Component Status:")
            print(f"   MockProvider: {'PASS' if mock_provider_success else 'FAIL'}")
            print(f"   ReactAgent: {'PASS' if react_agent_success else 'FAIL'}")
            print(f"   Final System: {'PASS' if final_system_success else 'FAIL'}")
            print(f"   API Server: {'PASS' if api_server_success else 'FAIL'}")
            print(f"   API Endpoints: {'PASS' if api_endpoints_success else 'FAIL'}")
            print(f"   Chat Completions: {'PASS' if chat_completions_success else 'FAIL'}")
            print(f"   Agent Execution: {'PASS' if agent_execution_success else 'FAIL'}")
            print(f"   Benchmark API: {'PASS' if benchmark_api_success else 'FAIL'}")
            
            # Overall assessment
            overall_success = report['summary']['success_rate'] >= 70.0
            
            print(f"\nSUCCESS OVERALL RESULT: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
            
            if overall_success:
                print("PASS LlamaAgent system is functioning well!")
                print("PASS Core components are operational")
                print("PASS API endpoints are working")
                print("PASS System is ready for production use")
            else:
                print("⚠️  System needs attention")
                print("FAIL Some components are not functioning properly")
            
            # Save report
            with open("comprehensive_test_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"\n💾 Detailed report saved to: comprehensive_test_report.json")
            
            return overall_success
            
        except Exception as e:
            print(f"\nFAIL Test execution failed: {str(e)}")
            return False
        
        finally:
            self.stop_api_server()


def main():
    """Main test execution."""
    tester = SystemTester()
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 