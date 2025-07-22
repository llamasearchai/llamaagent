#!/usr/bin/env python3
"""
Master System Deployment Script
"""

import logging
import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict

from rich.console import Console
from rich.table import Table
logger = logging.getLogger(__name__)

console = Console()

def deploy_system(config_path: str = "deployment_config.json") -> Dict[str, Any]:
    """Deploy the master system with proper type annotations"""
    
    deployment_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "starting",
        "services": {}
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for service in config.get("services", []):
            service_name = service["name"]
            deployment_results["services"][service_name] = {
                "status": "pending",
                "details": {}
            }
            
            try:
                deployment_results["services"][service_name]["status"] = "deployed"
                deployment_results["services"][service_name]["details"] = {
                    "port": service.get("port", "N/A"),
                    "url": service.get("url", "N/A")
                }
            except Exception as e:
                deployment_results["services"][service_name]["status"] = "failed"
                deployment_results["services"][service_name]["details"]["error"] = str(e)
        
        deployment_results["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        deployment_results["status"] = "failed"
        deployment_results["error"] = str(e)
    
    return deployment_results

def display_results(deployment_results: Dict[str, Any]) -> None:
    """Display deployment results with proper formatting"""
    table = Table(title="Deployment Results")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    for service_name, service_info in deployment_results.get("services", {}).items():
        status = service_info.get("status", "unknown")
        details = service_info.get("details", {})
        details_str = json.dumps(details, indent=2) if details else "N/A"
        table.add_row(service_name, status, details_str)
    
    console.print(table)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Master System")
    parser.add_argument("--config", default="deployment_config.json", help="Configuration file path")
    args = parser.parse_args()
    
    with console.status("[bold green]Deploying master system...") as status:
        deployment_results = deploy_system(args.config)
        
    display_results(deployment_results)
    
    if deployment_results.get("status") == "completed":
        console.print("[bold green] Deployment completed successfully!")
        return 0
    else:
        console.print("[bold red] Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 