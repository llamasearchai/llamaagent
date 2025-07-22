#!/usr/bin/env python3
"""
Standalone Master Diagnostics Runner

Runs comprehensive diagnostics on the LlamaAgent system and generates
a detailed problem report.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from llamaagent.diagnostics.master_diagnostics import MasterDiagnostics
    
    def main():
        """Run comprehensive diagnostics and generate report."""
        print("Scanning Starting LlamaAgent Master Diagnostics...")
        print("=" * 60)
        
        # Initialize diagnostics
        diagnostics = MasterDiagnostics(str(project_root))
        
        try:
            # Run comprehensive analysis
            report = diagnostics.run_comprehensive_analysis()
            
            # Save detailed report
            output_file = diagnostics.save_report_to_file(report)
            
            print(f"\nPASS Diagnostic report saved to: {output_file}")
            print(f"RESULTS Analysis Summary:")
            print(f"   • Total Issues: {report.total_problems}")
            print(f"   • Files Analyzed: {report.total_files_analyzed}")
            print(f"   • Lines Analyzed: {report.total_lines_analyzed}")
            print(f"   • Analysis Time: {report.analysis_duration:.2f}s")
            
            # Show severity breakdown
            severity_counts = report.problems_by_severity
            if severity_counts:
                print(f"\nPerformance Issues by Severity:")
                for severity, count in severity_counts.items():
                    if count > 0:
                        emoji = {
                            "CRITICAL": "URGENT",
                            "HIGH": "WARNING: ",
                            "MEDIUM": "INSIGHT",
                            "LOW": "ℹ ",
                            "INFO": "Response"
                        }.get(severity.value, "•")
                        print(f"   {emoji} {severity.value}: {count}")
            
            # Show top recommendations
            if report.recommendations:
                print(f"\nTARGET Top Recommendations:")
                for i, rec in enumerate(report.recommendations[:5], 1):
                    print(f"   {i}. {rec}")
                if len(report.recommendations) > 5:
                    print(f"   ... and {len(report.recommendations) - 5} more")
            
            # Show critical issues
            critical_issues = [p for p in report.problems if p.severity.value == "CRITICAL"]
            if critical_issues:
                print(f"\nURGENT CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
                for issue in critical_issues[:10]:  # Show first 10
                    print(f"   • {issue.title}")
                    print(f"      {issue.location}")
                    if issue.line_number:
                        print(f"      Line {issue.line_number}")
                    if issue.suggested_fix:
                        print(f"     INSIGHT {issue.suggested_fix}")
                    print()
                
                if len(critical_issues) > 10:
                    print(f"   ... and {len(critical_issues) - 10} more critical issues")
            
            # Exit code based on severity
            critical_count = severity_counts.get("CRITICAL", 0) if severity_counts else 0
            high_count = severity_counts.get("HIGH", 0) if severity_counts else 0
            
            if critical_count > 0:
                print(f"\nFAIL RESULT: {critical_count} critical issues found - system cannot function properly!")
                print("FIXING Fix critical issues first, then run diagnostics again.")
                return 1
            elif high_count > 0:
                print(f"\nWARNING:  RESULT: {high_count} high-priority issues found - attention recommended")
                print("LIST: See full report for detailed fixes.")
                return 0
            else:
                print(f"\nPASS RESULT: No critical issues found - system is healthy!")
                return 0
                
        except Exception as e:
            print(f"\nFAIL Error during analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return 1
    
except ImportError as e:
    def main():
        """Fallback when diagnostics module can't be imported."""
        print("FAIL Cannot import diagnostics module - running basic syntax check...")
        print(f"Import error: {e}")
        print("=" * 60)
        
        # Basic syntax checking
        python_files = list(project_root.rglob("src/**/*.py"))
        critical_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                import ast
                ast.parse(content)
                print(f"PASS {py_file.relative_to(project_root)}")
                
            except SyntaxError as se:
                print(f"FAIL {py_file.relative_to(project_root)} - SYNTAX ERROR: {se}")
                critical_issues.append(f"Syntax error in {py_file.relative_to(project_root)}: {se}")
            except Exception as ex:
                print(f"WARNING:  {py_file.relative_to(project_root)} - Warning: {ex}")
        
        print(f"\nRESULTS Basic Analysis Complete:")
        print(f"   • Files Checked: {len(python_files)}")
        print(f"   • Critical Issues: {len(critical_issues)}")
        
        if critical_issues:
            print(f"\nURGENT CRITICAL SYNTAX ERRORS:")
            for issue in critical_issues:
                print(f"   • {issue}")
            
            print(f"\nFAIL RESULT: {len(critical_issues)} critical syntax errors prevent system operation!")
            print("FIXING Fix syntax errors first, then run full diagnostics.")
            return 1
        else:
            print(f"\nPASS RESULT: No syntax errors found!")
            print("INSIGHT Run full diagnostics after fixing import issues.")
            return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 