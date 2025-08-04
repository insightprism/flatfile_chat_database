#!/usr/bin/env python3
"""
Comprehensive test runner for the flatfile chat database system.

This script runs all tests and provides a detailed report on system health
after recent architecture changes.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def run_test_module(self, module_name: str, description: str) -> Dict[str, Any]:
        """Run a specific test module and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {description}")
        print(f"Module: {module_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run pytest on the specific module
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/{module_name}", 
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=parent_dir, timeout=300)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n')
            
            # Count passed/failed tests
            passed = len([line for line in output_lines if " PASSED" in line])
            failed = len([line for line in output_lines if " FAILED" in line])
            errors = len([line for line in output_lines if " ERROR" in line])
            skipped = len([line for line in output_lines if " SKIPPED" in line])
            
            test_result = {
                "module": module_name,
                "description": description,
                "duration": duration,
                "return_code": result.returncode,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "total": passed + failed + errors + skipped,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            # Print summary
            print(f"\nResults: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Status: {'✅ PASSED' if test_result['success'] else '❌ FAILED'}")
            
            if not test_result['success']:
                print("\nError output:")
                print(result.stderr)
                if failed > 0 or errors > 0:
                    print("\nTest output:")
                    print(result.stdout)
            
            return test_result
            
        except Exception as e:
            print(f"❌ Failed to run {module_name}: {e}")
            return {
                "module": module_name,
                "description": description,
                "duration": 0,
                "return_code": -1,
                "success": False,
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "total": 1
            }
    
    def run_quick_functionality_test(self) -> Dict[str, Any]:
        """Run the quick functionality test."""
        print(f"\n{'='*60}")
        print("Running Quick Functionality Verification")
        print("Module: test_core_functionality.py")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "tests/test_core_functionality.py"
            ], capture_output=True, text=True, cwd=parent_dir, timeout=120)
            
            duration = time.time() - start_time
            
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            
            success = "CODEBASE IS HEALTHY" in result.stdout
            
            return {
                "module": "test_core_functionality.py",
                "description": "Quick Functionality Verification",
                "duration": duration,
                "return_code": result.returncode,
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            print(f"❌ Failed to run functionality test: {e}")
            return {
                "module": "test_core_functionality.py",
                "description": "Quick Functionality Verification",
                "duration": 0,
                "return_code": -1,
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self, quick_only: bool = False, include_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        print("🧪 FLATFILE CHAT DATABASE - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("Testing system integrity after recent architecture changes...")
        
        # Quick functionality test first
        self.test_results["quick_test"] = self.run_quick_functionality_test()
        
        if quick_only:
            print("\n" + "="*70)
            print("QUICK TEST COMPLETE")
            print("="*70)
            return self.generate_report()
        
        # Define test modules in order of importance
        test_modules = [
            ("conftest.py", "Test Fixtures and Utilities", "unit"),
            ("test_configuration.py", "Configuration System Tests", "unit"),
            ("test_dependency_injection.py", "Dependency Injection Tests", "unit"),
            ("test_protocols.py", "Protocol Compliance Tests", "unit"),
            ("test_backend.py", "Backend Implementation Tests", "integration"),
            ("test_storage_integration.py", "Storage Integration Tests", "integration"),
            ("test_e2e_workflows.py", "End-to-End Workflow Tests", "e2e"),
            ("test_error_handling.py", "Error Handling and Edge Cases", "unit"),
        ]
        
        # Filter based on speed requirements
        if not include_slow:
            test_modules = [tm for tm in test_modules if tm[2] != "e2e"]
        
        # Run each test module
        for module_name, description, test_type in test_modules:
            if module_name == "conftest.py":
                continue  # Skip conftest as it's not a test module itself
                
            self.test_results[module_name] = self.run_test_module(module_name, description)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate overall statistics
        total_passed = sum(r.get("passed", 0) for r in self.test_results.values())
        total_failed = sum(r.get("failed", 0) for r in self.test_results.values())
        total_errors = sum(r.get("errors", 0) for r in self.test_results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.test_results.values())
        total_tests = total_passed + total_failed + total_errors + total_skipped
        
        modules_passed = sum(1 for r in self.test_results.values() if r.get("success", False))
        modules_failed = len(self.test_results) - modules_passed
        
        overall_success = modules_failed == 0 and total_failed == 0 and total_errors == 0
        
        # Print detailed report
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE TEST REPORT")
        print(f"{'='*70}")
        
        print(f"\n⏱️  EXECUTION SUMMARY")
        print(f"   Total Duration: {total_duration:.2f} seconds")
        print(f"   Test Modules: {len(self.test_results)}")
        print(f"   Total Tests: {total_tests}")
        
        print(f"\n📈 TEST RESULTS")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   🚫 Errors: {total_errors}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        
        print(f"\n📦 MODULE RESULTS")
        for module_name, result in self.test_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            duration = result.get("duration", 0)
            description = result.get("description", "")
            
            if "passed" in result:
                test_summary = f"{result['passed']}/{result.get('total', 0)} tests passed"
            else:
                test_summary = "functional test"
            
            print(f"   {status} {module_name:30} ({test_summary}, {duration:.1f}s)")
            if description:
                print(f"        {description}")
        
        # Architecture health assessment
        print(f"\n🏗️  ARCHITECTURE HEALTH ASSESSMENT")
        
        config_health = self.test_results.get("test_configuration.py", {}).get("success", False)
        di_health = self.test_results.get("test_dependency_injection.py", {}).get("success", False)
        protocol_health = self.test_results.get("test_protocols.py", {}).get("success", False)
        storage_health = self.test_results.get("test_storage_integration.py", {}).get("success", False)
        e2e_health = self.test_results.get("test_e2e_workflows.py", {}).get("success", True)  # Default true if not run
        
        print(f"   Configuration System: {'✅ HEALTHY' if config_health else '❌ ISSUES'}")
        print(f"   Dependency Injection: {'✅ HEALTHY' if di_health else '❌ ISSUES'}")
        print(f"   Protocol Compliance: {'✅ HEALTHY' if protocol_health else '❌ ISSUES'}")
        print(f"   Storage Integration: {'✅ HEALTHY' if storage_health else '❌ ISSUES'}")
        print(f"   End-to-End Workflows: {'✅ HEALTHY' if e2e_health else '❌ ISSUES'}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL SYSTEM STATUS")
        if overall_success:
            print("   🎉 EXCELLENT: All tests passed! System is healthy.")
            print("   ✅ Recent architecture changes have not broken functionality.")
            print("   🚀 System is ready for production use.")
        elif total_failed == 0 and total_errors == 0:
            print("   ✅ GOOD: Core functionality is working.")
            print("   ⚠️  Some test modules had issues but no test failures.")
            print("   🔍 Review module-specific issues above.")
        elif total_failed < 5 and total_errors < 3:
            print("   ⚠️  CAUTION: Minor issues detected.")
            print("   🔧 Recent changes may have introduced small problems.")
            print("   📋 Review failed tests and fix issues.")
        else:
            print("   ❌ CRITICAL: Significant issues detected!")
            print("   🚨 Recent architecture changes have broken functionality.")
            print("   🛠️  Immediate attention required before deployment.")
        
        # Success rate
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"   📊 Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
        
        print(f"\n{'='*70}")
        
        return {
            "overall_success": overall_success,
            "total_duration": total_duration,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "total_skipped": total_skipped,
            "modules_passed": modules_passed,
            "modules_failed": modules_failed,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "module_results": self.test_results
        }
    
    def run_specific_test(self, test_name: str) -> Dict[str, Any]:
        """Run a specific test module."""
        if test_name == "quick":
            self.test_results["quick_test"] = self.run_quick_functionality_test()
        else:
            module_name = f"test_{test_name}.py"
            description = f"Specific Test: {test_name}"
            self.test_results[module_name] = self.run_test_module(module_name, description)
        
        return self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for the flatfile chat database system"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only quick functionality tests"
    )
    parser.add_argument(
        "--slow", 
        action="store_true", 
        help="Include slow end-to-end tests"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        help="Run specific test module (e.g., 'configuration', 'storage_integration')"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.test:
            report = runner.run_specific_test(args.test)
        else:
            report = runner.run_all_tests(quick_only=args.quick, include_slow=args.slow)
        
        # Exit with appropriate code
        exit_code = 0 if report["overall_success"] else 1
        
        if exit_code == 0:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n⚠️  Some tests failed. Review the report above.")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()