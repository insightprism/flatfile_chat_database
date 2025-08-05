#!/usr/bin/env python3
# FF Chat API Comprehensive Test Runner
"""
Comprehensive test runner for FF Chat API Phase 4 Production Ready testing.
Runs all API tests including core functionality, security, performance, and use case validation.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class FFChatAPITestRunner:
    """Comprehensive test runner for FF Chat API"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_test_suite(self, test_type: str, test_path: str, markers: List[str] = None) -> Dict[str, Any]:
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {test_type} Tests")
        print(f"{'='*60}")
        
        # Build pytest command - use python3 to match system
        cmd = ["python3", "-m", "pytest", test_path, "-v", "--tb=short"]
        
        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add coverage if available
        try:
            # Check if pytest-cov is available
            import subprocess
            result = subprocess.run(["python3", "-c", "import pytest_cov"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cmd.extend(["--cov=ff_chat_api", "--cov=ff_chat_auth", "--cov=ff_chat_application"])
        except:
            pass  # Coverage not required
        
        start_time = time.time()
        
        try:
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            test_result = {
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            # Parse test results from output
            if "failed" in result.stdout.lower():
                # Extract failed test count
                lines = result.stdout.split('\n')
                for line in lines:
                    if "failed" in line.lower() and "passed" in line.lower():
                        test_result["summary"] = line.strip()
                        break
            elif "passed" in result.stdout.lower():
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line.lower():
                        test_result["summary"] = line.strip()
                        break
            
            print(f"\n{test_type} Tests Results:")
            print(f"Status: {'PASSED' if success else 'FAILED'}")
            print(f"Duration: {duration:.2f} seconds")
            
            if "summary" in test_result:
                print(f"Summary: {test_result['summary']}")
            
            if not success:
                print(f"STDERR: {result.stderr[:500]}...")
                print(f"STDOUT: {result.stdout[-500:]}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"{test_type} tests timed out after 10 minutes")
            return {
                "success": False,
                "duration": 600,
                "error": "Test timeout",
                "summary": "Tests timed out"
            }
        except Exception as e:
            print(f"Error running {test_type} tests: {e}")
            return {
                "success": False,
                "duration": 0,
                "error": str(e),
                "summary": f"Test execution error: {str(e)}"
            }
    
    def run_all_tests(self, skip_slow=False, skip_performance=False):
        """Run all comprehensive API tests"""
        print("🚀 Starting FF Chat API Comprehensive Test Suite")
        print(f"Test Root: {project_root}")
        
        self.start_time = time.time()
        
        # Test suites to run
        test_suites = [
            {
                "name": "Core API Tests",
                "path": "tests/api/test_ff_chat_api_core.py",
                "markers": ["integration"],
                "skip": False
            },
            {
                "name": "Security Tests",
                "path": "tests/security/test_ff_api_security.py",
                "markers": ["security"],
                "skip": False
            },
            {
                "name": "Performance Tests",
                "path": "tests/load/test_ff_api_performance.py",
                "markers": ["performance"],
                "skip": skip_performance or skip_slow
            },
            {
                "name": "Use Case Tests",
                "path": "tests/system/test_ff_api_use_cases.py",
                "markers": ["system"],
                "skip": False
            },
            {
                "name": "Existing Integration Tests",
                "path": "tests/test_ff_chat_phase1_integration.py",
                "markers": ["integration"],
                "skip": False
            }
        ]
        
        # Run each test suite
        for suite in test_suites:
            if suite["skip"]:
                print(f"\n⏭️  Skipping {suite['name']} (disabled)")
                self.test_results[suite["name"]] = {
                    "success": None,
                    "duration": 0,
                    "summary": "Skipped"
                }
                continue
            
            # Check if test file exists
            test_path = project_root / suite["path"]
            if not test_path.exists():
                print(f"\n⚠️  Test file not found: {suite['path']}")
                self.test_results[suite["name"]] = {
                    "success": False,
                    "duration": 0,
                    "summary": "Test file not found"
                }
                continue
            
            # Run the test suite
            result = self.run_test_suite(
                suite["name"],
                suite["path"],
                suite.get("markers")
            )
            
            self.test_results[suite["name"]] = result
        
        self.end_time = time.time()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        print(f"\n{'='*80}")
        print("FF CHAT API COMPREHENSIVE TEST REPORT")
        print(f"{'='*80}")
        
        print(f"Total Test Duration: {total_duration:.2f} seconds")
        print(f"Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary statistics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for result in self.test_results.values() if result.get("success") is True)
        failed_suites = sum(1 for result in self.test_results.values() if result.get("success") is False)
        skipped_suites = sum(1 for result in self.test_results.values() if result.get("success") is None)
        
        print(f"\nTest Suite Summary:")
        print(f"  Total Suites: {total_suites}")
        print(f"  Passed: {passed_suites}")
        print(f"  Failed: {failed_suites}")
        print(f"  Skipped: {skipped_suites}")
        
        if total_suites > 0:
            success_rate = (passed_suites / (passed_suites + failed_suites)) * 100 if (passed_suites + failed_suites) > 0 else 0
            print(f"  Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print(f"{'Suite Name':<25} {'Status':<10} {'Duration':<10} {'Summary'}")
        print(f"{'-'*70}")
        
        for suite_name, result in self.test_results.items():
            status = "PASSED" if result.get("success") is True else \
                    "FAILED" if result.get("success") is False else "SKIPPED"
            duration = f"{result.get('duration', 0):.1f}s"
            summary = result.get("summary", "No summary")[:30]
            
            print(f"{suite_name[:24]:<25} {status:<10} {duration:<10} {summary}")
        
        # Failure details
        failed_tests = {name: result for name, result in self.test_results.items() 
                       if result.get("success") is False}
        
        if failed_tests:
            print(f"\nFailure Details:")
            for suite_name, result in failed_tests.items():
                print(f"\n❌ {suite_name}:")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                if "stderr" in result and result["stderr"]:
                    print(f"   STDERR: {result['stderr'][:200]}...")
        
        # Success details
        passed_tests = {name: result for name, result in self.test_results.items() 
                       if result.get("success") is True}
        
        if passed_tests:
            print(f"\nSuccessful Test Suites:")
            for suite_name in passed_tests.keys():
                print(f"✅ {suite_name}")
        
        # Final status
        overall_success = failed_suites == 0 and passed_suites > 0
        
        print(f"\n{'='*80}")
        if overall_success:
            print("🎉 FF CHAT API COMPREHENSIVE TESTS: PASSED")
            print("✅ Phase 4: Production Ready - API Testing Complete")
            if passed_suites == total_suites - skipped_suites:
                print("🏆 ALL TEST SUITES PASSED - PRODUCTION READY!")
        else:
            print("❌ FF CHAT API COMPREHENSIVE TESTS: FAILED")
            print(f"⚠️  {failed_suites} test suite(s) failed")
            print("🔧 Review failures before production deployment")
        
        print(f"{'='*80}")
        
        return overall_success
    
    def run_quick_validation(self):
        """Run quick validation tests for basic functionality"""
        print("🏃 Running Quick API Validation Tests")
        
        quick_tests = [
            "tests/api/test_ff_chat_api_core.py::TestFFChatAPICoreEndpoints::test_api_health_check",
            "tests/system/test_ff_api_use_cases.py::TestFFChatAPIUseCaseCoverage::test_use_case_coverage_summary"
        ]
        
        for test in quick_tests:
            test_path = project_root / test.split("::")[0]
            if test_path.exists():
                result = self.run_test_suite(f"Quick Test", test)
                if not result.get("success"):
                    print(f"❌ Quick validation failed")
                    return False
        
        print("✅ Quick validation passed")
        return True

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="FF Chat API Comprehensive Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests only")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow-running tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--suite", type=str, help="Run specific test suite only")
    
    args = parser.parse_args()
    
    runner = FFChatAPITestRunner()
    
    if args.quick:
        success = runner.run_quick_validation()
        sys.exit(0 if success else 1)
    
    if args.suite:
        # Run specific suite
        suite_map = {
            "core": "tests/api/test_ff_chat_api_core.py",
            "security": "tests/security/test_ff_api_security.py", 
            "performance": "tests/load/test_ff_api_performance.py",
            "use-cases": "tests/system/test_ff_api_use_cases.py"
        }
        
        if args.suite in suite_map:
            result = runner.run_test_suite(f"Specific Suite: {args.suite}", suite_map[args.suite])
            sys.exit(0 if result.get("success") else 1)
        else:
            print(f"Unknown test suite: {args.suite}")
            print(f"Available suites: {list(suite_map.keys())}")
            sys.exit(1)
    
    # Run all tests
    success = runner.run_all_tests(
        skip_slow=args.skip_slow,
        skip_performance=args.skip_performance
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()