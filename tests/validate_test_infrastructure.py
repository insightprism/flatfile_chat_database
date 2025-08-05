#!/usr/bin/env python3
# FF Chat API Test Infrastructure Validation
"""
Validates that the comprehensive API testing infrastructure is properly set up.
Checks test files, dependencies, and basic functionality.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import importlib.util

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestInfrastructureValidator:
    """Validates FF Chat API test infrastructure"""
    
    def __init__(self):
        self.validation_results = {}
        self.issues = []
        self.recommendations = []
    
    def validate_test_files(self) -> bool:
        """Validate that all test files exist and are properly structured"""
        print("🔍 Validating test files...")
        
        required_test_files = [
            "tests/conftest.py",
            "tests/api/test_ff_chat_api_core.py",
            "tests/security/test_ff_api_security.py",
            "tests/load/test_ff_api_performance.py",
            "tests/system/test_ff_api_use_cases.py",
            "tests/run_comprehensive_api_tests.py",
            "tests/validate_test_infrastructure.py",
            "pytest.ini"
        ]
        
        missing_files = []
        valid_files = []
        
        for test_file in required_test_files:
            file_path = project_root / test_file
            if file_path.exists():
                valid_files.append(test_file)
                
                # Check file has content
                if file_path.stat().st_size == 0:
                    self.issues.append(f"Test file {test_file} is empty")
                else:
                    # Basic syntax check
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'import pytest' in content or 'def test_' in content or test_file.endswith('.ini'):
                                print(f"  ✅ {test_file}")
                            else:
                                self.issues.append(f"Test file {test_file} may not be a valid test file")
                                print(f"  ⚠️  {test_file} (questionable content)")
                    except Exception as e:
                        self.issues.append(f"Error reading {test_file}: {e}")
                        print(f"  ❌ {test_file} (read error)")
            else:
                missing_files.append(test_file)
                print(f"  ❌ {test_file} (missing)")
        
        success = len(missing_files) == 0
        
        self.validation_results["test_files"] = {
            "success": success,
            "valid_files": len(valid_files),
            "missing_files": len(missing_files),
            "details": {
                "valid": valid_files,
                "missing": missing_files
            }
        }
        
        if missing_files:
            self.issues.extend([f"Missing test file: {f}" for f in missing_files])
        
        print(f"Test files validation: {'PASSED' if success else 'FAILED'}")
        return success
    
    def validate_test_dependencies(self) -> bool:
        """Validate that required test dependencies are available"""
        print("\n🔍 Validating test dependencies...")
        
        required_deps = [
            ("pytest", "Core testing framework"),
            ("asyncio", "Async support (builtin)"),
        ]
        
        optional_deps = [
            ("pytest_asyncio", "Async test support"),
            ("pytest_cov", "Coverage reporting"),
            ("pytest_xdist", "Parallel test execution"),
            ("fastapi", "API framework"),
            ("httpx", "HTTP client for testing")
        ]
        
        available_deps = []
        missing_deps = []
        
        # Check required dependencies
        for dep_name, description in required_deps:
            try:
                if dep_name == "asyncio":
                    import asyncio
                else:
                    __import__(dep_name)
                available_deps.append((dep_name, description, True))
                print(f"  ✅ {dep_name} - {description}")
            except ImportError:
                missing_deps.append((dep_name, description, True))
                print(f"  ❌ {dep_name} - {description} (REQUIRED)")
        
        # Check optional dependencies
        for dep_name, description in optional_deps:
            try:
                __import__(dep_name)
                available_deps.append((dep_name, description, False))
                print(f"  ✅ {dep_name} - {description} (optional)")
            except ImportError:
                missing_deps.append((dep_name, description, False))
                print(f"  ⚠️  {dep_name} - {description} (optional, not available)")
        
        # Check for FF Chat API components
        ff_components = [
            ("ff_chat_api", "Main API module"),
            ("ff_chat_auth", "Authentication module"),
            ("ff_chat_application", "Application module")
        ]
        
        for component, description in ff_components:
            component_path = project_root / f"{component}.py"
            if component_path.exists():
                available_deps.append((component, description, False))
                print(f"  ✅ {component} - {description}")
            else:
                missing_deps.append((component, description, False))
                print(f"  ⚠️  {component} - {description} (not found)")
        
        required_missing = [dep for dep in missing_deps if dep[2]]  # Required deps only
        success = len(required_missing) == 0
        
        self.validation_results["dependencies"] = {
            "success": success,
            "available": len(available_deps),
            "missing_required": len(required_missing),
            "missing_optional": len([dep for dep in missing_deps if not dep[2]]),
            "details": {
                "available": available_deps,
                "missing": missing_deps
            }
        }
        
        if required_missing:
            self.issues.extend([f"Missing required dependency: {dep[0]}" for dep in required_missing])
        
        if missing_deps and not required_missing:
            self.recommendations.append("Consider installing optional dependencies for enhanced test features")
        
        print(f"Dependencies validation: {'PASSED' if success else 'FAILED'}")
        return success
    
    def validate_test_structure(self) -> bool:
        """Validate test directory structure and organization"""
        print("\n🔍 Validating test structure...")
        
        required_dirs = [
            "tests",
            "tests/api",
            "tests/security", 
            "tests/load",
            "tests/system",
            "tests/fixtures"
        ]
        
        missing_dirs = []
        valid_dirs = []
        
        for test_dir in required_dirs:
            dir_path = project_root / test_dir
            if dir_path.exists() and dir_path.is_dir():
                valid_dirs.append(test_dir)
                print(f"  ✅ {test_dir}/")
            else:
                missing_dirs.append(test_dir)
                print(f"  ❌ {test_dir}/ (missing)")
        
        success = len(missing_dirs) == 0
        
        self.validation_results["structure"] = {
            "success": success,
            "valid_dirs": len(valid_dirs),
            "missing_dirs": len(missing_dirs),
            "details": {
                "valid": valid_dirs,
                "missing": missing_dirs
            }
        }
        
        if missing_dirs:
            self.issues.extend([f"Missing test directory: {d}" for d in missing_dirs])
        
        print(f"Test structure validation: {'PASSED' if success else 'FAILED'}")
        return success
    
    def validate_test_content(self) -> bool:
        """Validate test content and coverage"""
        print("\n🔍 Validating test content...")
        
        test_categories = {
            "API Core Tests": "tests/api/test_ff_chat_api_core.py",
            "Security Tests": "tests/security/test_ff_api_security.py",
            "Performance Tests": "tests/load/test_ff_api_performance.py",
            "Use Case Tests": "tests/system/test_ff_api_use_cases.py"
        }
        
        valid_categories = []
        invalid_categories = []
        
        for category, test_file in test_categories.items():
            file_path = project_root / test_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for test classes and functions
                    has_test_classes = "class Test" in content
                    has_test_functions = "def test_" in content or "async def test_" in content
                    has_proper_imports = "import pytest" in content
                    
                    if has_test_classes and has_test_functions and has_proper_imports:
                        valid_categories.append(category)
                        print(f"  ✅ {category}")
                    else:
                        invalid_categories.append(category)
                        issues = []
                        if not has_test_classes:
                            issues.append("no test classes")
                        if not has_test_functions:
                            issues.append("no test functions")
                        if not has_proper_imports:
                            issues.append("missing pytest import")
                        print(f"  ⚠️  {category} ({', '.join(issues)})")
                
                except Exception as e:
                    invalid_categories.append(category)
                    print(f"  ❌ {category} (read error: {e})")
            else:
                invalid_categories.append(category)
                print(f"  ❌ {category} (file missing)")
        
        success = len(invalid_categories) == 0
        
        self.validation_results["content"] = {
            "success": success,
            "valid_categories": len(valid_categories),
            "invalid_categories": len(invalid_categories),
            "details": {
                "valid": valid_categories,
                "invalid": invalid_categories
            }
        }
        
        if invalid_categories:
            self.issues.extend([f"Invalid test category: {c}" for c in invalid_categories])
        
        print(f"Test content validation: {'PASSED' if success else 'FAILED'}")
        return success
    
    def validate_use_case_coverage(self) -> bool:
        """Validate that all 22 use cases are covered in tests"""
        print("\n🔍 Validating use case coverage...")
        
        expected_use_cases = [
            "basic_chat", "memory_chat", "rag_chat", "multimodal_chat",
            "multi_ai_panel", "personal_assistant", "translation_chat",
            "code_assistant", "creative_writing", "research_assistant",
            "educational_tutor", "business_advisor", "ai_debate",
            "prompt_sandbox", "document_qa", "workflow_automation",
            "data_analysis", "content_creation", "technical_consulting",
            "strategic_planning", "crisis_management", "scene_critic"
        ]
        
        use_case_test_file = project_root / "tests/system/test_ff_api_use_cases.py"
        
        if not use_case_test_file.exists():
            print("  ❌ Use case test file missing")
            self.issues.append("Use case test file missing")
            return False
        
        try:
            with open(use_case_test_file, 'r') as f:
                content = f.read()
            
            covered_use_cases = []
            missing_use_cases = []
            
            for use_case in expected_use_cases:
                if use_case in content:
                    covered_use_cases.append(use_case)
                else:
                    missing_use_cases.append(use_case)
            
            coverage_percent = (len(covered_use_cases) / len(expected_use_cases)) * 100
            
            print(f"  Use cases covered: {len(covered_use_cases)}/{len(expected_use_cases)} ({coverage_percent:.1f}%)")
            
            if missing_use_cases:
                print(f"  Missing use cases: {missing_use_cases[:5]}{'...' if len(missing_use_cases) > 5 else ''}")
            
            success = coverage_percent >= 90  # Allow 90% coverage
            
            self.validation_results["use_case_coverage"] = {
                "success": success,
                "covered": len(covered_use_cases),
                "total": len(expected_use_cases),
                "coverage_percent": coverage_percent,
                "details": {
                    "covered": covered_use_cases,
                    "missing": missing_use_cases
                }
            }
            
            if not success:
                self.issues.append(f"Use case coverage {coverage_percent:.1f}% below 90% threshold")
            
            print(f"Use case coverage validation: {'PASSED' if success else 'FAILED'}")
            return success
            
        except Exception as e:
            print(f"  ❌ Error validating use case coverage: {e}")
            self.issues.append(f"Error validating use case coverage: {e}")
            return False
    
    def generate_validation_report(self) -> bool:
        """Generate comprehensive validation report"""
        print(f"\n{'='*70}")
        print("FF CHAT API TEST INFRASTRUCTURE VALIDATION REPORT")
        print(f"{'='*70}")
        
        # Summary statistics
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for result in self.validation_results.values() if result.get("success"))
        failed_validations = total_validations - passed_validations
        
        print(f"\nValidation Summary:")
        print(f"  Total Validations: {total_validations}")
        print(f"  Passed: {passed_validations}")
        print(f"  Failed: {failed_validations}")
        
        if total_validations > 0:
            success_rate = (passed_validations / total_validations) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for validation_name, result in self.validation_results.items():
            status = "PASSED" if result.get("success") else "FAILED"
            print(f"  {validation_name.replace('_', ' ').title()}: {status}")
        
        # Issues
        if self.issues:
            print(f"\nIssues Found ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  ❌ {issue}")
        
        # Recommendations
        if self.recommendations:
            print(f"\nRecommendations ({len(self.recommendations)}):")
            for rec in self.recommendations:
                print(f"  💡 {rec}")
        
        # Overall status
        overall_success = failed_validations == 0
        
        print(f"\n{'='*70}")
        if overall_success:
            print("✅ TEST INFRASTRUCTURE VALIDATION: PASSED")
            print("🚀 All test infrastructure components are properly configured")
            print("✅ Ready for comprehensive API testing")
        else:
            print("❌ TEST INFRASTRUCTURE VALIDATION: FAILED")
            print(f"⚠️  {failed_validations} validation(s) failed")
            print("🔧 Fix issues before running comprehensive tests")
        
        print(f"{'='*70}")
        
        return overall_success
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting FF Chat API Test Infrastructure Validation")
        
        validations = [
            ("test_files", self.validate_test_files),
            ("dependencies", self.validate_test_dependencies),
            ("structure", self.validate_test_structure),
            ("content", self.validate_test_content),
            ("use_case_coverage", self.validate_use_case_coverage)
        ]
        
        for validation_name, validation_func in validations:
            try:
                success = validation_func()
                if not success:
                    self.issues.append(f"{validation_name} validation failed")
            except Exception as e:
                print(f"❌ Error during {validation_name} validation: {e}")
                self.validation_results[validation_name] = {"success": False, "error": str(e)}
                self.issues.append(f"{validation_name} validation error: {e}")
        
        return self.generate_validation_report()

def main():
    """Main validation entry point"""
    validator = TestInfrastructureValidator()
    success = validator.run_all_validations()
    
    if success:
        print("\n🎯 Next steps:")
        print("  1. Run quick validation: python tests/run_comprehensive_api_tests.py --quick")
        print("  2. Run full test suite: python tests/run_comprehensive_api_tests.py")
        print("  3. Run specific suite: python tests/run_comprehensive_api_tests.py --suite core")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()