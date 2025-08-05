"""
FF Tools Sandbox - Secure Tool Execution Environment

Provides sandboxed execution environment for tools to prevent
security vulnerabilities and system compromise.
"""

import asyncio
import os
import sys
import tempfile
import shutil
import subprocess
import time
import signal
import resource
import psutil
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_tools_config import FFToolsSecurityConfigDTO, FFToolSecurityLevel

logger = get_logger(__name__)


class FFSandboxStatus(Enum):
    """Status of sandbox execution"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    KILLED = "killed"


@dataclass
class FFSandboxLimits:
    """Resource limits for sandbox execution"""
    max_memory_mb: int = 100
    max_cpu_time_seconds: int = 10
    max_wall_time_seconds: int = 30
    max_file_descriptors: int = 100
    max_processes: int = 5
    max_file_size_mb: int = 10
    max_output_size_kb: int = 1024


@dataclass
class FFSandboxEnvironment:
    """Sandbox environment configuration"""
    sandbox_id: str
    working_directory: str
    temp_directory: str
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    limits: FFSandboxLimits = field(default_factory=FFSandboxLimits)
    network_access: bool = False
    internet_access: bool = False
    allowed_domains: List[str] = field(default_factory=list)


@dataclass
class FFSandboxResult:
    """Result of sandbox execution"""
    sandbox_id: str
    status: FFSandboxStatus
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    error_message: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)


class FFToolsSandbox:
    """
    Secure sandbox for tool execution using system-level isolation.
    
    Provides resource limiting, file system isolation, and network
    restrictions to safely execute external tools.
    """
    
    def __init__(self, security_config: FFToolsSecurityConfigDTO):
        """
        Initialize FF tools sandbox.
        
        Args:
            security_config: Security configuration for sandbox
        """
        self.security_config = security_config
        self.logger = get_logger(__name__)
        
        # Sandbox management
        self.active_sandboxes: Dict[str, FFSandboxEnvironment] = {}
        self.sandbox_counter = 0
        
        # Default limits from security config
        self.default_limits = FFSandboxLimits(
            max_memory_mb=security_config.max_memory_mb,
            max_cpu_time_seconds=security_config.max_cpu_time,
            max_wall_time_seconds=security_config.sandbox_timeout,
            max_processes=security_config.max_processes
        )
        
        # Validate sandbox capabilities
        self._validate_sandbox_capabilities()
    
    def _validate_sandbox_capabilities(self) -> None:
        """Validate that sandbox features are available on the system"""
        try:
            # Check if we can set resource limits
            resource.getrlimit(resource.RLIMIT_AS)
            
            # Check if we can create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test.txt"
                test_file.write_text("test")
                assert test_file.exists()
            
            self.logger.info("Sandbox capabilities validated successfully")
            
        except Exception as e:
            self.logger.error(f"Sandbox capabilities validation failed: {e}")
            raise RuntimeError(f"Sandbox not available: {e}")
    
    async def create_sandbox(self, 
                           sandbox_id: Optional[str] = None,
                           limits: Optional[FFSandboxLimits] = None) -> FFSandboxEnvironment:
        """
        Create a new sandbox environment.
        
        Args:
            sandbox_id: Optional custom sandbox ID
            limits: Optional custom resource limits
            
        Returns:
            FFSandboxEnvironment: Created sandbox environment
        """
        try:
            # Generate sandbox ID
            if not sandbox_id:
                self.sandbox_counter += 1
                sandbox_id = f"ff_sandbox_{self.sandbox_counter}_{int(time.time())}"
            
            # Use provided limits or defaults
            sandbox_limits = limits or self.default_limits
            
            # Create temporary directories
            temp_dir = tempfile.mkdtemp(prefix=f"ff_sandbox_{sandbox_id}_")
            working_dir = os.path.join(temp_dir, "work")
            os.makedirs(working_dir, mode=0o700)
            
            # Set up allowed and blocked paths
            allowed_paths = [working_dir, temp_dir]
            blocked_paths = list(self.security_config.blocked_file_paths)
            
            # Create sandbox environment
            sandbox_env = FFSandboxEnvironment(
                sandbox_id=sandbox_id,
                working_directory=working_dir,
                temp_directory=temp_dir,
                allowed_paths=allowed_paths,
                blocked_paths=blocked_paths,
                limits=sandbox_limits,
                network_access=False,  # Disabled by default
                internet_access=False,  # Disabled by default
                allowed_domains=list(self.security_config.allowed_domains)
            )
            
            # Register sandbox
            self.active_sandboxes[sandbox_id] = sandbox_env
            
            self.logger.info(f"Created sandbox {sandbox_id} at {temp_dir}")
            return sandbox_env
            
        except Exception as e:
            self.logger.error(f"Failed to create sandbox: {e}")
            raise
    
    async def execute_command(self,
                            sandbox_env: FFSandboxEnvironment,
                            command: str,
                            args: List[str] = None,
                            input_data: Optional[str] = None,
                            security_level: FFToolSecurityLevel = FFToolSecurityLevel.RESTRICTED) -> FFSandboxResult:
        """
        Execute a command in the sandbox environment.
        
        Args:
            sandbox_env: Sandbox environment to use
            command: Command to execute
            args: Command arguments
            input_data: Optional input data for the command
            security_level: Security level for execution
            
        Returns:
            FFSandboxResult: Execution result
        """
        start_time = time.time()
        result = FFSandboxResult(
            sandbox_id=sandbox_env.sandbox_id,
            status=FFSandboxStatus.CREATED,
            exit_code=-1
        )
        
        try:
            # Validate command security
            if not self._validate_command_security(command, security_level):
                raise SecurityError(f"Command '{command}' not allowed at security level {security_level}")
            
            # Prepare command execution
            cmd_args = [command] + (args or [])
            
            # Set up environment variables
            env = os.environ.copy()
            env.update(sandbox_env.environment_variables)
            
            # Remove potentially dangerous environment variables
            dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
            for var in dangerous_vars:
                env.pop(var, None)
            
            # Prepare execution with resource limits
            result.status = FFSandboxStatus.RUNNING
            
            # Execute command with limits
            process_result = await self._execute_with_limits(
                cmd_args,
                cwd=sandbox_env.working_directory,
                env=env,
                input_data=input_data,
                limits=sandbox_env.limits
            )
            
            # Process results
            result.exit_code = process_result['returncode']
            result.stdout = process_result['stdout']
            result.stderr = process_result['stderr']
            result.execution_time_seconds = time.time() - start_time
            result.memory_used_mb = process_result.get('memory_used_mb', 0.0)
            
            # Check execution status
            if process_result['timeout']:
                result.status = FFSandboxStatus.TIMEOUT
                result.error_message = "Execution timed out"
            elif result.exit_code == 0:
                result.status = FFSandboxStatus.COMPLETED
            else:
                result.status = FFSandboxStatus.ERROR
                result.error_message = f"Command failed with exit code {result.exit_code}"
            
            # Scan for created/modified files
            result.files_created, result.files_modified = await self._scan_file_changes(sandbox_env)
            
            self.logger.info(f"Executed command in sandbox {sandbox_env.sandbox_id}: {result.status}")
            return result
            
        except Exception as e:
            result.status = FFSandboxStatus.ERROR
            result.error_message = str(e)
            result.execution_time_seconds = time.time() - start_time
            
            self.logger.error(f"Sandbox execution failed: {e}")
            return result
    
    async def _execute_with_limits(self,
                                 cmd_args: List[str],
                                 cwd: str,
                                 env: Dict[str, str],
                                 input_data: Optional[str],
                                 limits: FFSandboxLimits) -> Dict[str, Any]:
        """Execute command with resource limits"""
        
        def preexec_function():
            """Set resource limits before execution"""
            try:
                # Memory limit
                if limits.max_memory_mb > 0:
                    max_memory = limits.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
                
                # CPU time limit
                if limits.max_cpu_time_seconds > 0:
                    resource.setrlimit(resource.RLIMIT_CPU, (limits.max_cpu_time_seconds, limits.max_cpu_time_seconds))
                
                # File descriptor limit
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_file_descriptors, limits.max_file_descriptors))
                
                # Process limit
                resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
                
                # File size limit
                if limits.max_file_size_mb > 0:
                    max_file_size = limits.max_file_size_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
                
            except Exception as e:
                # Log but don't fail - some limits might not be available
                logger.warning(f"Could not set resource limit: {e}")
        
        try:
            # Start process with limits
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_function
            )
            
            # Monitor process execution
            timeout_occurred = False
            memory_used_mb = 0.0
            
            try:
                # Execute with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode() if input_data else None),
                    timeout=limits.max_wall_time_seconds
                )
                
                # Get memory usage if process is still accessible
                try:
                    proc_info = psutil.Process(process.pid)
                    memory_used_mb = proc_info.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
            except asyncio.TimeoutError:
                timeout_occurred = True
                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
                stdout, stderr = b"", b"[TIMEOUT] Process killed due to timeout"
            
            # Truncate output if too large
            max_output_bytes = limits.max_output_size_kb * 1024
            if len(stdout) > max_output_bytes:
                stdout = stdout[:max_output_bytes] + b"\n[OUTPUT TRUNCATED]"
            if len(stderr) > max_output_bytes:
                stderr = stderr[:max_output_bytes] + b"\n[OUTPUT TRUNCATED]"
            
            return {
                'returncode': process.returncode if not timeout_occurred else -1,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'timeout': timeout_occurred,
                'memory_used_mb': memory_used_mb
            }
            
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'timeout': False,
                'memory_used_mb': 0.0
            }
    
    def _validate_command_security(self, command: str, security_level: FFToolSecurityLevel) -> bool:
        """Validate that a command is allowed at the given security level"""
        try:
            # Extract base command name
            base_command = command.split()[0] if ' ' in command else command
            base_command = os.path.basename(base_command)
            
            # Check blocked commands
            if base_command in self.security_config.blocked_commands:
                return False
            
            # Check allowed commands based on security level
            if security_level == FFToolSecurityLevel.READ_ONLY:
                # Only safe read-only commands
                safe_commands = {'cat', 'head', 'tail', 'grep', 'find', 'ls', 'wc', 'file', 'stat'}
                return base_command in safe_commands
            
            elif security_level == FFToolSecurityLevel.RESTRICTED:
                # Allowed commands for restricted level
                return base_command in self.security_config.allowed_commands
            
            elif security_level == FFToolSecurityLevel.SANDBOXED:
                # Most commands allowed in sandbox
                return base_command not in self.security_config.blocked_commands
            
            elif security_level == FFToolSecurityLevel.TRUSTED:
                # All commands allowed (use with extreme caution)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Command validation error: {e}")
            return False
    
    async def _scan_file_changes(self, sandbox_env: FFSandboxEnvironment) -> tuple[List[str], List[str]]:
        """Scan for files created or modified in the sandbox"""
        try:
            created_files = []
            modified_files = []
            
            # Scan working directory
            work_path = Path(sandbox_env.working_directory)
            if work_path.exists():
                for file_path in work_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = str(file_path.relative_to(work_path))
                        created_files.append(relative_path)
            
            return created_files, modified_files
            
        except Exception as e:
            self.logger.error(f"File scan error: {e}")
            return [], []
    
    async def cleanup_sandbox(self, sandbox_id: str) -> bool:
        """
        Clean up and remove a sandbox environment.
        
        Args:
            sandbox_id: ID of sandbox to cleanup
            
        Returns:
            bool: True if cleanup successful
        """
        try:
            if sandbox_id not in self.active_sandboxes:
                self.logger.warning(f"Sandbox {sandbox_id} not found for cleanup")
                return False
            
            sandbox_env = self.active_sandboxes[sandbox_id]
            
            # Remove temporary directory
            if os.path.exists(sandbox_env.temp_directory):
                shutil.rmtree(sandbox_env.temp_directory, ignore_errors=True)
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            self.logger.info(f"Cleaned up sandbox {sandbox_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
            return False
    
    async def cleanup_all_sandboxes(self) -> int:
        """
        Clean up all active sandbox environments.
        
        Returns:
            int: Number of sandboxes cleaned up
        """
        cleaned_count = 0
        sandbox_ids = list(self.active_sandboxes.keys())
        
        for sandbox_id in sandbox_ids:
            if await self.cleanup_sandbox(sandbox_id):
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} sandboxes")
        return cleaned_count
    
    def get_sandbox_info(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a sandbox"""
        if sandbox_id not in self.active_sandboxes:
            return None
        
        sandbox_env = self.active_sandboxes[sandbox_id]
        return {
            "sandbox_id": sandbox_env.sandbox_id,
            "working_directory": sandbox_env.working_directory,
            "temp_directory": sandbox_env.temp_directory,
            "limits": {
                "max_memory_mb": sandbox_env.limits.max_memory_mb,
                "max_cpu_time": sandbox_env.limits.max_cpu_time_seconds,
                "max_wall_time": sandbox_env.limits.max_wall_time_seconds
            },
            "network_access": sandbox_env.network_access,
            "created_at": os.path.getctime(sandbox_env.temp_directory)
        }
    
    def list_active_sandboxes(self) -> List[str]:
        """List all active sandbox IDs"""
        return list(self.active_sandboxes.keys())
    
    @contextmanager
    def sandbox_context(self, limits: Optional[FFSandboxLimits] = None):
        """Context manager for automatic sandbox cleanup"""
        sandbox_env = None
        try:
            # Create sandbox
            loop = asyncio.get_event_loop()
            sandbox_env = loop.run_until_complete(self.create_sandbox(limits=limits))
            yield sandbox_env
        finally:
            # Cleanup sandbox
            if sandbox_env:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.cleanup_sandbox(sandbox_env.sandbox_id))


class SecurityError(Exception):
    """Security-related error in sandbox execution"""
    pass