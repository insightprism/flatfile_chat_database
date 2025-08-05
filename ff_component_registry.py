"""
FF Component Registry - Phase 2 Implementation

Manages component lifecycle and dependencies using existing FF dependency
injection manager. Provides dynamic component loading and configuration.
"""

import asyncio
import importlib
import inspect
import time
from typing import Dict, Any, List, Optional, Type, Set
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# Import existing FF infrastructure
from ff_dependency_injection_manager import ff_get_container, ff_register_service, ff_get_service
from ff_class_configs.ff_component_registry_config import (
    FFComponentRegistryConfigDTO, FFComponentRegistrationConfigDTO,
    FFComponentEnvironmentConfigDTO, FFComponentLoadingStrategy, FFComponentLifecycle
)
from ff_protocols.ff_chat_component_protocol import (
    FFComponentRegistryProtocol, FFChatComponentProtocol, 
    get_required_components_for_use_case, get_use_cases_for_component
)
from ff_utils.ff_logging import get_logger


@dataclass
class FFComponentRegistration:
    """Component registration information"""
    name: str
    component_class: Type[FFChatComponentProtocol]
    config_class: Type
    config: Any
    dependencies: List[str]
    ff_manager_dependencies: List[str]
    priority: int
    lifecycle: str
    use_cases: List[str]
    capabilities: List[str]
    registered_at: datetime
    instance: Optional[FFChatComponentProtocol] = None
    initialized: bool = False
    load_count: int = 0
    last_accessed: Optional[datetime] = None


class FFComponentRegistry(FFComponentRegistryProtocol):
    """
    FF Component Registry managing chat component lifecycle.
    
    Integrates with existing FF dependency injection manager to provide
    dynamic component loading, dependency resolution, and lifecycle management.
    """
    
    def __init__(self, config: FFComponentRegistryConfigDTO):
        """
        Initialize FF Component Registry.
        
        Args:
            config: Component registry configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Component management
        self._registered_components: Dict[str, FFComponentRegistration] = {}
        self._component_instances: Dict[str, FFChatComponentProtocol] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        
        # FF dependency injection integration
        self._ff_container = None
        self._ff_managers: Dict[str, Any] = {}
        
        # Lifecycle management
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        self._health_check_results: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_collection_task: Optional[asyncio.Task] = None
        
        # Registry state
        self._initialized = False
        self._loading_strategy = FFComponentLoadingStrategy(config.loading_strategy)
        
        # Metrics and statistics
        self._registry_stats = {
            "total_registered": 0,
            "total_loaded": 0,
            "total_failed_loads": 0,
            "average_load_time": 0.0,
            "health_check_runs": 0,
            "last_health_check": None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize component registry with FF dependency injection.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing FF Component Registry...")
            
            # Get FF dependency injection container
            if self.config.use_ff_dependency_injection:
                self._ff_container = ff_get_container()
                if not self._ff_container:
                    raise RuntimeError("Failed to get FF dependency injection container")
            
            # Initialize FF managers
            await self._initialize_ff_managers()
            
            # Auto-discover components if enabled
            if self.config.auto_discovery_enabled:
                await self._auto_discover_components()
            
            # Register built-in components
            await self._register_builtin_components()
            
            # Load components based on strategy
            if self._loading_strategy == FFComponentLoadingStrategy.EAGER:
                await self._load_all_components()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            self.logger.info(f"FF Component Registry initialized with {len(self._registered_components)} registered components")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Component Registry: {e}")
            return False
    
    async def _initialize_ff_managers(self) -> None:
        """Initialize FF manager dependencies"""
        try:
            # Get FF managers from dependency injection
            ff_manager_names = [
                "ff_storage", "ff_search", "ff_vector", "ff_panel", 
                "ff_document", "ff_dependency_injection"
            ]
            
            for manager_name in ff_manager_names:
                try:
                    manager_instance = ff_get_service(manager_name)
                    if manager_instance:
                        self._ff_managers[manager_name] = manager_instance
                        self.logger.debug(f"Found FF manager: {manager_name}")
                except Exception as e:
                    self.logger.debug(f"FF manager {manager_name} not available: {e}")
            
            if not self._ff_managers:
                self.logger.warning("No FF managers found - components may fail to initialize")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF managers: {e}")
    
    async def _auto_discover_components(self) -> None:
        """Auto-discover components in specified packages"""
        try:
            discovered_count = 0
            
            for package_name in self.config.component_scan_packages:
                try:
                    # Scan package for component classes
                    components = await self._scan_package_for_components(package_name)
                    discovered_count += len(components)
                    
                    # Register discovered components
                    for component_info in components:
                        await self._register_discovered_component(component_info)
                        
                except Exception as e:
                    self.logger.error(f"Failed to scan package {package_name}: {e}")
            
            self.logger.info(f"Auto-discovered {discovered_count} components")
            
        except Exception as e:
            self.logger.error(f"Auto-discovery failed: {e}")
    
    async def _scan_package_for_components(self, package_name: str) -> List[Dict[str, Any]]:
        """Scan package for component classes"""
        components = []
        
        try:
            # Import package
            package = importlib.import_module(package_name)
            package_path = Path(package.__file__).parent
            
            # Scan Python files in package
            for py_file in package_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = f"{package_name}.{py_file.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Look for component classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (hasattr(obj, '__bases__') and 
                            any(issubclass(base, FFChatComponentProtocol) for base in obj.__bases__ if base != FFChatComponentProtocol)):
                            
                            component_info = {
                                "name": name.lower().replace("component", "").replace("ff", ""),
                                "class": obj,
                                "module": module_name,
                                "file": str(py_file)
                            }
                            components.append(component_info)
                            
                except Exception as e:
                    self.logger.debug(f"Failed to scan module {module_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to scan package {package_name}: {e}")
        
        return components
    
    async def _register_discovered_component(self, component_info: Dict[str, Any]) -> None:
        """Register auto-discovered component"""
        try:
            # Create basic registration config
            registration_config = FFComponentRegistrationConfigDTO(
                component_name=component_info["name"],
                component_class_path=f"{component_info['module']}.{component_info['class'].__name__}",
                config_class_path="",  # Would need to be inferred or specified
                ff_manager_dependencies=["ff_storage"]  # Default dependency
            )
            
            # Register the component
            # Note: This is simplified - full implementation would need better config detection
            
        except Exception as e:
            self.logger.error(f"Failed to register discovered component {component_info['name']}: {e}")
    
    async def _register_builtin_components(self) -> None:
        """Register built-in Phase 2 components"""
        try:
            # Import Phase 2 components
            from ff_text_chat_component import FFTextChatComponent
            from ff_class_configs.ff_text_chat_config import FFTextChatConfigDTO
            
            from ff_memory_component import FFMemoryComponent
            from ff_class_configs.ff_memory_config import FFMemoryConfigDTO
            
            from ff_multi_agent_component import FFMultiAgentComponent
            from ff_class_configs.ff_multi_agent_config import FFMultiAgentConfigDTO
            
            # Register text chat component
            self.register_component(
                name="text_chat",
                component_class=FFTextChatComponent,
                config_class=FFTextChatConfigDTO,
                config=FFTextChatConfigDTO(),
                dependencies=[],
                ff_manager_dependencies=["ff_storage", "ff_search"],
                priority=100
            )
            
            # Register memory component
            self.register_component(
                name="memory",
                component_class=FFMemoryComponent,
                config_class=FFMemoryConfigDTO,
                config=FFMemoryConfigDTO(),
                dependencies=[],
                ff_manager_dependencies=["ff_storage", "ff_vector", "ff_search"],
                priority=90
            )
            
            # Register multi-agent component
            self.register_component(
                name="multi_agent",
                component_class=FFMultiAgentComponent,
                config_class=FFMultiAgentConfigDTO,
                config=FFMultiAgentConfigDTO(),
                dependencies=[],
                ff_manager_dependencies=["ff_storage", "ff_panel", "ff_search", "ff_vector"],
                priority=80
            )
            
            self.logger.info("Registered built-in Phase 2 components")
            
        except Exception as e:
            self.logger.error(f"Failed to register built-in components: {e}")
    
    def register_component(self,
                           name: str,
                           component_class: Type[FFChatComponentProtocol],
                           config_class: Type,
                           config: Any,
                           dependencies: List[str],
                           ff_manager_dependencies: List[str] = None,
                           priority: int = 100) -> None:
        """
        Register FF chat component.
        
        Args:
            name: Component identifier
            component_class: Component class
            config_class: Configuration class  
            config: Configuration instance
            dependencies: List of component dependencies
            ff_manager_dependencies: List of FF service dependencies
            priority: Loading priority
        """
        try:
            if name in self._registered_components:
                self.logger.warning(f"Component {name} already registered, updating...")
            
            # Get component capabilities and use cases
            temp_instance = component_class(config)
            if hasattr(temp_instance, 'component_info'):
                component_info = temp_instance.component_info
                capabilities = component_info.get("capabilities", [])
                use_cases = component_info.get("use_cases", [])
            else:
                capabilities = []
                use_cases = []
            
            # Create registration
            registration = FFComponentRegistration(
                name=name,
                component_class=component_class,
                config_class=config_class,
                config=config,
                dependencies=dependencies,
                ff_manager_dependencies=ff_manager_dependencies or [],
                priority=priority,
                lifecycle=FFComponentLifecycle.SINGLETON.value,  # Default to singleton
                use_cases=use_cases,
                capabilities=capabilities,
                registered_at=datetime.now()
            )
            
            self._registered_components[name] = registration
            self._loading_locks[name] = asyncio.Lock()
            
            # Update dependency graph
            self._dependency_graph[name] = set(dependencies)
            
            # Register with FF dependency injection if enabled
            if self.config.use_ff_dependency_injection:
                ff_register_service(f"ff_chat_component_{name}", component_class)
            
            self._registry_stats["total_registered"] += 1
            self.logger.info(f"Registered component: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register component {name}: {e}")
            raise
    
    async def load_components(self, component_names: List[str]) -> Dict[str, FFChatComponentProtocol]:
        """
        Load and initialize specified components.
        
        Args:
            component_names: List of component names to load
            
        Returns:
            Dictionary of loaded component instances
        """
        try:
            loaded_components = {}
            
            # Resolve dependency order
            load_order = await self._resolve_dependency_order(component_names)
            
            # Load components in dependency order
            for name in load_order:
                component = await self._load_single_component(name)
                if component:
                    loaded_components[name] = component
                else:
                    self.logger.error(f"Failed to load component: {name}")
                    if not self.config.enable_component_fallbacks:
                        raise RuntimeError(f"Critical component {name} failed to load")
            
            self.logger.info(f"Successfully loaded {len(loaded_components)} components")
            return loaded_components
            
        except Exception as e:
            self.logger.error(f"Failed to load components: {e}")
            return {}
    
    async def _load_single_component(self, name: str) -> Optional[FFChatComponentProtocol]:
        """Load and initialize a single component"""
        if name not in self._registered_components:
            self.logger.error(f"Component {name} not registered")
            return None
        
        registration = self._registered_components[name]
        
        # Check if already loaded and lifecycle allows reuse
        if (registration.instance and registration.initialized and 
            registration.lifecycle == FFComponentLifecycle.SINGLETON.value):
            registration.last_accessed = datetime.now()
            return registration.instance
        
        # Acquire loading lock
        async with self._loading_locks[name]:
            start_time = time.time()
            
            try:
                self.logger.debug(f"Loading component: {name}")
                
                # Create component instance
                component_instance = registration.component_class(registration.config)
                
                # Prepare dependencies
                dependencies = await self._prepare_component_dependencies(registration)
                
                # Initialize component
                success = await component_instance.initialize(dependencies)
                if not success:
                    raise RuntimeError(f"Component {name} initialization failed")
                
                # Store instance
                registration.instance = component_instance
                registration.initialized = True
                registration.load_count += 1
                registration.last_accessed = datetime.now()
                
                self._component_instances[name] = component_instance
                
                # Update statistics
                load_time = time.time() - start_time
                self._registry_stats["total_loaded"] += 1
                
                current_avg = self._registry_stats["average_load_time"]
                total_loaded = self._registry_stats["total_loaded"]
                self._registry_stats["average_load_time"] = ((current_avg * (total_loaded - 1)) + load_time) / total_loaded
                
                self.logger.info(f"Successfully loaded component {name} in {load_time:.2f}s")
                return component_instance
                
            except Exception as e:
                self.logger.error(f"Failed to load component {name}: {e}")
                self._registry_stats["total_failed_loads"] += 1
                return None
    
    async def _prepare_component_dependencies(self, registration: FFComponentRegistration) -> Dict[str, Any]:
        """Prepare dependencies for component initialization"""
        dependencies = {}
        
        # Add FF manager dependencies
        for ff_manager_name in registration.ff_manager_dependencies:
            if ff_manager_name in self._ff_managers:
                dependencies[ff_manager_name] = self._ff_managers[ff_manager_name]
            else:
                self.logger.warning(f"FF manager {ff_manager_name} not available for component {registration.name}")
        
        # Add component dependencies
        for dep_name in registration.dependencies:
            if dep_name in self._component_instances:
                dependencies[f"component_{dep_name}"] = self._component_instances[dep_name]
            else:
                # Try to load dependency
                dep_component = await self._load_single_component(dep_name)
                if dep_component:
                    dependencies[f"component_{dep_name}"] = dep_component
                else:
                    self.logger.warning(f"Component dependency {dep_name} not available for {registration.name}")
        
        return dependencies
    
    async def _resolve_dependency_order(self, component_names: List[str]) -> List[str]:
        """Resolve component loading order based on dependencies"""
        try:
            # Topological sort implementation
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(name: str):
                if name in temp_visited:
                    raise RuntimeError(f"Circular dependency detected involving component {name}")
                
                if name not in visited:
                    temp_visited.add(name)
                    
                    # Visit dependencies first
                    for dep in self._dependency_graph.get(name, set()):
                        if dep in component_names:  # Only consider components we're loading
                            visit(dep)
                    
                    temp_visited.remove(name)
                    visited.add(name)
                    result.append(name)
            
            # Visit all requested components
            for name in component_names:
                if name not in visited:
                    visit(name)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resolve dependency order: {e}")
            # Fallback to priority-based ordering
            return sorted(component_names, 
                         key=lambda n: self._registered_components.get(n, FFComponentRegistration(
                             name=n, component_class=None, config_class=None, config=None,
                             dependencies=[], ff_manager_dependencies=[], priority=999,
                             lifecycle="", use_cases=[], capabilities=[], registered_at=datetime.now()
                         )).priority)
    
    async def _load_all_components(self) -> None:
        """Load all registered components (eager loading)"""
        try:
            all_component_names = list(self._registered_components.keys())
            await self.load_components(all_component_names)
            
        except Exception as e:
            self.logger.error(f"Failed to load all components: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Start health check task if enabled
        if self.config.component_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start metrics collection task if enabled
        if self.config.enable_component_metrics:
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        self.logger.debug("Background tasks started")
    
    async def _health_check_loop(self) -> None:
        """Background health check task loop"""
        while self._initialized:
            try:
                await self._perform_health_checks()
                self._registry_stats["health_check_runs"] += 1
                self._registry_stats["last_health_check"] = datetime.now().isoformat()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check task: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection task loop"""
        while self._initialized:
            try:
                await self._collect_component_metrics()
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection task: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all loaded components"""
        try:
            for name, component in self._component_instances.items():
                try:
                    # Basic health check - ensure component is still initialized
                    health_status = {
                        "component": name,
                        "healthy": True,
                        "last_checked": datetime.now().isoformat(),
                        "error": None
                    }
                    
                    # Check if component has custom health check
                    if hasattr(component, 'health_check'):
                        health_result = await component.health_check()
                        health_status["healthy"] = health_result.get("healthy", True)
                        health_status["error"] = health_result.get("error")
                    
                    self._health_check_results[name] = health_status
                    
                except Exception as e:
                    self._health_check_results[name] = {
                        "component": name,
                        "healthy": False,
                        "last_checked": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    self.logger.warning(f"Health check failed for component {name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    async def _collect_component_metrics(self) -> None:
        """Collect metrics from components"""
        try:
            for name, component in self._component_instances.items():
                try:
                    # Collect metrics if component supports it
                    if hasattr(component, 'get_metrics'):
                        metrics = await component.get_metrics()
                        # Store or process metrics as needed
                        self.logger.debug(f"Collected metrics for component {name}")
                        
                except Exception as e:
                    self.logger.debug(f"Failed to collect metrics for component {name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting component metrics: {e}")
    
    # Public interface methods
    
    def list_components(self) -> List[str]:
        """Get list of registered component names"""
        return list(self._registered_components.keys())
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered component"""
        if name not in self._registered_components:
            return None
        
        registration = self._registered_components[name]
        
        return {
            "name": name,
            "class_name": registration.component_class.__name__,
            "dependencies": registration.dependencies,
            "ff_manager_dependencies": registration.ff_manager_dependencies,
            "priority": registration.priority,
            "lifecycle": registration.lifecycle,
            "use_cases": registration.use_cases,
            "capabilities": registration.capabilities,
            "registered_at": registration.registered_at.isoformat(),
            "initialized": registration.initialized,
            "load_count": registration.load_count,
            "last_accessed": registration.last_accessed.isoformat() if registration.last_accessed else None
        }
    
    def get_components_for_use_case(self, use_case: str) -> List[str]:
        """Get components that support a specific use case"""
        components = []
        
        for name, registration in self._registered_components.items():
            if use_case in registration.use_cases:
                components.append(name)
        
        return components
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            **self._registry_stats,
            "registered_components": len(self._registered_components),
            "loaded_components": len(self._component_instances),
            "active_locks": len(self._loading_locks),
            "health_check_results": self._health_check_results.copy()
        }
    
    async def shutdown(self) -> None:
        """Shutdown component registry and cleanup resources"""
        try:
            self.logger.info("Shutting down FF Component Registry...")
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._metrics_collection_task:
                self._metrics_collection_task.cancel()
            
            # Cleanup all loaded components
            for name, component in self._component_instances.items():
                try:
                    await component.cleanup()
                    self.logger.debug(f"Cleaned up component: {name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up component {name}: {e}")
            
            # Clear registry state
            self._registered_components.clear()
            self._component_instances.clear()
            self._dependency_graph.clear()
            self._loading_locks.clear()
            self._health_check_results.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("FF Component Registry shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during FF Component Registry shutdown: {e}")
            raise