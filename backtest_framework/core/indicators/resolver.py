"""
Dependency resolver for managing indicator computation order.
"""
from typing import Dict, List, Set
from collections import defaultdict
from backtest_framework.core.indicators.registry import IndicatorRegistry

# Define base data columns that should be present in input data
BASE_DATA_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

class MissingDataError(Exception):
    """Exception raised when required data is missing."""
    pass

class CircularDependencyError(Exception):
    """Exception raised when circular dependencies are detected."""
    pass

class DependencyResolver:
    """
    Manages dependencies between indicators and determines computation order.
    """
    def __init__(self, available_columns=None):
        """
        Initialize the resolver with an empty DAG.
        
        Args:
            available_columns: Optional set of column names that are available (in addition to base columns)
        """
        self.dag = defaultdict(list)
        self.all_nodes = set()
        self.available_columns = set(BASE_DATA_COLUMNS)
        if available_columns:
            self.available_columns.update(available_columns)
    
    def add_indicator(self, name: str):
        """
        Recursively resolve dependencies for an indicator.
        
        Args:
            name: Name of the indicator to add
            
        Raises:
            MissingDataError: If a required base column is missing
        """
        # Skip if already processed
        if name in self.all_nodes:
            return
            
        # Add to tracking sets
        self.all_nodes.add(name)
        
        # If it's a registered indicator, process its dependencies
        if name in IndicatorRegistry._indicators:
            indicator = IndicatorRegistry.get(name)
            
            # Add dependencies first
            for dep in indicator['inputs']:
                if dep not in self.all_nodes:
                    if dep in IndicatorRegistry._indicators:
                        self.add_indicator(dep)
                    else:
                        self.add_column(dep)
                # Add edge from dependency to current indicator
                self.dag[dep].append(name)
            
            # Add outputs to available columns
            for output in indicator.get('outputs', [name]):
                self.available_columns.add(output)
        else:
            # If it's not a registered indicator, it should be a base column or already computed
            self.add_column(name)
    
    def add_column(self, column: str):
        """
        Add a base data column to the DAG.
        
        Args:
            column: Name of the column to add
            
        Raises:
            MissingDataError: If column is not a recognized base column
        """
        if column not in self.available_columns and column not in IndicatorRegistry._indicators:
            available_indicators = list(IndicatorRegistry._indicators.keys())
            raise MissingDataError(
                f"Column '{column}' is not in base data columns {list(self.available_columns)} "
                f"and is not a registered indicator! Available indicators: {available_indicators}"
            )
        self.all_nodes.add(column)
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort to determine computation order.
        
        Returns:
            List of indicators in computation order
            
        Raises:
            CircularDependencyError: If circular dependencies are detected
        """
        # Kahn's algorithm for topological sort
        in_degree = {node: 0 for node in self.all_nodes}
        
        # Calculate in-degree for each node (number of dependencies pointing to it)
        for node in self.all_nodes:
            for neighbor in self.dag[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Start with nodes that have no dependencies
        queue = [node for node in self.all_nodes if in_degree[node] == 0]
        sorted_order = []
        
        while queue:
            current = queue.pop(0)
            sorted_order.append(current)
            
            # Decrease in-degree for all neighbors
            for neighbor in self.dag[current]:
                in_degree[neighbor] -= 1
                # If neighbor has no more dependencies, add to queue
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we couldn't sort all nodes, there must be a cycle
        if len(sorted_order) != len(self.all_nodes):
            raise CircularDependencyError("Circular dependency detected in indicators!")
        
        # Filter out base columns
        return [node for node in sorted_order if node in IndicatorRegistry._indicators]
