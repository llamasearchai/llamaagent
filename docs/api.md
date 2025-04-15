# API Reference

## llamaagent API

### Class: Client

The main client interface for interacting with the llamaagent library.

#### Methods

```python
def __init__(self, api_key=None, config=None):
    """
    Initialize the client.
    
    Args:
        api_key (str, optional): API key for authentication
        config (dict, optional): Configuration options
    """
```

```python
def process(self, query, options=None):
    """
    Process a query using the llamaagent engine.
    
    Args:
        query (str): The query to process
        options (dict, optional): Processing options
        
    Returns:
        dict: The processing results
    """
```

```python
def batch_process(self, queries, options=None):
    """
    Process multiple queries in batch.
    
    Args:
        queries (List[str]): List of queries to process
        options (dict, optional): Processing options
        
    Returns:
        List[dict]: List of processing results
    """
```

### Class: DataProcessor

Handles data processing operations.

```python
def transform(self, data, transformer_type="standard"):
    """
    Transform data using the specified transformer.
    
    Args:
        data (Any): The data to transform
        transformer_type (str): The type of transformer to use
        
    Returns:
        Any: The transformed data
    """
```
