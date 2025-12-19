# MCP Server Information Summary

## Summary Information

- **Collection Time**: 2025-10-14T12:20:04.142700
- **Connection Mode**: INDIVIDUAL
- **Total Servers**: 28
- **Successful Connections**: 25
- **Failed Connections**: 3
- **Total Tools Discovered**: 15
- **Servers That Needed Retry**: 3
- **Total Retry Attempts**: 12

## Servers That Needed Retry

- **Wikipedia**: 5 attempts, 0 tools, final status: failed
- **BioMCP**: 5 attempts, 0 tools, final status: failed
- **Reddit**: 5 attempts, 0 tools, final status: failed

## Failed Servers

- **Wikipedia**: Connection timeout after 30 seconds (attempted 5 times)
- **BioMCP**: Connection timeout after 30 seconds (attempted 5 times)
- **Reddit**: Connection timeout after 30 seconds (attempted 5 times)

## Server Details

###  OpenAPI Explorer

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Unit Converter

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Wikipedia

**Description**: 

**Connection Status**: failed

**Error Message**: Connection timeout after 30 seconds

**Available Tools**: None

---

###  Google Maps

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Bibliomantic

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  BioMCP

**Description**: 

**Connection Status**: failed

**Error Message**: Connection timeout after 30 seconds

**Available Tools**: None

---

###  Call for Papers

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Car Price Evaluator

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Context7

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  DEX Paprika

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  FruityVice

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Game Trends

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Huge Icons

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Hugging Face

**Description**: 

**Connection Status**: success

**Available Tools** (10 ):

#### search-models

**Description**: Search for models on Hugging Face Hub

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search term (e.g., 'bert', 'gpt')"
    },
    "author": {
      "type": "string",
      "description": "Filter by author/organization (e.g., 'huggingface', 'google')"
    },
    "tags": {
      "type": "string",
      "description": "Filter by tags (e.g., 'text-classification', 'translation')"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return"
    }
  }
}
```

#### get-model-info

**Description**: Get detailed information about a specific model

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "model_id": {
      "type": "string",
      "description": "The ID of the model (e.g., 'google/bert-base-uncased')"
    }
  },
  "required": [
    "model_id"
  ]
}
```

#### search-datasets

**Description**: Search for datasets on Hugging Face Hub

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search term"
    },
    "author": {
      "type": "string",
      "description": "Filter by author/organization"
    },
    "tags": {
      "type": "string",
      "description": "Filter by tags"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return"
    }
  }
}
```

#### get-dataset-info

**Description**: Get detailed information about a specific dataset

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "dataset_id": {
      "type": "string",
      "description": "The ID of the dataset (e.g., 'squad')"
    }
  },
  "required": [
    "dataset_id"
  ]
}
```

#### search-spaces

**Description**: Search for Spaces on Hugging Face Hub

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search term"
    },
    "author": {
      "type": "string",
      "description": "Filter by author/organization"
    },
    "tags": {
      "type": "string",
      "description": "Filter by tags"
    },
    "sdk": {
      "type": "string",
      "description": "Filter by SDK (e.g., 'streamlit', 'gradio', 'docker')"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return"
    }
  }
}
```

#### get-space-info

**Description**: Get detailed information about a specific Space

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "space_id": {
      "type": "string",
      "description": "The ID of the Space (e.g., 'huggingface/diffusers-demo')"
    }
  },
  "required": [
    "space_id"
  ]
}
```

#### get-paper-info

**Description**: Get information about a specific paper on Hugging Face

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "arxiv_id": {
      "type": "string",
      "description": "The arXiv ID of the paper (e.g., '1810.04805')"
    }
  },
  "required": [
    "arxiv_id"
  ]
}
```

#### get-daily-papers

**Description**: Get the list of daily papers curated by Hugging Face

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {}
}
```

#### search-collections

**Description**: Search for collections on Hugging Face Hub

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "owner": {
      "type": "string",
      "description": "Filter by owner"
    },
    "item": {
      "type": "string",
      "description": "Filter by item (e.g., 'models/teknium/OpenHermes-2.5-Mistral-7B')"
    },
    "query": {
      "type": "string",
      "description": "Search term for titles and descriptions"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return"
    }
  }
}
```

#### get-collection-info

**Description**: Get detailed information about a specific collection

**Input Parameters**:
```json
{
  "type": "object",
  "properties": {
    "namespace": {
      "type": "string",
      "description": "The namespace of the collection (user or organization)"
    },
    "collection_id": {
      "type": "string",
      "description": "The ID part of the collection"
    }
  },
  "required": [
    "namespace",
    "collection_id"
  ]
}
```

---

###  Math MCP

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  NixOS

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  OSINT Intelligence

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Reddit

**Description**: 

**Connection Status**: failed

**Error Message**: Connection timeout after 30 seconds

**Available Tools**: None

---

###  National Parks

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Medical Calculator

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Metropolitan Museum

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Movie Recommender

**Description**: 

**Connection Status**: success

**Available Tools** (1 ):

#### get_movies

**Description**: 
    Get movie suggestions based on keyword.
    

**Input Parameters**:
```json
{
  "properties": {
    "keyword": {
      "title": "Keyword",
      "type": "string"
    }
  },
  "required": [
    "keyword"
  ],
  "title": "get_moviesArguments",
  "type": "object"
}
```

---

###  NASA Data

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  OKX Exchange

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Paper Search

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Scientific Computing

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

###  Weather Data

**Description**: 

**Connection Status**: success

**Available Tools** (4 ):

#### get_current_weather_tool

**Description**: 
    Get current weather information for a specific city.

    Args:
        city: Name of the city to get weather for

    Returns:
        Current weather data including temperature, conditions, humidity, wind, etc.
    

**Input Parameters**:
```json
{
  "properties": {
    "city": {
      "title": "City",
      "type": "string"
    }
  },
  "required": [
    "city"
  ],
  "title": "get_current_weather_toolArguments",
  "type": "object"
}
```

#### get_weather_forecast_tool

**Description**: 
    Get weather forecast for a specific city.

    Args:
        city: Name of the city to get forecast for
        days: Number of days to forecast (1-10, default: 3)

    Returns:
        Weather forecast data for the specified number of days
    

**Input Parameters**:
```json
{
  "properties": {
    "city": {
      "title": "City",
      "type": "string"
    },
    "days": {
      "default": 3,
      "title": "Days",
      "type": "integer"
    }
  },
  "required": [
    "city"
  ],
  "title": "get_weather_forecast_toolArguments",
  "type": "object"
}
```

#### search_locations_tool

**Description**: 
    Search for locations by name.

    Args:
        query: Location name or partial name to search for

    Returns:
        List of matching locations with their details
    

**Input Parameters**:
```json
{
  "properties": {
    "query": {
      "title": "Query",
      "type": "string"
    }
  },
  "required": [
    "query"
  ],
  "title": "search_locations_toolArguments",
  "type": "object"
}
```

#### get_live_temp

**Description**: 
    Legacy tool: Get current temperature for a city (for backward compatibility).
    Use get_current_weather_tool for more detailed information.
    

**Input Parameters**:
```json
{
  "properties": {
    "city": {
      "title": "City",
      "type": "string"
    }
  },
  "required": [
    "city"
  ],
  "title": "get_live_tempArguments",
  "type": "object"
}
```

---

###  Time MCP

**Description**: 

**Connection Status**: success_no_tools

**Available Tools**: None

---

