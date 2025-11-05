# Urban/Rural Obesity Probability - Implementation Guide

## Overview
This guide shows how to make obesity probability conditional on urban/rural status in Synthea. I've created an example module (`urban_rural_obesity_example.json`) that demonstrates this pattern.

## Basic Implementation

### 1. Assign Urban/Rural Status at Birth
The example module assigns urban/rural status early in the patient's life:

```json
"Assign_Urban_Rural_Status": {
  "type": "Simple",
  "distributed_transition": [
    {
      "distribution": 0.70,
      "transition": "Set_Urban_Flag"
    },
    {
      "distribution": 0.30,
      "transition": "Set_Rural_Flag"
    }
  ]
}
```

### 2. Store Status as Attributes
Store both string and numeric versions for flexible condition checking:

```json
"Set_Urban_Flag": {
  "type": "SetAttribute",
  "attribute": "urban_rural_status",
  "value": "urban",
  "direct_transition": "Set_Urban_Numeric"
},
"Set_Urban_Numeric": {
  "type": "SetAttribute",
  "attribute": "is_urban",
  "value": 1,
  "direct_transition": "Wait_Until_Adult"
}
```

### 3. Apply Conditional Probabilities
Use `complex_transition` with attribute conditions to apply different probabilities:

```json
"Check_Urban_Rural_For_Obesity": {
  "type": "Simple",
  "complex_transition": [
    {
      "condition": {
        "condition_type": "Attribute",
        "attribute": "urban_rural_status",
        "operator": "==",
        "value": "urban"
      },
      "distributions": [
        {
          "distribution": 0.35,  // 35% obesity risk for urban
          "transition": "Urban_Obesity_Risk_Factors"
        },
        {
          "distribution": 0.65,
          "transition": "No_Obesity_Yet"
        }
      ]
    },
    {
      "condition": {
        "condition_type": "Attribute",
        "attribute": "urban_rural_status",
        "operator": "==",
        "value": "rural"
      },
      "distributions": [
        {
          "distribution": 0.25,  // 25% obesity risk for rural
          "transition": "Rural_Obesity_Risk_Factors"
        },
        {
          "distribution": 0.75,
          "transition": "No_Obesity_Yet"
        }
      ]
    }
  ]
}
```

## Integration with Existing Wellness Module

The existing `wellness_encounters.json` module checks BMI during wellness visits. To integrate:

### Option 1: Modify wellness_encounters.json directly

Edit `/src/main/resources/modules/wellness_encounters.json` to check urban/rural status:

```json
"Obese_Check": {
  "type": "Simple",
  "complex_transition": [
    {
      "condition": {
        "condition_type": "And",
        "conditions": [
          {
            "condition_type": "Vital Sign",
            "vital_sign": "BMI",
            "operator": ">=",
            "value": 30
          },
          {
            "condition_type": "Attribute",
            "attribute": "urban_rural_status",
            "operator": "==",
            "value": "urban"
          }
        ]
      },
      "distributions": [
        {
          "distribution": 0.9,  // Urban dwellers more likely to be diagnosed
          "transition": "Obesity"
        },
        {
          "distribution": 0.1,
          "transition": "Record_BP"
        }
      ]
    },
    {
      "condition": {
        "condition_type": "And",
        "conditions": [
          {
            "condition_type": "Vital Sign",
            "vital_sign": "BMI",
            "operator": ">=",
            "value": 30
          },
          {
            "condition_type": "Attribute",
            "attribute": "urban_rural_status",
            "operator": "==",
            "value": "rural"
          }
        ]
      },
      "distributions": [
        {
          "distribution": 0.7,  // Rural dwellers less likely to be diagnosed
          "transition": "Obesity"
        },
        {
          "distribution": 0.3,
          "transition": "Record_BP"
        }
      ]
    },
    {
      "transition": "Record_BP"
    }
  ]
}
```

### Option 2: Create a Lifecycle Module

Create a module that runs throughout the patient's life and sets attributes:

```json
{
  "name": "Urban Rural Demographics",
  "remarks": ["Assigns demographic attributes based on location"],
  "states": {
    "Initial": {
      "type": "Initial",
      "direct_transition": "Determine_Location"
    },

    "Determine_Location": {
      "type": "CallSubmodule",
      "submodule": "geographic_demographics",
      "direct_transition": "Set_Urban_Rural"
    },

    "Set_Urban_Rural": {
      "type": "Simple",
      "remarks": ["Could use actual ZIP code data here"],
      "distributed_transition": [
        {
          "distribution": 0.8,
          "transition": "Urban_Setting"
        },
        {
          "distribution": 0.2,
          "transition": "Rural_Setting"
        }
      ]
    },

    "Urban_Setting": {
      "type": "SetAttribute",
      "attribute": "urban_rural_status",
      "value": "urban",
      "direct_transition": "Set_Urban_Multiplier"
    },

    "Set_Urban_Multiplier": {
      "type": "SetAttribute",
      "attribute": "obesity_risk_multiplier",
      "value": 1.4,
      "direct_transition": "Terminal"
    },

    "Rural_Setting": {
      "type": "SetAttribute",
      "attribute": "urban_rural_status",
      "value": "rural",
      "direct_transition": "Set_Rural_Multiplier"
    },

    "Set_Rural_Multiplier": {
      "type": "SetAttribute",
      "attribute": "obesity_risk_multiplier",
      "value": 1.0,
      "direct_transition": "Terminal"
    },

    "Terminal": {
      "type": "Terminal"
    }
  }
}
```

## Advanced: Using Lookup Tables

For more sophisticated modeling, use lookup tables to vary probability by age, gender, and location:

### 1. Create a lookup table file

Create `/src/main/resources/modules/lookup_tables/obesity_risk_by_demographics.csv`:

```csv
age_min,age_max,gender,location,obesity_probability
0,17,M,urban,0.18
0,17,F,urban,0.16
0,17,M,rural,0.15
0,17,F,rural,0.14
18,29,M,urban,0.28
18,29,F,urban,0.26
18,29,M,rural,0.22
18,29,F,rural,0.20
30,44,M,urban,0.38
30,44,F,urban,0.36
30,44,M,rural,0.30
30,44,F,rural,0.28
45,64,M,urban,0.42
45,64,F,urban,0.40
45,64,M,rural,0.35
45,64,F,rural,0.33
65,999,M,urban,0.35
65,999,F,urban,0.33
65,999,M,rural,0.30
65,999,F,rural,0.28
```

### 2. Use TableTransition in the module

```json
"Obesity_Risk_By_Demographics": {
  "type": "Simple",
  "lookup_table_transition": [
    {
      "transition": "Develop_Obesity",
      "default_probability": 0.25,
      "lookup_table_name": "obesity_risk_by_demographics.csv",
      "attributes": [
        "age",
        "age",
        "gender",
        "urban_rural_status"
      ]
    },
    {
      "transition": "No_Obesity"
    }
  ]
}
```

## Testing Your Implementation

### 1. Generate test patients
```bash
./run_synthea -p 100 -s 12345
```

### 2. Check the output
Look for the urban_rural_status attribute in the generated patient records.

### 3. Analyze distributions
Check if obesity rates differ between urban and rural populations as expected.

## Additional Considerations

### Geographic-Based Assignment
Instead of random assignment, use actual geographic data:

```json
"Check_State_For_Urban_Rural": {
  "type": "Simple",
  "conditional_transition": [
    {
      "condition": {
        "condition_type": "Attribute",
        "attribute": "state",
        "operator": "==",
        "value": "Massachusetts"
      },
      "distributions": [
        {
          "distribution": 0.92,  // MA is highly urban
          "transition": "Set_Urban"
        },
        {
          "distribution": 0.08,
          "transition": "Set_Rural"
        }
      ]
    },
    {
      "condition": {
        "condition_type": "Attribute",
        "attribute": "state",
        "operator": "==",
        "value": "Montana"
      },
      "distributions": [
        {
          "distribution": 0.44,  // MT is more rural
          "transition": "Set_Urban"
        },
        {
          "distribution": 0.56,
          "transition": "Set_Rural"
        }
      ]
    }
  ]
}
```

### Multiple Risk Factors
Combine urban/rural with other factors:

```json
"Complex_Obesity_Risk": {
  "type": "Simple",
  "complex_transition": [
    {
      "condition": {
        "condition_type": "And",
        "conditions": [
          {
            "condition_type": "Attribute",
            "attribute": "urban_rural_status",
            "operator": "==",
            "value": "urban"
          },
          {
            "condition_type": "Socioeconomic Status",
            "category": "Low"
          },
          {
            "condition_type": "Race",
            "race": "black"
          }
        ]
      },
      "distributions": [
        {
          "distribution": 0.45,  // Highest risk group
          "transition": "High_Risk_Obesity_Path"
        },
        {
          "distribution": 0.55,
          "transition": "Continue"
        }
      ]
    }
  ]
}
```

## Running the Module

1. Place the module in `/src/main/resources/modules/`
2. Ensure it's referenced in `synthea.properties` or loaded automatically
3. Generate patients:
```bash
./run_synthea -p 1000 -m urban_rural_obesity_example
```

## Validation

To validate your implementation works:

1. Generate a test population
2. Export to CSV format
3. Analyze obesity rates by urban/rural status
4. Compare to expected probabilities

This approach gives you fine-grained control over obesity probability based on demographic factors while integrating seamlessly with Synthea's existing modules.