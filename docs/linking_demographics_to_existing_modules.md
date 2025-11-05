# Linking Demographics to Existing Synthea Modules: Technical Implementation Guide

## Overview

This guide provides specific, actionable methods for modifying existing Synthea modules to incorporate demographic-based health outcome probabilities. We'll show exactly how to edit current modules to make disease probabilities conditional on factors like urban/rural status, socioeconomic status, race, and other demographics.

## 1. Core Implementation Pattern

The key pattern for linking demographics to existing modules involves three steps:

1. **Set demographic attributes early** (in lifecycle or demographics module)
2. **Check attributes in existing modules** (using conditional transitions)
3. **Apply different probabilities** based on demographic factors

## 2. Modifying Existing Allergy Module

Let's modify the existing `allergies.json` module to make allergies more common in urban areas (due to hygiene hypothesis and pollution).

### Current Module Structure (simplified)
```json
{
  "name": "Allergies and Intolerances",
  "states": {
    "Allergy_Incidence": {
      "type": "Simple",
      "distributed_transition": [
        {
          "distribution": 0.20,
          "transition": "Allergy_Onset"
        },
        {
          "distribution": 0.80,
          "transition": "No_Allergy"
        }
      ]
    }
  }
}
```

### Modified Module with Urban/Rural Differentiation
```json
{
  "name": "Allergies and Intolerances",
  "states": {
    "Allergy_Incidence": {
      "type": "Simple",
      "remarks": [
        "Urban areas: 25% allergy rate (pollution, less early exposure to allergens)",
        "Suburban areas: 20% allergy rate",
        "Rural areas: 15% allergy rate (hygiene hypothesis, early farm exposure)"
      ],
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
              "distribution": 0.25,
              "transition": "Allergy_Onset_Urban"
            },
            {
              "distribution": 0.75,
              "transition": "No_Allergy"
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
              "distribution": 0.15,
              "transition": "Allergy_Onset_Rural"
            },
            {
              "distribution": 0.85,
              "transition": "No_Allergy"
            }
          ]
        },
        {
          "remarks": "Default for suburban or unspecified",
          "distributions": [
            {
              "distribution": 0.20,
              "transition": "Allergy_Onset"
            },
            {
              "distribution": 0.80,
              "transition": "No_Allergy"
            }
          ]
        }
      ]
    },

    "Allergy_Onset_Urban": {
      "type": "Simple",
      "remarks": "Urban allergies more likely to be environmental",
      "distributed_transition": [
        {
          "distribution": 0.40,
          "transition": "Environmental_Allergy"
        },
        {
          "distribution": 0.35,
          "transition": "Food_Allergy"
        },
        {
          "distribution": 0.25,
          "transition": "Drug_Allergy"
        }
      ]
    },

    "Allergy_Onset_Rural": {
      "type": "Simple",
      "remarks": "Rural allergies more likely to be seasonal",
      "distributed_transition": [
        {
          "distribution": 0.50,
          "transition": "Seasonal_Allergy"
        },
        {
          "distribution": 0.30,
          "transition": "Food_Allergy"
        },
        {
          "distribution": 0.20,
          "transition": "Drug_Allergy"
        }
      ]
    }
  }
}
```

## 3. Modifying Asthma Module

### Add demographic factors to `asthma.json`:

```json
{
  "Asthma_Prevalence": {
    "type": "Simple",
    "remarks": [
      "Asthma affected by multiple demographic factors:",
      "- Urban areas: higher due to air pollution",
      "- Low SES: higher due to housing conditions",
      "- Race: higher in Black and Puerto Rican populations"
    ],
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
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": "<=",
              "value": 2
            },
            {
              "condition_type": "Race",
              "race": "black"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.20,
            "remarks": "Highest risk group: 20% prevalence",
            "transition": "Childhood_Asthma"
          },
          {
            "distribution": 0.80,
            "transition": "No_Asthma"
          }
        ]
      },
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Attribute",
              "attribute": "urban_rural_status",
              "operator": "==",
              "value": "rural"
            },
            {
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": ">=",
              "value": 4
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.06,
            "remarks": "Lowest risk group: 6% prevalence",
            "transition": "Childhood_Asthma"
          },
          {
            "distribution": 0.94,
            "transition": "No_Asthma"
          }
        ]
      },
      {
        "remarks": "Default prevalence for unmatched demographics",
        "distributions": [
          {
            "distribution": 0.089,
            "transition": "Childhood_Asthma"
          },
          {
            "distribution": 0.911,
            "transition": "No_Asthma"
          }
        ]
      }
    ]
  }
}
```

## 4. Modifying Metabolic Syndrome Module

### Edit `metabolic_syndrome_disease.json` to include SES factors:

```json
{
  "Veteran_Diabetes_Prevalence": {
    "type": "Simple",
    "remarks": "Original veteran check - now enhanced with SES",
    "complex_transition": [
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Attribute",
              "attribute": "veteran",
              "operator": "is not nil"
            },
            {
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": "<=",
              "value": 2
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.35,
            "remarks": "Low SES veterans: 35% diabetes risk",
            "transition": "Onset_Prediabetes"
          },
          {
            "distribution": 0.10,
            "transition": "Onset_Diabetes"
          },
          {
            "distribution": 0.55,
            "transition": "No_Diabetes"
          }
        ]
      },
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Attribute",
              "attribute": "veteran",
              "operator": "is not nil"
            },
            {
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": ">=",
              "value": 4
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.20,
            "remarks": "High SES veterans: 20% diabetes risk",
            "transition": "Onset_Prediabetes"
          },
          {
            "distribution": 0.05,
            "transition": "Onset_Diabetes"
          },
          {
            "distribution": 0.75,
            "transition": "No_Diabetes"
          }
        ]
      }
    ]
  }
}
```

## 5. Creating a Demographics Assignment Module

Create a new module that runs at birth to assign all demographic attributes:

### `demographics_assignment.json`
```json
{
  "name": "Demographics Assignment",
  "remarks": [
    "This module assigns demographic attributes at birth that will be used",
    "by other modules to determine health outcome probabilities"
  ],
  "states": {
    "Initial": {
      "type": "Initial",
      "direct_transition": "Assign_Birth_Location"
    },

    "Assign_Birth_Location": {
      "type": "Simple",
      "remarks": "Assign based on state population distributions",
      "conditional_transition": [
        {
          "condition": {
            "condition_type": "Attribute",
            "attribute": "state",
            "operator": "==",
            "value": "New York"
          },
          "distributions": [
            {
              "distribution": 0.88,
              "transition": "Set_Urban"
            },
            {
              "distribution": 0.12,
              "transition": "Set_Rural"
            }
          ]
        },
        {
          "condition": {
            "condition_type": "Attribute",
            "attribute": "state",
            "operator": "==",
            "value": "Vermont"
          },
          "distributions": [
            {
              "distribution": 0.35,
              "transition": "Set_Urban"
            },
            {
              "distribution": 0.65,
              "transition": "Set_Rural"
            }
          ]
        },
        {
          "remarks": "National average as default",
          "distributions": [
            {
              "distribution": 0.80,
              "transition": "Set_Urban"
            },
            {
              "distribution": 0.20,
              "transition": "Set_Rural"
            }
          ]
        }
      ]
    },

    "Set_Urban": {
      "type": "SetAttribute",
      "attribute": "urban_rural_status",
      "value": "urban",
      "direct_transition": "Calculate_SES"
    },

    "Set_Rural": {
      "type": "SetAttribute",
      "attribute": "urban_rural_status",
      "value": "rural",
      "direct_transition": "Calculate_SES"
    },

    "Calculate_SES": {
      "type": "Simple",
      "remarks": "Assign SES quintile (1=lowest, 5=highest)",
      "distributed_transition": [
        {
          "distribution": 0.20,
          "transition": "SES_Quintile_1"
        },
        {
          "distribution": 0.20,
          "transition": "SES_Quintile_2"
        },
        {
          "distribution": 0.20,
          "transition": "SES_Quintile_3"
        },
        {
          "distribution": 0.20,
          "transition": "SES_Quintile_4"
        },
        {
          "distribution": 0.20,
          "transition": "SES_Quintile_5"
        }
      ]
    },

    "SES_Quintile_1": {
      "type": "SetAttribute",
      "attribute": "ses_quintile",
      "value": 1,
      "direct_transition": "Set_Health_Multipliers"
    },

    "SES_Quintile_2": {
      "type": "SetAttribute",
      "attribute": "ses_quintile",
      "value": 2,
      "direct_transition": "Set_Health_Multipliers"
    },

    "SES_Quintile_3": {
      "type": "SetAttribute",
      "attribute": "ses_quintile",
      "value": 3,
      "direct_transition": "Set_Health_Multipliers"
    },

    "SES_Quintile_4": {
      "type": "SetAttribute",
      "attribute": "ses_quintile",
      "value": 4,
      "direct_transition": "Set_Health_Multipliers"
    },

    "SES_Quintile_5": {
      "type": "SetAttribute",
      "attribute": "ses_quintile",
      "value": 5,
      "direct_transition": "Set_Health_Multipliers"
    },

    "Set_Health_Multipliers": {
      "type": "Simple",
      "remarks": "Set risk multipliers based on combined demographics",
      "direct_transition": "Terminal"
    },

    "Terminal": {
      "type": "Terminal"
    }
  }
}
```

## 6. Modifying Hypertension Module

### Edit `hypertension.json` for demographic factors:

```json
{
  "Chance_of_Hypertension": {
    "type": "Simple",
    "complex_transition": [
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Race",
              "race": "black"
            },
            {
              "condition_type": "Age",
              "operator": ">",
              "quantity": 40,
              "unit": "years"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.58,
            "remarks": "58% of Black adults >40 have hypertension",
            "transition": "Onset_Hypertension"
          },
          {
            "distribution": 0.42,
            "transition": "No_Hypertension"
          }
        ]
      },
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Race",
              "race": "white"
            },
            {
              "condition_type": "Age",
              "operator": ">",
              "quantity": 40,
              "unit": "years"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.33,
            "remarks": "33% of White adults >40 have hypertension",
            "transition": "Onset_Hypertension"
          },
          {
            "distribution": 0.67,
            "transition": "No_Hypertension"
          }
        ]
      }
    ]
  },

  "Hypertension_Severity": {
    "type": "Simple",
    "complex_transition": [
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": "<=",
              "value": 2
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
            "distribution": 0.40,
            "remarks": "Low SES + rural = poor BP control",
            "transition": "Severe_Hypertension"
          },
          {
            "distribution": 0.60,
            "transition": "Moderate_Hypertension"
          }
        ]
      }
    ]
  }
}
```

## 7. Modifying Opioid Addiction Module

### Edit `opioid_addiction.json` for geographic and demographic factors:

```json
{
  "Prescription_Opioid_Use": {
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
              "value": "rural"
            },
            {
              "condition_type": "Race",
              "race": "white"
            },
            {
              "condition_type": "Age",
              "operator": ">=",
              "quantity": 25,
              "unit": "years"
            },
            {
              "condition_type": "Age",
              "operator": "<=",
              "quantity": 54,
              "unit": "years"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.08,
            "remarks": "Higher risk in rural white populations",
            "transition": "Opioid_Dependence_Pathway"
          },
          {
            "distribution": 0.92,
            "transition": "No_Addiction"
          }
        ]
      },
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
              "condition_type": "Race",
              "race": "black"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.02,
            "remarks": "Lower prescription opioid risk, but may transition to heroin",
            "transition": "Alternative_Substance_Risk"
          },
          {
            "distribution": 0.98,
            "transition": "No_Addiction"
          }
        ]
      }
    ]
  }
}
```

## 8. Modifying Mental Health Modules

### Edit `depression.json` for SES and urban/rural factors:

```json
{
  "Depression_Screening": {
    "type": "Simple",
    "complex_transition": [
      {
        "condition": {
          "condition_type": "And",
          "conditions": [
            {
              "condition_type": "Attribute",
              "attribute": "ses_quintile",
              "operator": "<=",
              "value": 2
            },
            {
              "condition_type": "Gender",
              "gender": "F"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.25,
            "remarks": "Low SES women: 25% depression risk",
            "transition": "Major_Depression"
          },
          {
            "distribution": 0.20,
            "transition": "Minor_Depression"
          },
          {
            "distribution": 0.55,
            "transition": "No_Depression"
          }
        ]
      }
    ]
  },

  "Treatment_Access": {
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
              "value": "rural"
            },
            {
              "condition_type": "Attribute",
              "attribute": "depression",
              "operator": "is not nil"
            }
          ]
        },
        "distributions": [
          {
            "distribution": 0.30,
            "remarks": "Only 30% of rural depression patients get treatment",
            "transition": "Receive_Treatment"
          },
          {
            "distribution": 0.70,
            "transition": "No_Treatment"
          }
        ]
      }
    ]
  }
}
```

## 9. Using Lookup Tables for Complex Demographics

### Create lookup table: `chronic_disease_risk.csv`
```csv
condition,age_min,age_max,gender,race,urban_rural,ses_quintile,annual_risk
diabetes,18,29,M,white,urban,5,0.003
diabetes,18,29,M,white,urban,1,0.008
diabetes,18,29,M,black,urban,5,0.005
diabetes,18,29,M,black,urban,1,0.015
diabetes,18,29,M,hispanic,rural,1,0.018
diabetes,30,44,M,white,urban,5,0.008
diabetes,30,44,M,white,urban,1,0.022
diabetes,30,44,M,native_american,rural,1,0.045
hypertension,18,29,M,black,urban,1,0.08
hypertension,18,29,M,white,urban,5,0.02
hypertension,45,64,F,black,rural,1,0.15
hypertension,45,64,F,white,urban,5,0.06
```

### Use in module:
```json
{
  "Annual_Chronic_Disease_Risk": {
    "type": "Simple",
    "lookup_table_transition": [
      {
        "transition": "Develop_Condition",
        "default_probability": 0.01,
        "lookup_table_name": "chronic_disease_risk.csv",
        "attributes": [
          "condition_to_check",
          "age",
          "age",
          "gender",
          "race",
          "urban_rural_status",
          "ses_quintile"
        ]
      },
      {
        "transition": "Remain_Healthy"
      }
    ]
  }
}
```

## 10. Implementation Checklist

### Step 1: Create Demographics Module
- [ ] Create `demographics_assignment.json`
- [ ] Add urban/rural assignment logic
- [ ] Add SES calculation
- [ ] Add environmental factors

### Step 2: Modify Target Modules
For each module you want to modify:
- [ ] Identify key decision points (where probabilities are set)
- [ ] Replace `distributed_transition` with `complex_transition`
- [ ] Add demographic condition checks
- [ ] Set different probability distributions

### Step 3: Test and Validate
- [ ] Generate test population (n=1000)
- [ ] Verify demographic attributes are set
- [ ] Check disease distributions match expectations
- [ ] Validate against real-world data

### Step 4: Configure
Add to `synthea.properties`:
```properties
# Enable demographic modules
generate.demographics.enabled = true
generate.demographics.socioeconomic = true
generate.demographics.urban_rural = true

# Set demographic distributions
generate.demographics.urban_percentage = 0.80
generate.demographics.ses_distribution = equal  # or "realistic"
```

## 11. Common Patterns

### Pattern 1: Simple Binary Demographics
```json
{
  "condition": {
    "condition_type": "Attribute",
    "attribute": "is_urban",
    "operator": "==",
    "value": 1
  },
  "distributions": [
    {"distribution": 0.30, "transition": "Urban_Outcome"},
    {"distribution": 0.70, "transition": "No_Outcome"}
  ]
}
```

### Pattern 2: Multiple Demographic Factors
```json
{
  "condition": {
    "condition_type": "And",
    "conditions": [
      {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "<=", "value": 2},
      {"condition_type": "Race", "race": "black"},
      {"condition_type": "Gender", "gender": "F"}
    ]
  },
  "distributions": [...]
}
```

### Pattern 3: Graduated Risk by SES
```json
{
  "complex_transition": [
    {
      "condition": {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "==", "value": 1},
      "distributions": [{"distribution": 0.40, "transition": "Disease"}]
    },
    {
      "condition": {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "==", "value": 2},
      "distributions": [{"distribution": 0.35, "transition": "Disease"}]
    },
    {
      "condition": {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "==", "value": 3},
      "distributions": [{"distribution": 0.30, "transition": "Disease"}]
    },
    {
      "condition": {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "==", "value": 4},
      "distributions": [{"distribution": 0.25, "transition": "Disease"}]
    },
    {
      "condition": {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "==", "value": 5},
      "distributions": [{"distribution": 0.20, "transition": "Disease"}]
    }
  ]
}
```

## 12. Best Practices

### 1. Always Provide Defaults
```json
{
  "complex_transition": [
    {"condition": {...}, "distributions": [...]},
    {"condition": {...}, "distributions": [...]},
    {
      "remarks": "Default for unmatched demographics",
      "distributions": [{"distribution": 0.10, "transition": "Default_Outcome"}]
    }
  ]
}
```

### 2. Document Your Sources
```json
{
  "remarks": [
    "Probabilities based on CDC BRFSS 2021 data",
    "Urban: 25% prevalence (95% CI: 23-27%)",
    "Rural: 18% prevalence (95% CI: 16-20%)",
    "Source: https://www.cdc.gov/..."
  ]
}
```

### 3. Use Meaningful Attribute Names
```json
{
  "attribute": "urban_rural_status",  // Clear naming
  "value": "urban"                    // String for readability
}
```

### 4. Consider Intersectionality
Don't just add factors - consider how they interact:
```json
{
  "remarks": "Low SES amplifies racial disparities",
  "condition": {
    "condition_type": "And",
    "conditions": [
      {"condition_type": "Race", "race": "black"},
      {"condition_type": "Attribute", "attribute": "ses_quintile", "operator": "<=", "value": 2}
    ]
  }
}
```

## Conclusion

This focused guide shows how to modify existing Synthea modules to incorporate demographic-based health disparities. The key is to:

1. Set demographic attributes early in the patient's life
2. Check these attributes at disease onset decision points
3. Apply different probabilities based on demographic combinations
4. Use lookup tables for complex multi-factor relationships
5. Always provide documentation and defaults

By following these patterns, you can create synthetic populations that accurately reflect real-world health disparities while maintaining the modular structure of Synthea.