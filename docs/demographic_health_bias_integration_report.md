# Integrating Demographic Data to Create Realistic Health Outcome Bias in Synthea

## Executive Summary

This report outlines strategies for incorporating demographic-based health disparities into Synthea's synthetic patient generation to create more realistic population health data. By implementing conditional probabilities based on social determinants of health (SDOH), we can generate synthetic populations that accurately reflect real-world health inequities while maintaining privacy and ethical standards.

## 1. Introduction

### 1.1 Purpose
Synthea currently generates synthetic patients with some demographic variation, but health outcomes could be more closely tied to demographic factors to reflect documented health disparities. This integration is crucial for:
- Training healthcare AI systems that won't perpetuate biases
- Testing health equity interventions
- Modeling population health accurately
- Educational purposes in health disparities research

### 1.2 Ethical Considerations
While implementing realistic health disparities, we must:
- Document that these are statistical patterns, not deterministic rules
- Provide clear documentation about implemented biases
- Allow configurability to enable/disable bias modeling
- Ensure the data is used for improving health equity, not perpetuating discrimination

## 2. Key Demographic Factors and Their Health Correlations

### 2.1 Geographic Location (Urban/Rural/Suburban)

**Implementation Strategy:**
```json
{
  "geographic_health_factors": {
    "urban": {
      "obesity_risk_multiplier": 1.3,
      "diabetes_risk_multiplier": 1.2,
      "mental_health_access": 0.8,
      "specialist_access": 0.9,
      "air_quality_impact": 1.4,
      "violent_injury_risk": 1.5
    },
    "suburban": {
      "obesity_risk_multiplier": 1.1,
      "diabetes_risk_multiplier": 1.1,
      "mental_health_access": 0.7,
      "specialist_access": 0.8,
      "air_quality_impact": 1.0,
      "violent_injury_risk": 0.7
    },
    "rural": {
      "obesity_risk_multiplier": 1.2,
      "diabetes_risk_multiplier": 1.15,
      "mental_health_access": 0.4,
      "specialist_access": 0.3,
      "air_quality_impact": 0.8,
      "violent_injury_risk": 0.6,
      "agricultural_injury_risk": 2.5,
      "opioid_risk_multiplier": 1.4
    }
  }
}
```

**Key Health Disparities:**
- Rural areas: Higher rates of chronic disease, limited healthcare access, higher mortality rates
- Urban areas: Higher environmental health risks, stress-related conditions, infectious disease spread
- Suburban areas: Generally better health outcomes but rising obesity rates

### 2.2 Socioeconomic Status (SES)

**Implementation Approach:**
Create an SES score based on:
- Education level
- Income quintile
- Employment status
- Insurance type

```java
public class SocioeconomicStatus {
  public static double calculateSESScore(Person person) {
    double educationScore = getEducationScore(person);
    double incomeScore = getIncomeScore(person);
    double employmentScore = getEmploymentScore(person);
    double insuranceScore = getInsuranceScore(person);

    return (educationScore * 0.3 + incomeScore * 0.35 +
            employmentScore * 0.2 + insuranceScore * 0.15);
  }

  public static double getHealthRiskMultiplier(double sesScore) {
    // Lower SES = higher health risks
    return 2.0 - sesScore; // Inverts the 0-1 score
  }
}
```

**SES-Correlated Health Outcomes:**
- Low SES: 2-3x higher rates of diabetes, heart disease, stroke
- Healthcare utilization: Emergency department vs. preventive care
- Medication adherence: Lower SES = 40-60% adherence vs. 70-80% for high SES
- Mental health: Depression and anxiety rates inversely correlated with SES

### 2.3 Race and Ethnicity

**Documented Health Disparities to Model:**
```json
{
  "health_disparities_by_race": {
    "black": {
      "hypertension_risk": 1.4,
      "diabetes_risk": 1.3,
      "maternal_mortality_risk": 3.0,
      "stroke_risk": 1.5,
      "asthma_risk": 1.4,
      "prostate_cancer_risk": 1.7,
      "sickle_cell_carrier": 0.08
    },
    "hispanic": {
      "diabetes_risk": 1.5,
      "obesity_risk": 1.2,
      "liver_disease_risk": 1.5,
      "teen_pregnancy_risk": 1.5,
      "uninsured_probability": 1.8
    },
    "asian": {
      "diabetes_risk_at_lower_bmi": true,
      "bmi_threshold_adjustment": -2,
      "hepatitis_b_risk": 2.0,
      "stomach_cancer_risk": 2.0,
      "osteoporosis_risk": 1.2
    },
    "native_american": {
      "diabetes_risk": 2.0,
      "alcohol_use_disorder_risk": 1.7,
      "suicide_risk": 1.5,
      "unintentional_injury_risk": 1.4,
      "infant_mortality_risk": 1.6
    },
    "white": {
      "skin_cancer_risk": 20.0,
      "osteoporosis_risk": 1.3,
      "opioid_addiction_risk": 1.2
    }
  }
}
```

### 2.4 Age and Gender Intersections

**Complex Intersection Modeling:**
```python
def calculate_condition_probability(person):
    base_prob = get_base_probability(condition)

    # Age adjustment
    age_factor = get_age_risk_curve(person.age, condition)

    # Gender adjustment
    gender_factor = get_gender_risk_factor(person.gender, condition)

    # Intersection effects
    if person.gender == 'F' and person.age > 45:
        # Post-menopausal considerations
        if condition in ['osteoporosis', 'heart_disease']:
            gender_factor *= 1.5

    if person.gender == 'M' and person.age < 25:
        # Young male risk behaviors
        if condition in ['injury', 'substance_use']:
            gender_factor *= 1.8

    return base_prob * age_factor * gender_factor
```

### 2.5 Environmental Factors

**ZIP Code Level Data Integration:**
```json
{
  "environmental_health_factors": {
    "air_quality_index": {
      "good": {"asthma_risk": 1.0, "copd_risk": 1.0},
      "moderate": {"asthma_risk": 1.2, "copd_risk": 1.1},
      "unhealthy": {"asthma_risk": 1.5, "copd_risk": 1.3},
      "hazardous": {"asthma_risk": 2.0, "copd_risk": 1.6}
    },
    "food_desert": {
      "obesity_risk_multiplier": 1.3,
      "diabetes_risk_multiplier": 1.4,
      "hypertension_risk_multiplier": 1.2
    },
    "walkability_score": {
      "high": {"physical_activity_level": 1.3, "obesity_risk": 0.8},
      "medium": {"physical_activity_level": 1.0, "obesity_risk": 1.0},
      "low": {"physical_activity_level": 0.7, "obesity_risk": 1.2}
    }
  }
}
```

## 3. Implementation Architecture

### 3.1 Module Structure

Create a hierarchical module system:

```
modules/
├── demographics/
│   ├── ses_assignment.json
│   ├── urban_rural_assignment.json
│   └── environmental_factors.json
├── health_disparities/
│   ├── chronic_disease_disparities.json
│   ├── maternal_health_disparities.json
│   ├── mental_health_disparities.json
│   └── substance_use_disparities.json
└── lookup_tables/
    ├── disease_risk_by_demographics.csv
    ├── healthcare_utilization_by_ses.csv
    └── medication_adherence_by_factors.csv
```

### 3.2 Attribute Management System

```java
public class DemographicAttributeManager {

  public void initializePersonDemographics(Person person) {
    // Core demographics
    assignUrbanRuralStatus(person);
    calculateSESScore(person);
    assignEnvironmentalFactors(person);

    // Derived risk factors
    calculateCompositeRiskScores(person);
    setHealthcareBehaviorPatterns(person);
  }

  private void calculateCompositeRiskScores(Person person) {
    // Combine multiple factors
    double diabetesRisk = baseDiabetesRisk;
    diabetesRisk *= getRaceMultiplier(person, "diabetes");
    diabetesRisk *= getSESMultiplier(person);
    diabetesRisk *= getGeographicMultiplier(person, "diabetes");
    diabetesRisk *= getEnvironmentalMultiplier(person, "diabetes");

    person.attributes.put("diabetes_risk_score", diabetesRisk);
  }
}
```

### 3.3 Lookup Table System

Create comprehensive lookup tables for nuanced modeling:

**obesity_risk_factors.csv:**
```csv
age_min,age_max,gender,race,ses_quintile,urban_rural,obesity_probability,severe_obesity_probability
18,29,M,white,1,urban,0.22,0.05
18,29,M,white,1,rural,0.25,0.06
18,29,M,white,5,urban,0.15,0.02
18,29,M,black,1,urban,0.28,0.08
18,29,M,black,5,urban,0.18,0.03
18,29,F,white,1,urban,0.24,0.06
18,29,F,black,1,urban,0.35,0.12
18,29,F,hispanic,1,urban,0.30,0.09
30,44,M,white,1,urban,0.35,0.10
30,44,M,black,1,urban,0.38,0.13
# ... continues with all combinations
```

### 3.4 Healthcare Access Patterns

Model realistic healthcare utilization based on demographics:

```json
{
  "healthcare_utilization_patterns": {
    "high_ses_urban": {
      "preventive_care_probability": 0.85,
      "emergency_dept_as_primary": 0.05,
      "specialist_referral_follow_through": 0.90,
      "telehealth_adoption": 0.70
    },
    "low_ses_urban": {
      "preventive_care_probability": 0.35,
      "emergency_dept_as_primary": 0.45,
      "specialist_referral_follow_through": 0.40,
      "telehealth_adoption": 0.25
    },
    "low_ses_rural": {
      "preventive_care_probability": 0.25,
      "emergency_dept_as_primary": 0.30,
      "specialist_referral_follow_through": 0.20,
      "telehealth_adoption": 0.35
    }
  }
}
```

## 4. Advanced Integration Strategies

### 4.1 Life Course Modeling

Implement cumulative disadvantage over the lifespan:

```java
public class LifeCourseHealthModel {

  public void applyLifeCourseEffects(Person person, long time) {
    double cumulativeStress = 0;

    // Childhood adversity
    if (person.attributes.get("childhood_ses") < 0.3) {
      cumulativeStress += 0.2;
      applyEarlyLifeHealthEffects(person);
    }

    // Accumulation over time
    int yearsInPoverty = calculateYearsInPoverty(person, time);
    cumulativeStress += yearsInPoverty * 0.05;

    // Apply allostatic load
    if (cumulativeStress > 0.5) {
      accelerateChronicDiseaseOnset(person, cumulativeStress);
    }
  }
}
```

### 4.2 Social Network Effects

Model health behaviors spreading through social networks:

```json
{
  "social_network_health_effects": {
    "obesity_social_contagion": {
      "friend_influence": 0.171,
      "sibling_influence": 0.25,
      "spouse_influence": 0.37
    },
    "smoking_cessation_social_support": {
      "with_support": 0.35,
      "without_support": 0.15
    }
  }
}
```

### 4.3 Intergenerational Effects

```java
public class IntergenerationalHealth {

  public void applyParentalHealthEffects(Person child, Person mother, Person father) {
    // Maternal health during pregnancy
    if (mother.attributes.get("gestational_diabetes") == true) {
      child.attributes.put("diabetes_risk_multiplier", 1.4);
    }

    // Parental SES effects
    double parentalSES = (mother.getSES() + father.getSES()) / 2;
    child.attributes.put("childhood_ses", parentalSES);

    // Epigenetic factors (simplified)
    if (parentalSES < 0.3) {
      child.attributes.put("stress_response_dysregulation", true);
    }
  }
}
```

## 5. Configuration and Control

### 5.1 Master Configuration File

Create `synthea_health_disparities.properties`:

```properties
# Master switch for health disparities modeling
health_disparities.enabled = true

# Individual factor toggles
health_disparities.use_racial_disparities = true
health_disparities.use_ses_disparities = true
health_disparities.use_geographic_disparities = true
health_disparities.use_environmental_factors = true

# Disparity magnitude controls (0.0 = no disparity, 1.0 = full real-world disparity)
health_disparities.racial_disparity_magnitude = 0.8
health_disparities.ses_disparity_magnitude = 0.9
health_disparities.geographic_disparity_magnitude = 0.7

# Data sources
health_disparities.use_cdc_data = true
health_disparities.use_census_data = true
health_disparities.custom_disparities_file = ./custom_disparities.json
```

### 5.2 Validation Framework

```java
public class DisparityValidator {

  public void validatePopulationHealth(Population population) {
    // Calculate observed disparities
    Map<String, Double> obesityByRace = calculateOutcomeByDemographic(
      population, "obesity", "race"
    );

    // Compare to expected disparities
    validateDisparity(obesityByRace, expectedObesityByRace, 0.1);

    // Generate report
    generateDisparityReport(population);
  }

  private void generateDisparityReport(Population population) {
    // Create detailed HTML report showing:
    // - Observed vs expected disparities
    // - Statistical significance tests
    // - Visualization of health outcomes by demographics
  }
}
```

## 6. Example Implementation: Diabetes Disparities

### 6.1 Complete Module Example

```json
{
  "name": "Diabetes with Demographic Disparities",
  "states": {
    "Initial": {
      "type": "Initial",
      "direct_transition": "Assign_Demographics"
    },

    "Assign_Demographics": {
      "type": "CallSubmodule",
      "submodule": "demographics/comprehensive_demographics",
      "direct_transition": "Calculate_Diabetes_Risk"
    },

    "Calculate_Diabetes_Risk": {
      "type": "SetAttribute",
      "attribute": "diabetes_risk_score",
      "expression": {
        "type": "calculated",
        "formula": "base_risk * race_multiplier * ses_multiplier * bmi_factor * age_factor * family_history_factor"
      },
      "direct_transition": "Annual_Diabetes_Check"
    },

    "Annual_Diabetes_Check": {
      "type": "Delay",
      "exact": {"quantity": 1, "unit": "years"},
      "conditional_transition": [
        {
          "condition": {
            "condition_type": "Attribute",
            "attribute": "diabetes",
            "operator": "is not nil"
          },
          "transition": "Living_With_Diabetes"
        },
        {
          "transition": "Diabetes_Risk_Evaluation"
        }
      ]
    },

    "Diabetes_Risk_Evaluation": {
      "type": "Simple",
      "complex_transition": [
        {
          "condition": {
            "condition_type": "And",
            "conditions": [
              {
                "condition_type": "Attribute",
                "attribute": "race",
                "operator": "==",
                "value": "native_american"
              },
              {
                "condition_type": "Attribute",
                "attribute": "ses_quintile",
                "operator": "<=",
                "value": 2
              },
              {
                "condition_type": "Vital Sign",
                "vital_sign": "BMI",
                "operator": ">=",
                "value": 25
              }
            ]
          },
          "distributions": [
            {
              "distribution": 0.08,  // 8% annual risk for highest risk group
              "transition": "Onset_Diabetes"
            },
            {
              "distribution": 0.92,
              "transition": "Annual_Diabetes_Check"
            }
          ]
        },
        {
          "condition": {
            "condition_type": "And",
            "conditions": [
              {
                "condition_type": "Attribute",
                "attribute": "race",
                "operator": "==",
                "value": "white"
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
              "distribution": 0.015,  // 1.5% annual risk for lowest risk group
              "transition": "Onset_Diabetes"
            },
            {
              "distribution": 0.985,
              "transition": "Annual_Diabetes_Check"
            }
          ]
        }
      ]
    }
  }
}
```

## 7. Data Sources and Evidence Base

### 7.1 Recommended Data Sources

1. **CDC WONDER Database**: Mortality and morbidity by demographics
2. **NHANES**: National health examination survey data
3. **BRFSS**: Behavioral risk factors by state and demographics
4. **ACS**: American Community Survey for socioeconomic data
5. **County Health Rankings**: County-level health determinants
6. **AHRQ Healthcare Cost and Utilization Project**: Healthcare disparities

### 7.2 Key References

- Williams DR, et al. "Racism and Health: Evidence and Needed Research." Annual Review of Public Health 2019
- Chetty R, et al. "The Association Between Income and Life Expectancy in the United States" JAMA 2016
- CDC Health Disparities and Inequalities Report (CHDIR)
- WHO Commission on Social Determinants of Health Final Report

## 8. Testing and Validation

### 8.1 Unit Tests

```java
@Test
public void testDiabetesDisparityByRace() {
  // Generate 10,000 patients with demographic disparities enabled
  Population pop = generator.generatePopulation(10000);

  Map<String, Double> diabetesRates = calculateDiabetesRatesByRace(pop);

  // Assert expected disparities within tolerance
  assertEquals(0.13, diabetesRates.get("native_american"), 0.02);
  assertEquals(0.11, diabetesRates.get("black"), 0.02);
  assertEquals(0.09, diabetesRates.get("hispanic"), 0.02);
  assertEquals(0.07, diabetesRates.get("white"), 0.02);
  assertEquals(0.08, diabetesRates.get("asian"), 0.02);
}
```

### 8.2 Population Validation

```bash
# Generate large population for statistical validation
./run_synthea -p 100000 --health_disparities.enabled=true

# Run validation script
python validate_health_disparities.py ./output/

# Generate disparity report
python generate_disparity_report.py ./output/ --output=disparity_report.html
```

## 9. Ethical Guidelines

### 9.1 Documentation Requirements

Every module implementing health disparities must include:

```json
{
  "remarks": [
    "This module implements documented health disparities based on CDC and peer-reviewed data.",
    "These are statistical patterns, not deterministic rules.",
    "Individual patient outcomes will vary.",
    "Purpose: Create realistic synthetic data for health equity research and intervention testing.",
    "Data sources: [List specific sources and years]"
  ]
}
```

### 9.2 Use Case Restrictions

Appropriate uses:
- Training bias-aware AI systems
- Testing health equity interventions
- Educational demonstrations
- Population health research

Inappropriate uses:
- Individual patient risk assessment
- Insurance underwriting
- Employment decisions
- Perpetuating discrimination

## 10. Future Enhancements

### 10.1 Machine Learning Integration

```python
class HealthDisparityLearner:
    def learn_disparities_from_real_data(self, real_population_data):
        """
        Use ML to learn complex, non-linear relationships between
        demographics and health outcomes from real population data
        """
        # Feature engineering
        features = extract_demographic_features(real_population_data)
        outcomes = extract_health_outcomes(real_population_data)

        # Train ensemble model
        model = train_gradient_boosting_model(features, outcomes)

        # Export to Synthea format
        export_to_synthea_rules(model)
```

### 10.2 Dynamic Disparity Updates

Implement a system to update disparities based on latest research:

```java
public class DisparityUpdater {
  @Scheduled(monthly)
  public void updateDisparities() {
    // Fetch latest CDC data
    Map<String, Object> latestData = fetchCDCData();

    // Update lookup tables
    updateLookupTables(latestData);

    // Validate changes
    validateDisparityChanges();

    // Version control
    commitChanges("Monthly disparity update: " + new Date());
  }
}
```

## Conclusion

Implementing demographic-based health disparities in Synthea requires careful consideration of:

1. **Accuracy**: Using evidence-based data sources
2. **Complexity**: Modeling intersectional effects
3. **Ethics**: Clear documentation and appropriate use guidelines
4. **Configurability**: Allowing users to control disparity modeling
5. **Validation**: Ensuring generated disparities match real-world patterns

This framework provides a comprehensive approach to creating synthetic patient populations that accurately reflect health inequities, enabling better research, intervention design, and healthcare system testing while maintaining ethical standards and scientific rigor.

## Appendix A: Sample Configuration Files

[Configuration examples provided in earlier sections]

## Appendix B: Statistical Methods

### B.1 Calculating Relative Risk

```python
def calculate_relative_risk(exposed_group, unexposed_group, outcome):
    risk_exposed = count_outcome(exposed_group, outcome) / len(exposed_group)
    risk_unexposed = count_outcome(unexposed_group, outcome) / len(unexposed_group)
    return risk_exposed / risk_unexposed
```

### B.2 Adjusting for Confounders

```python
def adjust_for_confounders(raw_risk, confounders):
    adjusted_risk = raw_risk
    for confounder in confounders:
        adjustment_factor = calculate_adjustment(confounder)
        adjusted_risk *= adjustment_factor
    return adjusted_risk
```

## Appendix C: Module Templates

[Provided throughout the document]