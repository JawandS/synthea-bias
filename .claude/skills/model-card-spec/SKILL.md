---
name: model-card-spec
description: Use when helping a user create, fill out, or generate model card JSON for an ML model — covers field structure, required vs optional fields, examples, and validation.
---

# Model Card JSON Specification

## Overview

Produces valid JSON for the model-card app (HuggingFace standard). **Workflow: start from the baseline JSON below, customize per model, validate.** Only two fields are required; everything else improves documentation quality.

## Required Fields

| Field | Rule |
|-------|------|
| `model_id` | Non-empty string |
| `developers` | Non-empty string |

---

## Workflow

**DO NOT build JSON from scratch. DO NOT ask questions you can answer by reading files.**

```
1. Dispatch an exploration subagent to scan the codebase for model facts:
     - README, paper references, config files (config.json, model_card.md, setup.py,
       pyproject.toml), training scripts, eval scripts, LICENSE
     - Collect: model name, architecture, task, language, license, base model,
       dataset names, metrics, URLs, authors, citation blocks
   Use subagents to keep exploration results out of your main context window.

2. Start from the baseline JSON below — copy it, fill in everything discovered in step 1.

3. Only after exhausting file evidence, ask the user — in a single message — about
   the specific fields that remain as placeholders. Never ask for information already
   in the codebase.

4. Apply the user's answers. Leave brackets like [N] only for fields the user
   explicitly declined to fill.

5. Validate:
     node .claude/skills/model-card-spec/validate.js model-card.json
   Fix every reported error.

6. Output the final JSON.
```

---

## Baseline JSON

Copy this verbatim and replace values. Every field is already valid.

```json
{
  "model_id": "org/model-name",
  "developers": "Model Author(s), Organization",
  "model_summary": "A [architecture]-based model for [task], achieving [headline metric] on [benchmark].",
  "model_description": "This is a [base model]-based model fine-tuned for [task]. The model uses [architecture details]. It has been optimized for [optimization goal].",
  "funded_by": "Funding source or N/A",
  "shared_by": "Organization or individual sharing the model",
  "model_type": "Transformer, Supervised Learning",
  "language": "English (en)",
  "license": "apache-2.0",
  "base_model": "base-model-id",
  "model_sources": {
    "repo": "https://github.com/org/model-repo",
    "paper": "https://arxiv.org/abs/0000.00000",
    "demo": "https://huggingface.co/spaces/org/model-demo"
  },
  "uses": {
    "direct_use": "This model can be used directly for [task] on [input type]. It accepts [input] and outputs [output]. No additional fine-tuning is required for [general use case].",
    "downstream_use": "The model can be fine-tuned for [specific domains]. It can also be integrated into larger [pipeline type] pipelines.",
    "out_of_scope_use": "This model should not be used for [prohibited use]. It is not suitable for [limitation]. The model may produce unreliable results on [edge case]."
  },
  "bias_risks": {
    "bias_risks_limitations": "The model may exhibit bias toward [demographic/perspective]. Performance may degrade on [underrepresented input type]. The model can produce [failure mode] and should not be used as the sole decision-making tool in high-stakes scenarios.",
    "bias_recommendations": "Users should validate model outputs with domain experts. Consider implementing confidence thresholds and human-in-the-loop review. Monitor performance across [relevant groups]. Regular retraining on diverse data is recommended."
  },
  "get_started_code": "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n\ntokenizer = AutoTokenizer.from_pretrained(\"org/model-name\")\nmodel = AutoModelForSequenceClassification.from_pretrained(\"org/model-name\")\n\ntext = \"Example input\"\ninputs = tokenizer(text, return_tensors=\"pt\")\noutputs = model(**inputs)",
  "training_details": {
    "training_data": "Trained on [dataset name(s)] totaling approximately [N] examples. See dataset card at [URL] for details.",
    "preprocessing": "Text was [normalization steps] using [tokenizer] with vocabulary size [N]. [Special tokens]. Maximum sequence length was set to [N] tokens.",
    "training_regime": "fp16 mixed precision",
    "speeds_sizes_times": "Training completed in [X hours] on [N× GPU model]. Final checkpoint size is [X MB]. Average throughput was [N samples/second]."
  },
  "evaluation": {
    "testing_data": "Evaluated on [test set name(s)]. Total test set size: [N] examples.",
    "testing_factors": "Performance was evaluated across [factors: text length, domain, intensity, etc.].",
    "testing_metrics": "Primary metrics: [Accuracy, Precision, Recall, F1-score]. Additional metrics: [ROC-AUC, etc.].",
    "results": "Overall test accuracy: [X]%. F1-score (weighted): [X]. Performance breakdown: [Domain A] ([X]%), [Domain B] ([X]%).",
    "results_summary": "The model achieves [headline result] on the test set, with [strength] but [weakness/limitation]."
  },
  "environmental_impact": {
    "hardware_type": "N× GPU Model",
    "hours_used": "X hours",
    "cloud_provider": "AWS",
    "cloud_region": "us-east-1",
    "co2_emitted": "X kg CO2eq"
  },
  "technical_specs": {
    "model_specs": "[N]-layer [architecture] with [hidden dim] hidden dimensions, [N] attention heads, and [N]M parameters.",
    "compute_infrastructure": "[Cloud provider] [instance type] with [GPU config].",
    "hardware_requirements": "Minimum [N] GB GPU memory for inference. Recommended: [GPU model] or better.",
    "software": "PyTorch [version], transformers [version], CUDA [version], Python [version]+"
  },
  "citation": {
    "citation_bibtex": "@article{author2024model,\n  title={Model Title},\n  author={Author, First and Author, Second},\n  journal={arXiv preprint arXiv:0000.00000},\n  year={2024}\n}",
    "citation_apa": "Author, F., & Author, S. (2024). Model Title. arXiv preprint arXiv:0000.00000."
  },
  "additional_info": {
    "model_examination": "[Interpretability methods] were used to interpret model predictions.",
    "glossary": "**[Term]**: [Definition]. **[Term]**: [Definition].",
    "more_information": "For details, see [URL]. Training code available at [URL].",
    "model_card_authors": "First Author, Second Author",
    "model_card_contact": "team@organization.com"
  },
  "metadata": {
    "license": "apache-2.0",
    "language": "en",
    "base_model": "base-model-id",
    "library_name": "transformers",
    "pipeline_tag": "text-classification",
    "tags": ["task-tag", "architecture-tag", "domain-tag"],
    "datasets": ["dataset-name"],
    "metrics": ["accuracy", "f1"],
    "inference": true
  }
}
```

---

## Field Quick Reference

| Section | Key fields | Notes |
|---------|-----------|-------|
| Top-level | `model_id`*, `developers`*, `model_summary`, `model_type`, `language`, `license`, `base_model` | * = required |
| `model_sources` | `repo`, `paper`, `demo` | Must be valid URLs |
| `uses` | `direct_use`, `downstream_use`, `out_of_scope_use` | |
| `bias_risks` | `bias_risks_limitations`, `bias_recommendations` | |
| `get_started_code` | string | Use `\n` for newlines |
| `training_details` | `training_data`, `preprocessing`, `training_regime`, `speeds_sizes_times` | `training_regime`: fp32/fp16/bf16 |
| `evaluation` | `testing_data`, `testing_factors`, `testing_metrics`, `results`, `results_summary` | |
| `environmental_impact` | `hardware_type`, `hours_used`, `cloud_provider`, `cloud_region`, `co2_emitted` | |
| `technical_specs` | `model_specs`, `compute_infrastructure`, `hardware_requirements`, `software` | |
| `citation` | `citation_bibtex`, `citation_apa` | |
| `additional_info` | `model_examination`, `glossary`, `more_information`, `model_card_authors`, `model_card_contact` | |
| `metadata` | `license`, `language`, `base_model`, `library_name`, `pipeline_tag`, `tags[]`, `datasets[]`, `metrics[]`, `inference` | arrays = string[] |

---

## Validation

Save the script below as `validate.js` and run:

```bash
node validate.js model-card.json
```

Fix every reported error before delivering the final JSON. Exits 0 on success, 1 with a field-by-field error list on failure.

```js
#!/usr/bin/env node
/**
 * validate.js — Validates model card JSON against the ModelCard schema
 * Usage: node validate.js <path-to-model-card.json>
 * No dependencies required — pure Node.js.
 */

const fs = require('fs');
const path = require('path');

function isValidUrl(str) {
  try { const u = new URL(str); return u.protocol === 'https:' || u.protocol === 'http:'; }
  catch { return false; }
}
function isNonEmptyString(val) { return typeof val === 'string' && val.trim().length > 0; }
function isStringArray(val) { return Array.isArray(val) && val.every(i => typeof i === 'string'); }

function validate(data) {
  const errors = [];

  if (!isNonEmptyString(data.model_id))
    errors.push('model_id: required non-empty string (got ' + JSON.stringify(data.model_id) + ')');
  if (!isNonEmptyString(data.developers))
    errors.push('developers: required non-empty string (got ' + JSON.stringify(data.developers) + ')');

  for (const key of ['model_summary','model_description','funded_by','shared_by','model_type','language','license','base_model','get_started_code']) {
    if (data[key] != null && typeof data[key] !== 'string')
      errors.push(`${key}: must be a string (got ${typeof data[key]})`);
  }

  function checkObject(obj, parent, keys) {
    if (typeof obj !== 'object' || Array.isArray(obj)) { errors.push(`${parent}: must be an object`); return; }
    for (const k of keys)
      if (obj[k] != null && typeof obj[k] !== 'string') errors.push(`${parent}.${k}: must be a string`);
  }

  if (data.model_sources != null) {
    checkObject(data.model_sources, 'model_sources', []);
    if (typeof data.model_sources === 'object' && !Array.isArray(data.model_sources)) {
      for (const f of ['repo','paper','demo']) {
        const v = data.model_sources[f];
        if (v != null && v !== '') {
          if (typeof v !== 'string') errors.push(`model_sources.${f}: must be a string URL`);
          else if (!isValidUrl(v)) errors.push(`model_sources.${f}: invalid URL "${v}"`);
        }
      }
    }
  }

  if (data.uses != null)               checkObject(data.uses, 'uses', ['direct_use','downstream_use','out_of_scope_use']);
  if (data.bias_risks != null)         checkObject(data.bias_risks, 'bias_risks', ['bias_risks_limitations','bias_recommendations']);
  if (data.training_details != null)   checkObject(data.training_details, 'training_details', ['training_data','preprocessing','training_regime','speeds_sizes_times']);
  if (data.evaluation != null)         checkObject(data.evaluation, 'evaluation', ['testing_data','testing_factors','testing_metrics','results','results_summary']);
  if (data.environmental_impact != null) checkObject(data.environmental_impact, 'environmental_impact', ['hardware_type','hours_used','cloud_provider','cloud_region','co2_emitted']);
  if (data.technical_specs != null)    checkObject(data.technical_specs, 'technical_specs', ['model_specs','compute_infrastructure','hardware_requirements','software']);
  if (data.citation != null)           checkObject(data.citation, 'citation', ['citation_bibtex','citation_apa']);
  if (data.additional_info != null)    checkObject(data.additional_info, 'additional_info', ['model_examination','glossary','more_information','model_card_authors','model_card_contact']);

  if (data.metadata != null) {
    checkObject(data.metadata, 'metadata', ['license','language','base_model','library_name','pipeline_tag']);
    if (typeof data.metadata === 'object' && !Array.isArray(data.metadata)) {
      for (const k of ['tags','datasets','metrics'])
        if (data.metadata[k] != null && !isStringArray(data.metadata[k])) errors.push(`metadata.${k}: must be an array of strings`);
      if (data.metadata.inference != null && typeof data.metadata.inference !== 'boolean')
        errors.push(`metadata.inference: must be a boolean (got ${typeof data.metadata.inference})`);
    }
  }

  const known = new Set(['card_data','model_id','model_summary','model_description','developers','funded_by','shared_by','model_type','language','license','base_model','model_sources','uses','bias_risks','get_started_code','training_details','evaluation','environmental_impact','technical_specs','citation','additional_info','metadata']);
  const unknown = Object.keys(data).filter(k => !known.has(k));
  if (unknown.length) errors.push(`Unknown top-level keys (silently dropped by app): ${unknown.join(', ')}`);

  return errors;
}

const filePath = process.argv[2];
if (!filePath) { console.error('Usage: node validate.js <path-to-model-card.json>'); process.exit(1); }
const absPath = path.resolve(filePath);
if (!fs.existsSync(absPath)) { console.error(`File not found: ${absPath}`); process.exit(1); }

let data;
try { data = JSON.parse(fs.readFileSync(absPath, 'utf-8')); }
catch (err) { console.error(`Failed to parse JSON: ${err.message}`); process.exit(1); }

if (typeof data !== 'object' || Array.isArray(data) || data === null) {
  console.error('Invalid: root must be a JSON object'); process.exit(1);
}

const errors = validate(data);
if (errors.length === 0) { console.log('✓ Valid model card JSON'); process.exit(0); }
else {
  console.error(`✗ Invalid model card JSON — ${errors.length} error(s):\n`);
  errors.forEach(e => console.error(`  • ${e}`));
  process.exit(1);
}
```
