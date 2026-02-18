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
