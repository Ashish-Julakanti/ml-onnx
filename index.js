'use strict';
// Page logic for the two ONNX models (same pattern as the class example:
// build a Float32Array -> ort.Tensor -> session.run({ input1 }) -> results.output1.data).

const HOUSING_MODEL_URL = './housing_regressor.onnx';
const CHURN_MODEL_URL   = './churn_classifier.onnx';
const HOUSING_CAP = 500001;   // the training target is capped at this value

// Category orders must match the notebooks exactly (one-hot column order).
const OCEAN    = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'];
const INTERNET = ['DSL', 'Fiber optic', 'No'];
const CONTRACT = ['Month-to-month', 'One year', 'Two year'];
const PAYMENT  = ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'];

// ---- generated from the notebooks' summary JSON files ----
const CHURN_THRESHOLD = 0.56;   // tuned on the validation set for best F1
const HOUSING_FACTS = "Model: ResidualNet (PyTorch). On 3,096 held-out block groups: R² = 0.82, RMSE = $48,803, MAE = $33,446.";
const CHURN_FACTS = "Model: ResidualNet (PyTorch), decision threshold 0.56. On 1,057 held-out customers: F1 for churn = 0.63, recall = 0.73, precision = 0.56, accuracy = 0.78.";
// Held-out test rows, with the prediction ONNX Runtime (Python) gave for each
const HOUSING_SAMPLES = [
  {
    "inputs": {
      "longitude": -119.65,
      "latitude": 36.51,
      "housing_median_age": 30.0,
      "total_rooms": 1671.0,
      "total_bedrooms": 319.0,
      "population": 966.0,
      "households": 282.0,
      "median_income": 3.1333,
      "ocean_proximity": "INLAND"
    },
    "actual": 100000.0,
    "onnx_pred": 76645.01
  },
  {
    "inputs": {
      "longitude": -117.01,
      "latitude": 33.97,
      "housing_median_age": 18.0,
      "total_rooms": 4775.0,
      "total_bedrooms": 886.0,
      "population": 1868.0,
      "households": 836.0,
      "median_income": 2.3355,
      "ocean_proximity": "INLAND"
    },
    "actual": 118800.0,
    "onnx_pred": 109054.16
  },
  {
    "inputs": {
      "longitude": -118.37,
      "latitude": 34.15,
      "housing_median_age": 29.0,
      "total_rooms": 2630.0,
      "total_bedrooms": 617.0,
      "population": 1071.0,
      "households": 573.0,
      "median_income": 3.3669,
      "ocean_proximity": "<1H OCEAN"
    },
    "actual": 376100.0,
    "onnx_pred": 375534.59
  }
];
const CHURN_SAMPLES = [
  {
    "raw": {
      "gender": "Male",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 17,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 45.05,
      "TotalCharges": "770.6"
    },
    "actual": 1,
    "onnx_prob": 0.68049
  },
  {
    "raw": {
      "gender": "Female",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 69.6,
      "TotalCharges": "69.6"
    },
    "actual": 1,
    "onnx_prob": 0.878844
  },
  {
    "raw": {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 43,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "No",
      "Contract": "One year",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Mailed check",
      "MonthlyCharges": 56.35,
      "TotalCharges": "2391.15"
    },
    "actual": 0,
    "onnx_prob": 0.221226
  }
];


// ---------- encoding: raw answers -> model input vector ----------

// 13 inputs: 8 numeric columns + 5 one-hot ocean_proximity columns
function encodeHousing(v) {
  const x = new Float32Array(13);
  x[0] = Number(v.longitude);
  x[1] = Number(v.latitude);
  x[2] = Number(v.housing_median_age);
  x[3] = Number(v.total_rooms);
  x[4] = Number(v.total_bedrooms);
  x[5] = Number(v.population);
  x[6] = Number(v.households);
  x[7] = Number(v.median_income);
  const k = OCEAN.indexOf(v.ocean_proximity);
  if (k < 0) throw new Error('Unknown ocean_proximity: ' + v.ocean_proximity);
  x[8 + k] = 1;
  return x;
}

// 26 inputs: 3 continuous + 13 binary + 3 + 3 + 4 one-hot columns
function encodeChurn(v) {
  const yes = (s) => (s === 'Yes' || s === true || s === 1 || s === '1') ? 1 : 0;
  const num = (s) => { const n = parseFloat(String(s).trim()); return Number.isFinite(n) ? n : 0; };
  const x = new Float32Array(26);
  let i = 0;
  x[i++] = num(v.tenure);
  x[i++] = num(v.MonthlyCharges);
  x[i++] = num(v.TotalCharges);                 // blank for brand-new customers -> 0
  x[i++] = num(v.SeniorCitizen);
  x[i++] = v.gender === 'Female' ? 1 : 0;
  for (const c of ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']) {
    x[i++] = yes(v[c]);                          // 'No internet service' / 'No phone service' -> 0
  }
  for (const [list, key] of [[INTERNET, 'InternetService'], [CONTRACT, 'Contract'], [PAYMENT, 'PaymentMethod']]) {
    const k = list.indexOf(v[key]);
    if (k < 0) throw new Error('Unknown ' + key + ': ' + v[key]);
    x[i + k] = 1;
    i += list.length;
  }
  return x;
}

// ---------- inference ----------

const sessions = {};
function getSession(url) {
  if (!sessions[url]) sessions[url] = ort.InferenceSession.create(url);
  return sessions[url];
}

async function predictHousing(v) {
  const session = await getSession(HOUSING_MODEL_URL);
  const tensorX = new ort.Tensor('float32', encodeHousing(v), [1, 13]);
  const results = await session.run({ input1: tensorX });
  return results.output1.data[0];               // USD
}

async function predictChurn(v) {
  const session = await getSession(CHURN_MODEL_URL);
  const tensorX = new ort.Tensor('float32', encodeChurn(v), [1, 26]);
  const results = await session.run({ input1: tensorX });
  return results.output1.data[1];               // P(churn); data[0] is P(no churn)
}

// ---------- page ----------

const usd = (n) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
const pct = (p) => Math.round(p * 100) + '%';
const $ = (id) => document.getElementById(id);

function readNumber(id, label, errors) {
  const el = $(id);
  const n = parseFloat(el.value);
  const ok = el.value.trim() !== '' && Number.isFinite(n);
  el.setAttribute('aria-invalid', ok ? 'false' : 'true');
  if (!ok) errors.push(label);
  return n;
}

function readHousing() {
  const errors = [];
  const v = {
    longitude:          readNumber('h-longitude', 'longitude', errors),
    latitude:           readNumber('h-latitude', 'latitude', errors),
    housing_median_age: readNumber('h-age', 'median age of homes', errors),
    total_rooms:        readNumber('h-rooms', 'total rooms', errors),
    total_bedrooms:     readNumber('h-bedrooms', 'total bedrooms', errors),
    population:         readNumber('h-population', 'population', errors),
    households:         readNumber('h-households', 'households', errors),
    median_income:      readNumber('h-income', 'median household income', errors),
    ocean_proximity:    $('h-ocean').value,
  };
  return { v, errors };
}

function readChurn() {
  const errors = [];
  const yn = (id) => ($(id).checked && !$(id).disabled) ? 'Yes' : 'No';
  const v = {
    tenure:           readNumber('c-tenure', 'tenure', errors),
    MonthlyCharges:   readNumber('c-monthly', 'monthly charges', errors),
    TotalCharges:     readNumber('c-total', 'total charges', errors),
    SeniorCitizen:    $('c-senior').checked ? 1 : 0,
    gender:           $('c-gender').value,
    Partner:          yn('c-partner'),
    Dependents:       yn('c-dependents'),
    PhoneService:     yn('c-phone'),
    MultipleLines:    yn('c-multiline'),
    OnlineSecurity:   yn('c-security'),
    OnlineBackup:     yn('c-backup'),
    DeviceProtection: yn('c-protection'),
    TechSupport:      yn('c-support'),
    StreamingTV:      yn('c-tv'),
    StreamingMovies:  yn('c-movies'),
    PaperlessBilling: yn('c-paperless'),
    InternetService:  $('c-internet').value,
    Contract:         $('c-contract').value,
    PaymentMethod:    $('c-payment').value,
  };
  return { v, errors };
}

function showError(outId, msg) {
  $(outId).innerHTML = `<p class="empty" style="color:var(--leave)">${msg}</p>`;
}

function renderHousing(pred, check) {
  const shown = Math.max(pred, 0);
  const pos = Math.min(Math.max(pred / HOUSING_CAP, 0), 1) * 100;
  let note = 'The training data stops at $500,001, so the scale ends there.';
  if (pred >= HOUSING_CAP) note = 'This is at or above the $500,001 cap in the training data; read it as “$500k or more”.';
  if (pred < 0) note = 'The inputs are far outside the training data, so the model gives a meaningless negative value.';
  $('house-out').innerHTML = `
    <div class="big">${usd(shown)}</div>
    <p class="caption">estimated median house value</p>
    <div class="scale" aria-hidden="true"><div class="fill" style="width:${pos.toFixed(1)}%"></div></div>
    <div class="scale-ends"><span>$0</span><span>$250k</span><span>$500k</span></div>
    <p class="note">${note}</p>
    ${check ? `<p class="check-row">Test row: actual value ${usd(check.actual)}. The notebook’s ONNX Runtime run gave ${usd(check.onnx_pred)}.</p>` : ''}`;
}

function renderChurn(prob, check) {
  const leave = prob >= CHURN_THRESHOLD;
  const cls = leave ? 'leave' : 'stay';
  $('churn-out').innerHTML = `
    <div class="big">${pct(prob)}</div>
    <p class="caption">chance of churning: <span class="verdict ${cls}">${leave ? 'likely to churn' : 'likely to stay'}</span></p>
    <div class="scale" aria-hidden="true">
      <div class="fill ${cls}" style="width:${(prob * 100).toFixed(1)}%"></div>
      <div class="tick" style="left:calc(${(CHURN_THRESHOLD * 100).toFixed(0)}% - 1px)"></div>
      <span class="tick-label" style="left:${(CHURN_THRESHOLD * 100).toFixed(0)}%">threshold ${CHURN_THRESHOLD.toFixed(2)}</span>
    </div>
    <div class="scale-ends"><span>0%</span><span>50%</span><span>100%</span></div>
    <p class="note">The model flags a customer once the probability reaches ${CHURN_THRESHOLD.toFixed(2)}, the threshold that gave the best F1 score on the validation set.</p>
    ${check ? `<p class="check-row">Test customer: actually ${check.actual ? 'churned' : 'stayed'}. The notebook’s ONNX Runtime run gave ${(check.onnx_prob * 100).toFixed(1)}%.</p>` : ''}`;
}

let houseCheck = null, churnCheck = null;

async function runHousing(e) {
  if (e) e.preventDefault();
  const { v, errors } = readHousing();
  if (errors.length) return showError('house-out', 'Enter a number for ' + errors.join(', ') + '.');
  $('house-run').disabled = true;
  try { renderHousing(await predictHousing(v), houseCheck); }
  catch (err) { console.error(err); showError('house-out', 'The model did not run: ' + err.message); }
  finally { $('house-run').disabled = false; houseCheck = null; }
}

async function runChurn(e) {
  if (e) e.preventDefault();
  const { v, errors } = readChurn();
  if (errors.length) return showError('churn-out', 'Enter a number for ' + errors.join(', ') + '.');
  $('churn-run').disabled = true;
  try { renderChurn(await predictChurn(v), churnCheck); }
  catch (err) { console.error(err); showError('churn-out', 'The model did not run: ' + err.message); }
  finally { $('churn-run').disabled = false; churnCheck = null; }
}

function syncServiceToggles() {
  const noNet = $('c-internet').value === 'No';
  document.querySelectorAll('label.check.net').forEach((lab) => {
    lab.querySelector('input').disabled = noNet;
    lab.classList.toggle('disabled', noNet);
  });
  const noPhone = !$('c-phone').checked;
  $('c-multiline').disabled = noPhone;
  $('c-multiline').closest('label').classList.toggle('disabled', noPhone);
}

function loadHousingSample(s) {
  const r = s.inputs;
  $('h-longitude').value = r.longitude; $('h-latitude').value = r.latitude;
  $('h-age').value = r.housing_median_age; $('h-rooms').value = r.total_rooms;
  $('h-bedrooms').value = r.total_bedrooms; $('h-population').value = r.population;
  $('h-households').value = r.households; $('h-income').value = r.median_income;
  $('h-ocean').value = r.ocean_proximity;
  houseCheck = s;
  runHousing();
}

function loadChurnSample(s) {
  const r = s.raw;
  const set = (id, val) => { $(id).checked = val === 'Yes' || val === 1; };
  $('c-tenure').value = r.tenure; $('c-monthly').value = r.MonthlyCharges;
  $('c-total').value = String(r.TotalCharges).trim() === '' ? 0 : r.TotalCharges;
  $('c-contract').value = r.Contract; $('c-payment').value = r.PaymentMethod;
  $('c-gender').value = r.gender; $('c-internet').value = r.InternetService;
  set('c-senior', r.SeniorCitizen); set('c-partner', r.Partner); set('c-dependents', r.Dependents);
  set('c-paperless', r.PaperlessBilling); set('c-phone', r.PhoneService); set('c-multiline', r.MultipleLines);
  set('c-security', r.OnlineSecurity); set('c-backup', r.OnlineBackup); set('c-protection', r.DeviceProtection);
  set('c-support', r.TechSupport); set('c-tv', r.StreamingTV); set('c-movies', r.StreamingMovies);
  syncServiceToggles();
  churnCheck = s;
  runChurn();
}

async function init() {
  ort.env.wasm.numThreads = 1;   // GitHub Pages is not cross-origin isolated, so use one thread
  $('house-form').addEventListener('submit', runHousing);
  $('churn-form').addEventListener('submit', runChurn);
  $('c-internet').addEventListener('change', syncServiceToggles);
  $('c-phone').addEventListener('change', syncServiceToggles);
  syncServiceToggles();

  HOUSING_SAMPLES.forEach((s, k) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'quiet'; b.textContent = 'Row ' + (k + 1);
    b.addEventListener('click', () => loadHousingSample(s));
    $('house-samples').appendChild(b); $('house-samples').append(' ');
  });
  CHURN_SAMPLES.forEach((s, k) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'quiet'; b.textContent = 'Customer ' + (k + 1);
    b.addEventListener('click', () => loadChurnSample(s));
    $('churn-samples').appendChild(b); $('churn-samples').append(' ');
  });

  $('house-facts').textContent = HOUSING_FACTS;
  $('churn-facts').textContent = CHURN_FACTS;

  try {
    await Promise.all([getSession(HOUSING_MODEL_URL), getSession(CHURN_MODEL_URL)]);
    $('status').textContent = 'Both models are loaded and ready.';
  } catch (err) {
    console.error(err);
    $('status').className = 'error';
    $('status').textContent = 'The model files did not load. Open this page from GitHub Pages or a local web server ' +
      '(python -m http.server); browsers block .onnx files opened directly from disk.';
  }
}

if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', init);
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { encodeHousing, encodeChurn, OCEAN, INTERNET, CONTRACT, PAYMENT,
                     get CHURN_THRESHOLD() { return CHURN_THRESHOLD; }, HOUSING_SAMPLES, CHURN_SAMPLES };
}
