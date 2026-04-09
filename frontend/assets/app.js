/* Smart-Factory AI — frontend logic
   Pure vanilla JS, no deps. Talks to the FastAPI backend on the same origin. */

const API = location.origin.replace(/\/$/, '');

// -----------------------------------------------------------------
// Language switcher
// -----------------------------------------------------------------
function setLang(lang) {
  document.body.setAttribute('data-lang', lang);
  document.getElementById('lang-en').classList.toggle('active', lang === 'en');
  document.getElementById('lang-uz').classList.toggle('active', lang === 'uz');
  try { localStorage.setItem('sfa-lang', lang); } catch (e) {}
}
try {
  const saved = localStorage.getItem('sfa-lang');
  if (saved === 'en' || saved === 'uz') setLang(saved);
} catch (e) {}

// -----------------------------------------------------------------
// Server health ping
// -----------------------------------------------------------------
(async () => {
  const chip = document.getElementById('server-status');
  try {
    const r = await fetch(`${API}/api/health`);
    if (!r.ok) throw new Error();
    chip.classList.add('online');
    chip.querySelector('[lang="en"]').textContent = `Server online · ${location.host}`;
    chip.querySelector('[lang="uz"]').textContent = `Server ishlayapti · ${location.host}`;
  } catch (e) {
    chip.classList.add('offline');
    chip.querySelector('[lang="en"]').textContent = 'Server offline — run backend';
    chip.querySelector('[lang="uz"]').textContent = "Server o'chiq — backendni ishga tushiring";
  }
})();

// -----------------------------------------------------------------
// Dropzone helper
// -----------------------------------------------------------------
function wireDrop(dropId, fileId, onFile) {
  const drop = document.getElementById(dropId);
  const file = document.getElementById(fileId);
  drop.addEventListener('click', () => file.click());
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) onFile(e.dataTransfer.files[0]);
  });
  file.addEventListener('change', (e) => {
    if (e.target.files.length > 0) onFile(e.target.files[0]);
  });
}

function showError(target, err) {
  target.innerHTML = `<div class="verdict-banner bad"><span>⚠️</span><div>${err}</div></div>`;
}

function pctFill(p) { return `<div class="bar"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div>`; }

// -----------------------------------------------------------------
// 1. Casting
// -----------------------------------------------------------------
(function casting() {
  let currentFile = null;
  const btn = document.getElementById('casting-go');
  const preview = document.getElementById('casting-preview');
  const result = document.getElementById('casting-result');

  wireDrop('casting-drop', 'casting-file', (file) => {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => { preview.src = e.target.result; preview.hidden = false; };
    reader.readAsDataURL(file);
    btn.disabled = false;
  });

  btn.addEventListener('click', async () => {
    if (!currentFile) return;
    btn.disabled = true;
    result.innerHTML = '<div class="placeholder">Running EfficientNet-B0…</div>';
    const fd = new FormData();
    fd.append('file', currentFile);
    try {
      const r = await fetch(`${API}/api/casting/predict`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = await r.json();
      const isOk = data.verdict === 'OK';
      const bannerClass = isOk ? 'ok' : 'bad';
      const bannerIcon = isOk ? '✓' : '✗';
      const bannerText = isOk ? 'ACCEPT · Casting looks good' : 'REJECT · Defect detected';
      let html = `
        <div class="verdict-banner ${bannerClass}">
          <span class="big">${bannerIcon}</span>
          <div><b>${bannerText}</b><br/><small>Predicted class: <code>${data.class_name}</code> · confidence ${(data.confidence*100).toFixed(1)}%</small></div>
        </div>
        <div class="metric-list">`;
      for (const [name, p] of Object.entries(data.probabilities)) {
        html += `<div class="prob-row">
                   <div class="label">${name} <b>${(p*100).toFixed(1)}%</b></div>
                   ${pctFill(p)}
                 </div>`;
      }
      html += `</div><div class="latency-tag">⏱ ${data.latency_ms} ms on CPU</div>`;
      result.innerHTML = html;
    } catch (e) {
      showError(result, 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();

// -----------------------------------------------------------------
// 2. PPE Hard Hat with canvas box rendering
// -----------------------------------------------------------------
(function ppe() {
  let currentFile = null;
  let loadedImg = null;
  const btn = document.getElementById('ppe-go');
  const canvas = document.getElementById('ppe-canvas');
  const ctx = canvas.getContext('2d');
  const result = document.getElementById('ppe-result');

  function drawImageOnly(img) {
    const maxW = 560, maxH = 360;
    const ratio = Math.min(maxW / img.width, maxH / img.height, 1);
    canvas.width = img.width * ratio;
    canvas.height = img.height * ratio;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.hidden = false;
    return ratio;
  }

  wireDrop('ppe-drop', 'ppe-file', (file) => {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      loadedImg = new Image();
      loadedImg.onload = () => drawImageOnly(loadedImg);
      loadedImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
    btn.disabled = false;
  });

  btn.addEventListener('click', async () => {
    if (!currentFile) return;
    btn.disabled = true;
    result.innerHTML = '<div class="placeholder">Running YOLOv8…</div>';
    const fd = new FormData();
    fd.append('file', currentFile);
    try {
      const r = await fetch(`${API}/api/ppe/predict`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = await r.json();

      // redraw image + boxes
      const ratio = drawImageOnly(loadedImg);
      for (const d of data.detections) {
        const [x1, y1, x2, y2] = d.box;
        const color = d.class === 'Hardhat' ? '#34d399' : '#f87171';
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.strokeRect(x1 * ratio, y1 * ratio, (x2 - x1) * ratio, (y2 - y1) * ratio);
        ctx.fillStyle = color;
        const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
        ctx.font = '12px ui-monospace, monospace';
        const labelW = ctx.measureText(label).width + 8;
        ctx.fillRect(x1 * ratio, y1 * ratio - 14, labelW, 14);
        ctx.fillStyle = '#07090f';
        ctx.fillText(label, x1 * ratio + 4, y1 * ratio - 3);
      }

      const isSafe = data.verdict === 'SAFE';
      const bClass = isSafe ? 'ok' : 'bad';
      const bText = isSafe
        ? `SAFE · ${data.counts.Hardhat} helmet${data.counts.Hardhat !== 1 ? 's' : ''} detected`
        : `UNSAFE · ${data.counts['NO-Hardhat']} worker(s) without a helmet`;
      let html = `<div class="verdict-banner ${bClass}"><span class="big">${isSafe ? '✓' : '⚠'}</span><div><b>${bText}</b></div></div>`;
      html += '<div class="metric-list">';
      html += `<div class="metric-row"><span class="label">Hardhat</span><span class="value">${data.counts.Hardhat || 0}</span></div>`;
      html += `<div class="metric-row"><span class="label">NO-Hardhat</span><span class="value" style="color:${data.counts['NO-Hardhat'] > 0 ? 'var(--accent-6)' : 'var(--accent-4)'}">${data.counts['NO-Hardhat'] || 0}</span></div>`;
      html += `<div class="metric-row"><span class="label">Total people</span><span class="value">${data.total_people_in_frame}</span></div>`;
      html += `<div class="metric-row"><span class="label">Conf threshold</span><span class="value">${data.conf_threshold}</span></div>`;
      html += '</div>';
      html += `<div class="latency-tag">⏱ ${data.latency_ms} ms · ${(1000/data.latency_ms).toFixed(1)} FPS</div>`;
      result.innerHTML = html;
    } catch (e) {
      showError(result, 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();

// -----------------------------------------------------------------
// 3. AI4I tabular
// -----------------------------------------------------------------
(function ai4i() {
  const btn = document.getElementById('ai4i-go');
  const result = document.getElementById('ai4i-result');

  const presets = {
    healthy:    { air: 298.1, proc: 308.6, rpm: 1551, torque: 42.8, wear: 108, type: 'M' },
    overstrain: { air: 299.3, proc: 309.8, rpm: 1282, torque: 68.4, wear: 215, type: 'L' },
    heat:       { air: 302.1, proc: 308.9, rpm: 1320, torque: 36.1, wear: 140, type: 'M' },
  };
  document.querySelectorAll('.preset-row [data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = presets[btn.dataset.preset];
      document.getElementById('ai4i-air').value = p.air;
      document.getElementById('ai4i-proc').value = p.proc;
      document.getElementById('ai4i-rpm').value = p.rpm;
      document.getElementById('ai4i-torque').value = p.torque;
      document.getElementById('ai4i-wear').value = p.wear;
      document.getElementById('ai4i-type').value = p.type;
    });
  });

  btn.addEventListener('click', async () => {
    const payload = {
      air_temp_k: parseFloat(document.getElementById('ai4i-air').value),
      process_temp_k: parseFloat(document.getElementById('ai4i-proc').value),
      rotational_speed_rpm: parseFloat(document.getElementById('ai4i-rpm').value),
      torque_nm: parseFloat(document.getElementById('ai4i-torque').value),
      tool_wear_min: parseFloat(document.getElementById('ai4i-wear').value),
      machine_type: document.getElementById('ai4i-type').value,
    };
    btn.disabled = true;
    result.innerHTML = '<div class="placeholder">Running XGBoost + LightGBM ensemble…</div>';
    try {
      const r = await fetch(`${API}/api/ai4i/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = await r.json();

      const p = data.probability_failure_ensemble;
      const fail = data.verdict === 'FAILURE LIKELY';
      const bClass = fail ? 'bad' : 'ok';
      let html = `<div class="verdict-banner ${bClass}">
        <span class="big">${fail ? '⚠' : '✓'}</span>
        <div><b>${data.verdict}</b><br/><small>Ensemble failure probability: <b>${(p*100).toFixed(2)}%</b> (threshold ${(data.threshold*100).toFixed(1)}%)</small></div>
      </div>`;
      html += `<div class="prob-row"><div class="label">XGBoost <b>${(data.probability_failure_xgboost*100).toFixed(2)}%</b></div>${pctFill(data.probability_failure_xgboost)}</div>`;
      html += `<div class="prob-row"><div class="label">LightGBM <b>${(data.probability_failure_lightgbm*100).toFixed(2)}%</b></div>${pctFill(data.probability_failure_lightgbm)}</div>`;
      html += `<div class="prob-row"><div class="label">Ensemble <b>${(p*100).toFixed(2)}%</b></div>${pctFill(p)}</div>`;

      if (data.top_feature_drivers && data.top_feature_drivers.length) {
        html += '<div class="metric-list" style="margin-top:10px">';
        html += '<div class="mcard-title" style="margin-bottom:6px">Top feature drivers (XGBoost gain)</div>';
        for (const d of data.top_feature_drivers) {
          html += `<div class="metric-row"><span class="label">${d.feature}</span><span class="value">${d.importance_pct}%</span></div>`;
        }
        html += '</div>';
      }
      html += `<div class="latency-tag">⏱ ${data.latency_ms} ms on CPU</div>`;
      result.innerHTML = html;
    } catch (e) {
      showError(result, 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();

// -----------------------------------------------------------------
// 4. NER
// -----------------------------------------------------------------
(function ner() {
  const btn = document.getElementById('ner-go');
  const text = document.getElementById('ner-text');
  const result = document.getElementById('ner-result');
  const examples = {
    1: "The hydraulic pump's drive belt broke at 08:15. A 40 mm bolt fell from the crane arm and hit the compressor housing at 6 bar. Operator at Fanuc robot line reported a loose gasket in the chiller.",
    2: "At the Samsung Pyeongtaek fab, a 120°C overheated bearing in the conveyor motor caused a 45 minute production stop. The gearbox output shaft was bent 3 mm and the drive belt snapped.",
    3: "Workers at POSCO Pohang Mill reported a ruptured hydraulic hose at the press line, leaking 18 liters of oil. A corroded flange on the cooling pipe was replaced; the forklift struck a loose bracket on the chain conveyor.",
  };
  document.querySelectorAll('[data-ner]').forEach((b) => {
    b.addEventListener('click', () => { text.value = examples[b.dataset.ner]; });
  });

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    result.innerHTML = '<div class="placeholder">Running BERT-NER…</div>';
    try {
      const r = await fetch(`${API}/api/ner/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.value }),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = await r.json();

      let html = '';
      if (data.rule_equipment && data.rule_equipment.length) {
        html += '<div class="entity-group"><div class="entity-group-title">Equipment</div>';
        for (const e of data.rule_equipment) html += `<span class="entity-tag equipment">${e}</span>`;
        html += '</div>';
      }
      if (data.rule_parts && data.rule_parts.length) {
        html += '<div class="entity-group"><div class="entity-group-title">Parts</div>';
        for (const e of data.rule_parts) html += `<span class="entity-tag part">${e}</span>`;
        html += '</div>';
      }
      if (data.rule_actions && data.rule_actions.length) {
        html += '<div class="entity-group"><div class="entity-group-title">Actions</div>';
        for (const e of data.rule_actions) html += `<span class="entity-tag action">${e}</span>`;
        html += '</div>';
      }
      if (data.rule_quantities && data.rule_quantities.length) {
        html += '<div class="entity-group"><div class="entity-group-title">Quantities</div>';
        for (const q of data.rule_quantities) html += `<span class="entity-tag quantity">${q.value} ${q.unit}</span>`;
        html += '</div>';
      }
      if (data.bert_entities && data.bert_entities.length) {
        html += '<div class="entity-group"><div class="entity-group-title">BERT entities (open-domain)</div>';
        for (const e of data.bert_entities) html += `<span class="entity-tag bert">${e.word} <small>(${e.type})</small></span>`;
        html += '</div>';
      }
      if (!html) html = '<div class="placeholder">No entities matched.</div>';
      html += `<div class="latency-tag">⏱ ${data.latency_ms} ms on CPU</div>`;
      result.innerHTML = html;
    } catch (e) {
      showError(result, 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();

// -----------------------------------------------------------------
// 5. SECOM
// -----------------------------------------------------------------
(function secom() {
  const btn = document.getElementById('secom-go');
  const text = document.getElementById('secom-text');
  const result = document.getElementById('secom-result');

  function gen(failBias) {
    // Deterministic-ish synthetic 590 vector. 'failBias' skews a random subset.
    const N = 590;
    const arr = [];
    for (let i = 0; i < N; i++) {
      // mix of scales — some large, some unit
      let v;
      if (i % 7 === 0) v = 2000 + Math.random() * 800;
      else if (i % 5 === 0) v = 100 + Math.random() * 20;
      else if (i % 3 === 0) v = Math.random() * 2;
      else v = Math.random();
      if (failBias) {
        // push ~15% of features into outliers
        if (Math.random() < 0.15) v *= (1.8 + Math.random() * 1.5);
      }
      arr.push(v.toFixed(4));
    }
    return arr.join(', ');
  }

  document.querySelector('.secom-gen-normal').addEventListener('click', () => { text.value = gen(false); });
  document.querySelector('.secom-gen-anom').addEventListener('click', () => { text.value = gen(true); });

  btn.addEventListener('click', async () => {
    const raw = text.value.trim();
    if (!raw) { showError(result, 'Paste sensor values first.'); return; }
    const values = raw.split(/[\s,;\r\n]+/).map((s) => {
      const v = parseFloat(s);
      return isFinite(v) ? v : NaN;
    });
    btn.disabled = true;
    result.innerHTML = '<div class="placeholder">Running 5-learner stack…</div>';
    try {
      const r = await fetch(`${API}/api/secom/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensor_values: values }),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = await r.json();
      const p = data.probabilities;
      const ens = p.ensemble_4tree;
      const fail = data.verdict === 'FAULT LIKELY';
      const bClass = fail ? 'bad' : 'ok';
      let html = `<div class="verdict-banner ${bClass}">
        <span class="big">${fail ? '⚠' : '✓'}</span>
        <div><b>${data.verdict}</b><br/><small>Ensemble score ${(ens*100).toFixed(2)}% (threshold ${(data.threshold*100).toFixed(1)}%)</small></div>
      </div>`;
      const rows = [
        ['XGBoost', p.xgboost],
        ['LightGBM', p.lightgbm],
        ['RandomForest', p.random_forest],
        ['ExtraTrees', p.extra_trees],
        ['Ensemble (4-tree rank-avg)', p.ensemble_4tree],
      ];
      for (const [name, v] of rows) {
        html += `<div class="prob-row"><div class="label">${name} <b>${(v*100).toFixed(2)}%</b></div>${pctFill(v)}</div>`;
      }
      html += `<div class="metric-list" style="margin-top:8px">
        <div class="metric-row"><span class="label">Sensors received</span><span class="value">${data.n_sensors_received}</span></div>
        <div class="metric-row"><span class="label">After cleaning</span><span class="value">${data.n_features_after_cleaning}</span></div>
        <div class="metric-row"><span class="label">MI selected</span><span class="value">${data.n_features_selected}</span></div>
      </div>`;
      html += `<div class="latency-tag">⏱ ${data.latency_ms} ms on CPU</div>`;
      result.innerHTML = html;
    } catch (e) {
      showError(result, 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();
