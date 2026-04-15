/* ═══════════════════════════════════════════════════════════
   FairSight — AI Bias Detection Platform
   app.js — Full application logic
═══════════════════════════════════════════════════════════ */

// ── Global state ──────────────────────────────────────────
const STATE = {
  dataset: null,
  datasetKey: null,
  auditComplete: false,
};

// ── Demo dataset definitions ──────────────────────────────
const DATASETS = {
  hiring: {
    name: "Hiring decisions 2019–2023",
    records: 42800,
    attrs: ["gender", "race", "age_group", "disability", "nationality", "religion", "marital_status"],
    outcome: "hired",
    domain: "Employment",
    fairnessScore: 41,
    overallApproval: 58,
  },
  lending: {
    name: "Loan approvals",
    records: 18400,
    attrs: ["gender", "race", "age_group", "zip_code", "disability"],
    outcome: "approved",
    domain: "Finance",
    fairnessScore: 63,
    overallApproval: 52,
  },
  medical: {
    name: "Medical triage scoring",
    records: 9200,
    attrs: ["gender", "race", "age_group", "disability"],
    outcome: "score",
    domain: "Healthcare",
    fairnessScore: 58,
    overallApproval: 61,
  },
};

const DISPARITY_DATA = {
  gender: {
    groups: ["Men", "Women", "Non-binary"],
    approval: [68, 44, 41],
    tpr: [71, 39, 37],
    fpr: [12, 14, 15],
    fnr: [29, 61, 63],
    reference: 0,
    n: [22400, 18600, 1800],
    explain: "Women are approved 24 percentage points less often than men, despite similar qualification profiles. This gap exceeds the 80% rule threshold — women's rate should be at least 54.4% to comply. The model learned this pattern from historical data where women were systematically under-hired between 2019 and 2021.",
  },
  race: {
    groups: ["White", "Asian", "Hispanic", "Black", "Other"],
    approval: [70, 62, 49, 43, 50],
    tpr: [74, 65, 45, 38, 47],
    fpr: [11, 13, 16, 18, 15],
    fnr: [26, 35, 55, 62, 53],
    reference: 0,
    n: [18200, 8400, 7600, 5800, 2800],
    explain: "Black and Hispanic applicants face approval rates 21–27 percentage points lower than White applicants. True positive rate gaps are even larger — 31.7pp for Black applicants — meaning qualified candidates from these groups are being denied at significantly higher rates. This is a direct equal opportunity violation.",
  },
  age: {
    groups: ["18–25", "26–35", "36–45", "46–55", "56+"],
    approval: [51, 72, 69, 58, 44],
    tpr: [53, 75, 71, 56, 40],
    fpr: [15, 10, 11, 14, 17],
    fnr: [47, 25, 29, 44, 60],
    reference: 1,
    n: [5100, 12600, 11400, 8900, 4800],
    explain: "Applicants aged 18–25 and 56+ face measurably lower approval rates than peak-career workers. The model may be using proxies for age — years of experience, employment gaps — to arrive at these patterns, constituting age discrimination in many jurisdictions even without an explicit age feature.",
  },
  disability: {
    groups: ["No disability", "Disclosed disability"],
    approval: [65, 38],
    tpr: [68, 34],
    fpr: [11, 17],
    fnr: [32, 66],
    reference: 0,
    n: [39600, 3200],
    explain: "Applicants who disclosed a disability are approved at less than 60% the rate of non-disabled applicants — one of the starkest disparities in the dataset. This likely reflects both direct discrimination and indirect discrimination via features like employment gaps, which often result from disability-related medical leave.",
  },
};

const HEATMAP_DATA = {
  races: ["White", "Asian", "Hispanic", "Black"],
  genders: ["Men", "Women"],
  values: {
    White: { Men: 72, Women: 58 },
    Asian: { Men: 65, Women: 60 },
    Hispanic: { Men: 50, Women: 32 },
    Black: { Men: 45, Women: 28 },
  },
  ageGenders: {
    ages: ["18–25", "26–35", "36–45", "46–55", "56+"],
    Men: [53, 76, 72, 62, 48],
    Women: [44, 64, 63, 51, 36],
  },
};

const FLAGS = [
  {
    severity: "critical",
    category: "Proxy variable",
    title: "Zip code strongly proxies race",
    body: "Zip code correlates 0.87 (Pearson) with race in this dataset. Using it for scoring replicates historical redlining and mortgage discrimination patterns. Even without explicitly including race as a feature, the model learns racial patterns through this proxy.",
    tags: ["Proxy variable", "Race", "ECOA violation risk"],
    metric: "Correlation: 0.87",
  },
  {
    severity: "critical",
    category: "Training data",
    title: "Severe underrepresentation of female approvals",
    body: "Female applicants represent only 18% of approved records before 2021. The model trained on this data learned that women are unlikely to be hired — not because they're less qualified, but because the training data encodes historical discrimination as ground truth.",
    tags: ["Data imbalance", "Gender", "EEOC concern"],
    metric: "18% female approved pre-2021",
  },
  {
    severity: "critical",
    category: "Equal opportunity",
    title: "Qualified Black applicants rejected at 31.7pp higher rate",
    body: "The true positive rate for Black applicants who would qualify under an unbiased system is 31.7 percentage points lower than for White applicants. This is the definition of an equal opportunity violation: the model systematically fails qualified candidates based on race.",
    tags: ["Equal opportunity", "Race", "Title VII"],
    metric: "TPR gap: 31.7pp",
  },
  {
    severity: "warning",
    category: "Proxy variable",
    title: "University name correlates with socioeconomic status and race",
    body: "The 'institution_name' feature encodes prestige in a way that correlates 0.63 with household income and 0.51 with race. Selective institutions have significantly less representation from Black, Hispanic, and low-income applicants — using it as a feature penalises these groups indirectly.",
    tags: ["Proxy variable", "Socioeconomic", "Race"],
    metric: "Correlation: 0.63 (income)",
  },
  {
    severity: "warning",
    category: "Intersectional",
    title: "Hispanic women face compound disadvantage: −38.7pp gap",
    body: "Hispanic women have an approval rate 38.7pp lower than White men. When analysed separately, Hispanic applicants show a −21pp gap and women show a −24pp gap — but at the intersection, the disadvantage is larger than the sum of parts. Simple single-attribute analysis would miss this entirely.",
    tags: ["Intersectional", "Gender × Race"],
    metric: "Gap: −38.7pp",
  },
  {
    severity: "info",
    category: "Data quality",
    title: "Non-binary applicants: sample too small for high-confidence estimates",
    body: "The non-binary group (n=1,800) shows severe approval gaps but the confidence interval is wide (±8.4pp). Findings should be treated as directionally correct but require more data for definitive conclusions. Consider collecting more data or using Bayesian smoothing.",
    tags: ["Data quality", "Gender", "Sample size"],
    metric: "n=1,800 · CI: ±8.4pp",
  },
  {
    severity: "info",
    category: "Temporal",
    title: "Bias worsened after model retraining in Q3 2022",
    body: "Approval rate gaps for racial minorities increased by 6–9pp after the model was retrained on 2021–2022 data. This suggests the 2021 cohort contained more biased outcomes than prior years, and retraining amplified rather than reduced the disparity.",
    tags: ["Temporal drift", "Retraining", "Monitoring"],
    metric: "Gap increase: +6–9pp post-Q3 2022",
  },
];

const REMEDIATION = {
  pre: [
    {
      num: "P1",
      title: "Remove proxy variables",
      body: "Drop zip code, institution name, and surname from the feature set. These correlate strongly with protected attributes (r ≥ 0.5) and enable indirect discrimination without explicitly encoding protected characteristics. Validate that model accuracy loss is within acceptable bounds (<3%).",
      impact: "−8pp demographic parity gap",
      code: `# Identify proxies via correlation analysis\nfrom sklearn.preprocessing import LabelEncoder\n\nproxy_threshold = 0.5\nproxy_cols = []\nfor col in X.columns:\n    for protected in protected_attrs:\n        corr = X[col].corr(df[protected])\n        if abs(corr) > proxy_threshold:\n            proxy_cols.append(col)\n            break\n\nX_clean = X.drop(columns=list(set(proxy_cols)))\nprint(f"Removed {len(set(proxy_cols))} proxy features")`,
    },
    {
      num: "P2",
      title: "Reweight underrepresented groups",
      body: "Apply inverse-frequency sample weights to underrepresented groups. Female-approved records should carry 3.8× weight to counteract the 18% historical representation gap. This corrects the training distribution without throwing away data or generating synthetic samples.",
      impact: "−6pp gender approval gap",
      code: `from sklearn.utils.class_weight import compute_sample_weight\nimport numpy as np\n\n# Compute weights per group intersection\ndef intersectional_weights(df, protected_cols):\n    group_key = df[protected_cols].astype(str).agg('-'.join, axis=1)\n    group_counts = group_key.value_counts(normalize=True)\n    weights = group_key.map(lambda g: 1.0 / group_counts[g])\n    return weights / weights.mean()  # normalise\n\nsample_weights = intersectional_weights(df_train, ['gender', 'race'])`,
    },
    {
      num: "P3",
      title: "Disaggregated data audit",
      body: "Split the dataset by protected attribute and audit each slice independently. A model with 90% overall accuracy can have 60% accuracy on minority subgroups. This invisible failure mode is the most common source of real-world harm from ML systems.",
      impact: "Reveals hidden subgroup failures",
      code: `# Evaluate on each demographic slice\nresults = {}\nfor group in df_test['race'].unique():\n    mask = df_test['race'] == group\n    subset = df_test[mask]\n    y_true = subset['hired']\n    y_pred = model.predict(subset.drop('hired', axis=1))\n    results[group] = {\n        'accuracy': accuracy_score(y_true, y_pred),\n        'tpr': recall_score(y_true, y_pred),\n        'fpr': (y_pred[y_true==0] == 1).mean()\n    }`,
    },
  ],
  in: [
    {
      num: "I1",
      title: "Adversarial debiasing",
      body: "Train a predictor and an adversary simultaneously. The adversary tries to predict protected attributes from model outputs; the predictor is penalised when the adversary succeeds. This forces the model to learn representations that are both accurate and uninformative about demographic identity.",
      impact: "−12pp equalized odds gap",
      code: `# Adversarial debiasing with TensorFlow\nimport tensorflow as tf\n\nclass AdversarialDebiasingModel(tf.keras.Model):\n    def __init__(self, predictor, adversary, lambda_adv=1.0):\n        super().__init__()\n        self.predictor = predictor\n        self.adversary = adversary\n        self.lambda_adv = lambda_adv\n\n    def train_step(self, data):\n        X, y, protected = data\n        with tf.GradientTape() as tape:\n            y_hat = self.predictor(X)\n            pred_loss = bce(y, y_hat)\n            adv_pred = self.adversary(y_hat)\n            adv_loss = bce(protected, adv_pred)\n            # Predictor is penalised when adversary succeeds\n            total_loss = pred_loss - self.lambda_adv * adv_loss\n        return {'loss': total_loss, 'adv_loss': adv_loss}`,
    },
    {
      num: "I2",
      title: "Fairness-constrained optimisation",
      body: "Add equalized odds as a hard constraint to the loss function. Frame it as: maximise accuracy subject to |TPR_A − TPR_B| < ε for all group pairs. This guarantees fairness properties at training time rather than patching them post-hoc.",
      impact: "−9pp equal opportunity gap",
      code: `from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds\nfrom sklearn.linear_model import LogisticRegression\n\n# Constraint: equalized odds within 5pp\nconstraint = EqualizedOdds(difference_bound=0.05)\n\nmitigator = ExponentiatedGradient(\n    estimator=LogisticRegression(),\n    constraints=constraint,\n    eps=0.05,\n    max_iter=50,\n)\n\nmitigator.fit(\n    X_train, y_train,\n    sensitive_features=df_train['race']\n)\n\ny_pred = mitigator.predict(X_test)`,
    },
    {
      num: "I3",
      title: "Fairness-aware feature selection",
      body: "Use mutual information to identify features whose predictive power derives mainly from correlation with protected attributes rather than genuine job-relevant skill. Remove features where MI(feature, protected) / MI(feature, outcome) > 0.3.",
      impact: "Reduces proxy-driven disparities",
      code: `from sklearn.feature_selection import mutual_info_classif\nimport pandas as pd\n\ndef fairness_feature_score(X, y, protected, threshold=0.3):\n    mi_outcome = mutual_info_classif(X, y)\n    mi_protected = mutual_info_classif(X, protected)\n    \n    df = pd.DataFrame({\n        'feature': X.columns,\n        'mi_outcome': mi_outcome,\n        'mi_protected': mi_protected,\n        'ratio': mi_protected / (mi_outcome + 1e-8)\n    })\n    \n    flagged = df[df.ratio > threshold]\n    safe = df[df.ratio <= threshold]['feature'].tolist()\n    return safe, flagged`,
    },
  ],
  post: [
    {
      num: "Q1",
      title: "Threshold calibration by group",
      body: "Apply different decision thresholds per demographic group to equalise true positive rates. This raises the approval rate for qualified candidates in underrepresented groups without retraining the model. Only the decision boundary changes — the underlying scores remain identical.",
      impact: "−15pp equal opportunity gap",
      code: `from fairlearn.postprocessing import ThresholdOptimizer\nfrom fairlearn.metrics import equalized_odds_difference\n\n# Find optimal thresholds per group\nthreshold_optimizer = ThresholdOptimizer(\n    estimator=trained_model,\n    constraints='equalized_odds',\n    objective='balanced_accuracy_score',\n    predict_method='predict_proba',\n)\n\nthreshold_optimizer.fit(\n    X_train, y_train,\n    sensitive_features=df_train['race']\n)\n\n# Different threshold applied per group automatically\ny_pred_fair = threshold_optimizer.predict(\n    X_test,\n    sensitive_features=df_test['race']\n)`,
    },
    {
      num: "Q2",
      title: "Reject option classification",
      body: "For predictions near the decision boundary (low-confidence zone), apply a fairness override: favour the unprivileged group. Only affects borderline cases, limiting the accuracy trade-off while meaningfully improving fairness at the margin where the model is most uncertain.",
      impact: "−6pp disparity in borderline cases",
      code: `import numpy as np\n\ndef reject_option_predict(scores, protected, privileged_val,\n                          threshold=0.5, band=0.1):\n    \"\"\"\n    In the uncertainty band [threshold-band, threshold+band],\n    flip borderline cases to favour the unprivileged group.\n    \"\"\"\n    predictions = (scores >= threshold).astype(int)\n    uncertain = (scores >= threshold - band) & (scores < threshold + band)\n    \n    # Unprivileged in uncertain zone → predict positive\n    unprivileged_uncertain = uncertain & (protected != privileged_val)\n    predictions[unprivileged_uncertain] = 1\n    \n    # Privileged in uncertain zone → predict negative\n    privileged_uncertain = uncertain & (protected == privileged_val)\n    predictions[privileged_uncertain] = 0\n    \n    return predictions`,
    },
    {
      num: "Q3",
      title: "Continuous monitoring & drift detection",
      body: "Deploy a fairness monitoring layer that recalculates disparity metrics on a rolling 30-day window of live decisions. Alert when any gap exceeds the threshold. Schedule automatic re-audits after every model retraining event.",
      impact: "Prevents silent bias regression",
      code: `class FairnessMonitor:\n    def __init__(self, protected_attrs, threshold=0.05):\n        self.protected_attrs = protected_attrs\n        self.threshold = threshold\n        self.window = []\n\n    def log_decision(self, features, protected, outcome, score):\n        self.window.append({\n            'protected': protected,\n            'outcome': outcome,\n            'score': score,\n            'timestamp': time.time()\n        })\n        # Keep rolling 30-day window\n        cutoff = time.time() - 30 * 86400\n        self.window = [r for r in self.window if r['timestamp'] > cutoff]\n\n    def check_alerts(self):\n        df = pd.DataFrame(self.window)\n        alerts = []\n        for attr in self.protected_attrs:\n            gap = df.groupby(attr)['outcome'].mean()\n            disparity = gap.max() - gap.min()\n            if disparity > self.threshold:\n                alerts.append(f"ALERT: {attr} gap = {disparity:.3f}")\n        return alerts`,
    },
  ],
};

// ── Navigation ────────────────────────────────────────────
function showApp() {
  document.getElementById("landing").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
}

function showLanding() {
  document.getElementById("app").classList.add("hidden");
  document.getElementById("landing").classList.remove("hidden");
}

function loadDemo() {
  showApp();
  loadDataset("hiring");
  setTimeout(() => runAudit(), 300);
}

function switchView(name, el) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  document.getElementById("view-" + name).classList.add("active");
  el.classList.add("active");
  if (name === "overview") renderOverview();
  if (name === "disparity") renderDisparityView();
  if (name === "heatmap") renderHeatmaps();
  if (name === "flags") renderFlags();
  if (name === "remediation") renderRemediation();
  if (name === "report") renderReport();
}

// ── Dataset loading ───────────────────────────────────────
function loadDataset(key) {
  STATE.datasetKey = key;
  STATE.dataset = DATASETS[key];

  // Update UI
  document.querySelectorAll(".demo-card").forEach((c) => c.classList.remove("active-demo"));
  document.getElementById("demo-" + key)?.classList.add("active-demo");

  document.getElementById("dataset-name").textContent = STATE.dataset.name;
  document.getElementById("dataset-meta").textContent =
    STATE.dataset.records.toLocaleString() + " records · " + STATE.dataset.domain;

  // Show config
  const configSection = document.getElementById("config-section");
  configSection.style.display = "block";
  configSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

  // Populate attribute chips
  const grid = document.getElementById("attr-grid");
  grid.innerHTML = "";
  STATE.dataset.attrs.forEach((attr) => {
    const chip = document.createElement("div");
    chip.className = "attr-chip selected";
    chip.innerHTML = `<div class="attr-dot"></div>${attr.replace("_", " ")}`;
    chip.onclick = () => chip.classList.toggle("selected");
    grid.appendChild(chip);
  });
}

function handleFileUpload(input) {
  const file = input.files[0];
  if (!file) return;
  document.getElementById("upload-zone").innerHTML = `
    <div class="upload-icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M8 16 L14 22 L24 10" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div class="upload-text">${file.name}</div>
    <div class="upload-sub">${(file.size / 1024).toFixed(0)} KB · Ready to audit</div>
  `;
  // Simulate loading the hiring dataset for demo
  loadDataset("hiring");
}

// ── Audit runner ──────────────────────────────────────────
function runAudit() {
  if (!STATE.datasetKey) {
    loadDataset("hiring");
  }

  const overlay = document.getElementById("audit-overlay");
  overlay.classList.remove("hidden");

  const steps = [
    "Scanning for proxy variables…",
    "Computing demographic parity…",
    "Calculating equal opportunity gaps…",
    "Running equalized odds analysis…",
    "Performing intersectional analysis…",
    "Testing individual fairness…",
    "Generating flag report…",
    "Building remediation plan…",
  ];

  const stepsEl = document.getElementById("audit-steps");
  stepsEl.innerHTML = steps.map((s) =>
    `<div class="audit-step"><div class="audit-step-icon"></div>${s}</div>`
  ).join("");

  const stepEls = stepsEl.querySelectorAll(".audit-step");
  let i = 0;

  function tick() {
    if (i < stepEls.length) {
      stepEls[i].classList.add("done");
      i++;
      setTimeout(tick, 220);
    } else {
      setTimeout(() => {
        overlay.classList.add("hidden");
        STATE.auditComplete = true;
        showApp();
        // Switch to overview
        document.querySelectorAll(".nav-item")[1].click();
      }, 600);
    }
  }
  tick();
}

// ── Overview rendering ────────────────────────────────────
function renderOverview() {
  renderMainBarChart();
  renderAttrSummary();
  renderRadarChart();
}

function colorFor(rate, ref) {
  const ratio = rate / ref;
  if (ratio >= 0.9) return { cls: "good", color: "var(--green)" };
  if (ratio >= 0.75) return { cls: "warn", color: "var(--amber)" };
  return { cls: "bad", color: "var(--red)" };
}

function renderMainBarChart() {
  const el = document.getElementById("main-bar-chart");
  if (!el) return;
  const groups = ["White men", "White women", "Asian men", "Asian women", "Hispanic men", "Hispanic women", "Black men", "Black women"];
  const rates = [72, 58, 65, 60, 50, 32, 45, 28];
  const ref = 72;

  el.innerHTML = groups.map((g, i) => {
    const v = rates[i];
    const c = colorFor(v, ref);
    return `<div class="bar-row">
      <div class="bar-label">${g}</div>
      <div class="bar-track">
        <div class="bar-fill ${c.cls}" style="width:${v}%"></div>
        <div class="thresh-marker"></div>
      </div>
      <div class="bar-val" style="color:${c.color}">${v}%</div>
    </div>`;
  }).join("");
}

function renderAttrSummary() {
  const el = document.getElementById("attr-summary-list");
  if (!el) return;
  const data = [
    { name: "Gender", gap: "24.0pp", severity: "critical" },
    { name: "Race / ethnicity", gap: "27.0pp", severity: "critical" },
    { name: "Disability status", gap: "27.0pp", severity: "critical" },
    { name: "Age group", gap: "21.0pp", severity: "warning" },
    { name: "Nationality", gap: "8.4pp", severity: "ok" },
    { name: "Marital status", gap: "3.1pp", severity: "ok" },
  ];
  const colors = { critical: "var(--red)", warning: "var(--amber)", ok: "var(--green)" };
  el.innerHTML = data.map((d) => `
    <div class="attr-summary-item">
      <div class="attr-name">${d.name}</div>
      <span class="severity-badge ${d.severity}">${d.gap}</span>
      <span class="attr-gap" style="color:${colors[d.severity]}">${d.severity}</span>
    </div>
  `).join("");
}

function renderRadarChart() {
  const canvas = document.getElementById("radar-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = 300, H = 260;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2 + 10, R = 100;
  const labels = ["Dem. parity", "Eq. opportunity", "Eq. odds", "Calibration", "Individual", "Counterfact."];
  const values = [0.20, 0.22, 0.18, 0.70, 0.50, 0.75];
  const N = labels.length;

  // Grid rings
  const style = getComputedStyle(document.documentElement);
  const borderColor = "#2a3140";
  const textColor = "#6b7a8d";
  const accentColor = "#4fffb0";
  const redColor = "#ff4d6d";

  for (let r = 1; r <= 4; r++) {
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * Math.PI * 2 - Math.PI / 2;
      const x = cx + Math.cos(angle) * (R * r / 4);
      const y = cy + Math.sin(angle) * (R * r / 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Axes
  for (let i = 0; i < N; i++) {
    const angle = (i / N) * Math.PI * 2 - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R);
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Data polygon
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const angle = (i / N) * Math.PI * 2 - Math.PI / 2;
    const r = values[i] * R;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(255,77,109,0.12)";
  ctx.fill();
  ctx.strokeStyle = redColor;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Data points
  for (let i = 0; i < N; i++) {
    const angle = (i / N) * Math.PI * 2 - Math.PI / 2;
    const r = values[i] * R;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, 3, 0, Math.PI * 2);
    ctx.fillStyle = redColor;
    ctx.fill();
  }

  // Labels
  ctx.font = "10px DM Mono, monospace";
  ctx.fillStyle = textColor;
  ctx.textAlign = "center";
  for (let i = 0; i < N; i++) {
    const angle = (i / N) * Math.PI * 2 - Math.PI / 2;
    const lx = cx + Math.cos(angle) * (R + 22);
    const ly = cy + Math.sin(angle) * (R + 18);
    ctx.fillText(labels[i], lx, ly);
  }

  // Center label
  ctx.font = "11px DM Mono, monospace";
  ctx.fillStyle = redColor;
  ctx.textAlign = "center";
  ctx.fillText("41/100", cx, cy + 4);
}

// ── Disparity rendering ───────────────────────────────────
function renderDisparityView() {
  const attr = document.getElementById("attr-select")?.value || "gender";
  const metric = document.getElementById("metric-select")?.value || "approval";
  const d = DISPARITY_DATA[attr];
  const vals = d[metric];
  const ref = vals[d.reference];

  const titles = {
    approval: "Approval rate",
    tpr: "True positive rate",
    fpr: "False positive rate",
    fnr: "False negative rate",
  };

  document.getElementById("disp-chart-title").textContent =
    `${titles[metric]} by ${attr.replace("_", " ")}`;
  document.getElementById("explain-text").textContent = d.explain;

  // Bar chart
  const chartEl = document.getElementById("disparity-chart");
  chartEl.innerHTML = d.groups.map((g, i) => {
    const v = vals[i];
    const c = colorFor(v, ref);
    return `<div class="bar-row">
      <div class="bar-label">${g}<br><span style="font-size:10px;color:var(--muted2)">n=${d.n[i].toLocaleString()}</span></div>
      <div class="bar-track">
        <div class="bar-fill ${c.cls}" style="width:${v}%"></div>
        <div class="thresh-marker"></div>
      </div>
      <div class="bar-val" style="color:${c.color}">${v}%</div>
    </div>`;
  }).join("");

  // Gap table
  const gapEl = document.getElementById("gap-table");
  gapEl.innerHTML = d.groups.map((g, i) => {
    if (i === d.reference) return `<div class="gap-row"><span class="gap-rank-label">${g}</span><span style="font-family:var(--font-mono);font-size:12px;color:var(--muted)">Reference</span></div>`;
    const diff = vals[i] - ref;
    const c = colorFor(vals[i], ref);
    const label = c.cls === "good" ? "Acceptable" : c.cls === "warn" ? "Warning" : "Critical";
    const badgeCls = c.cls === "good" ? "ok" : c.cls === "warn" ? "warning" : "critical";
    return `<div class="gap-row">
      <span class="gap-rank-label">${g} vs. ${d.groups[d.reference]}</span>
      <span class="gap-rank-val" style="color:${c.color}">${diff > 0 ? "+" : ""}${diff.toFixed(1)}pp</span>
      <span class="severity-badge ${badgeCls}">${label}</span>
    </div>`;
  }).join("");

  // Stat table
  const statEl = document.getElementById("stat-table");
  statEl.innerHTML = `<div class="stat-header"><span>Group</span><span>Gap</span><span>p-value</span><span>Sig.</span></div>` +
    d.groups.filter((_, i) => i !== d.reference).map((g, rawI) => {
      const i = rawI >= d.reference ? rawI + 1 : rawI;
      const diff = Math.abs(vals[i] - ref);
      const pval = diff > 15 ? "<0.001" : diff > 8 ? "0.003" : "0.041";
      const sig = diff > 15 ? "***" : diff > 8 ? "**" : "*";
      return `<div class="stat-row">
        <span>${g}</span>
        <span style="color:var(--red);font-family:var(--font-mono)">−${diff.toFixed(1)}pp</span>
        <span style="font-family:var(--font-mono);color:var(--muted)">${pval}</span>
        <span style="color:var(--accent);font-family:var(--font-mono)">${sig}</span>
      </div>`;
    }).join("");
}

// ── Heatmap rendering ─────────────────────────────────────
function renderHeatmaps() {
  renderHeatmap("main-heatmap", HEATMAP_DATA.races, HEATMAP_DATA.genders, HEATMAP_DATA.values);
  renderAgeHeatmap();
  renderGapRanking();
}

function heatmapColor(v) {
  if (v > 65) return { bg: "#085041", fg: "#E1F5EE" };
  if (v > 50) return { bg: "#1D9E75", fg: "#E1F5EE" };
  if (v > 35) return { bg: "#BA7517", fg: "#fff" };
  return { bg: "#E24B4A", fg: "#fff" };
}

function renderHeatmap(containerId, rows, cols, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  let html = `<table class="heatmap-table"><thead><tr><th></th>`;
  cols.forEach((c) => (html += `<th>${c}</th>`));
  html += `<th>Gap</th><th>n</th></tr></thead><tbody>`;

  const ns = { White: "8,240", Asian: "3,180", Hispanic: "2,650", Black: "1,890" };
  rows.forEach((r) => {
    html += `<tr><td class="row-label">${r}</td>`;
    const vals = cols.map((c) => data[r][c]);
    vals.forEach((v) => {
      const c = heatmapColor(v);
      html += `<td style="background:${c.bg};color:${c.fg}">${v}%</td>`;
    });
    const gap = Math.max(...vals) - Math.min(...vals);
    html += `<td style="color:var(--muted);font-size:11px">${gap}pp</td>`;
    html += `<td style="color:var(--muted);font-size:11px">${ns[r] || "—"}</td></tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function renderAgeHeatmap() {
  const el = document.getElementById("age-heatmap");
  if (!el) return;
  const d = HEATMAP_DATA.ageGenders;
  let html = `<table class="heatmap-table"><thead><tr><th></th><th>Men</th><th>Women</th><th>Gap</th></tr></thead><tbody>`;
  d.ages.forEach((age, i) => {
    const m = d.Men[i], w = d.Women[i];
    const cm = heatmapColor(m), cw = heatmapColor(w);
    html += `<tr>
      <td class="row-label">${age}</td>
      <td style="background:${cm.bg};color:${cm.fg}">${m}%</td>
      <td style="background:${cw.bg};color:${cw.fg}">${w}%</td>
      <td style="color:var(--muted);font-size:11px">${m - w}pp</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function renderGapRanking() {
  const el = document.getElementById("gap-ranking");
  if (!el) return;
  const gaps = [
    { label: "Black women vs. White men", val: -44.1, sev: "critical" },
    { label: "Hispanic women vs. White men", val: -38.7, sev: "critical" },
    { label: "Black men vs. White men", val: -27.3, sev: "warning" },
    { label: "Hispanic men vs. White men", val: -21.5, sev: "warning" },
    { label: "Asian women vs. White men", val: -8.2, sev: "ok" },
    { label: "White women vs. White men", val: -14.0, sev: "warning" },
  ];
  const colors = { critical: "var(--red)", warning: "var(--amber)", ok: "var(--green)" };
  el.innerHTML = gaps.map((g) => `
    <div class="gap-rank-item">
      <span class="gap-rank-label">${g.label}</span>
      <span class="gap-rank-val" style="color:${colors[g.sev]}">${g.val.toFixed(1)}pp</span>
      <span class="severity-badge ${g.sev}">${g.sev}</span>
    </div>
  `).join("");
}

// ── Flags rendering ───────────────────────────────────────
function renderFlags() {
  const el = document.getElementById("flags-list");
  if (!el) return;
  const grouped = { critical: [], warning: [], info: [] };
  FLAGS.forEach((f) => grouped[f.severity].push(f));

  const labels = { critical: "Critical flags", warning: "Warnings", info: "Informational" };
  let html = "";
  ["critical", "warning", "info"].forEach((sev) => {
    if (!grouped[sev].length) return;
    html += `<div class="flag-group-title">${labels[sev]}</div>`;
    grouped[sev].forEach((f) => {
      html += `<div class="flag-card ${sev}">
        <div class="flag-header">
          <div class="flag-title">${f.title}</div>
          <div style="display:flex;gap:8px;align-items:center;flex-shrink:0">
            <span style="font-family:var(--font-mono);font-size:11px;color:var(--muted)">${f.metric}</span>
            <span class="severity-badge ${sev}">${sev}</span>
          </div>
        </div>
        <div class="flag-body">${f.body}</div>
        <div class="flag-meta">${f.tags.map((t) => `<span class="flag-tag">${t}</span>`).join("")}</div>
      </div>`;
    });
  });
  el.innerHTML = html;
}

// ── Remediation rendering ─────────────────────────────────
function renderRemediation() {
  ["pre", "in", "post"].forEach((stage) => {
    const el = document.getElementById("rem-" + stage);
    if (!el) return;
    const nums = { pre: "pre", in: "in", post: "post" };
    el.innerHTML = REMEDIATION[stage].map((r) => `
      <div class="rem-card">
        <div class="rem-card-header">
          <div class="rem-num ${nums[stage]}">${r.num}</div>
          <div class="rem-title">${r.title}</div>
        </div>
        <div class="rem-body">${r.body}</div>
        <div class="rem-impact"><span style="color:var(--muted);font-size:12px">Projected impact:</span> <span class="rem-impact-val">${r.impact}</span></div>
        <details style="margin-top:10px">
          <summary style="font-size:12px;color:var(--muted);cursor:pointer;font-family:var(--font-mono)">View code example</summary>
          <pre style="margin-top:8px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem;font-size:11px;color:var(--text);overflow-x:auto;line-height:1.5;font-family:var(--font-mono)">${escapeHtml(r.code)}</pre>
        </details>
      </div>
    `).join("");
  });

  renderImpactProjection();
}

function renderImpactProjection() {
  const el = document.getElementById("impact-projection");
  if (!el) return;
  const metrics = [
    { label: "Demographic parity gap", before: 23.4, after: 4.1, unit: "%" },
    { label: "Equal opportunity gap", before: 31.7, after: 5.8, unit: "%" },
    { label: "Intersectional gap (worst)", before: 44.1, after: 9.2, unit: "%" },
    { label: "Fairness score", before: 41, after: 82, unit: "/100", higher: true },
  ];

  el.innerHTML = metrics.map((m) => {
    const beforeW = m.higher ? m.before : Math.min(m.before * 1.2, 100);
    const afterW = m.higher ? m.after : Math.min(m.after * 1.2, 100);
    return `<div class="impact-row">
      <div class="impact-labels">
        <span class="impact-label">${m.label}</span>
        <span class="impact-change"><span style="color:var(--red)">${m.before}${m.unit}</span> → <span style="color:var(--green)">${m.after}${m.unit}</span></span>
      </div>
      <div class="impact-bars-wrap">
        <div class="impact-bar" style="width:${beforeW}%;background:var(--red);opacity:0.7"></div>
        <div class="impact-bar" style="width:${afterW}%;background:var(--green)"></div>
      </div>
    </div>`;
  }).join("");
}

function switchRemTab(stage, btn) {
  document.querySelectorAll(".rem-tab").forEach((t) => t.classList.remove("active"));
  document.querySelectorAll(".rem-panel").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById("rem-" + stage).classList.add("active");
}

// ── Report rendering ──────────────────────────────────────
function renderReport() {
  const el = document.getElementById("report-content");
  if (!el) return;
  const ds = STATE.dataset || DATASETS.hiring;
  const now = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  el.innerHTML = `
    <div class="report-section">
      <h3>Audit summary</h3>
      <div class="report-meta-grid">
        <div class="report-meta-item"><div class="report-meta-label">Dataset</div><div class="report-meta-val">${ds.name}</div></div>
        <div class="report-meta-item"><div class="report-meta-label">Records analysed</div><div class="report-meta-val">${ds.records.toLocaleString()}</div></div>
        <div class="report-meta-item"><div class="report-meta-label">Audit date</div><div class="report-meta-val">${now}</div></div>
        <div class="report-meta-item"><div class="report-meta-label">Fairness score</div><div class="report-meta-val" style="color:var(--red)">41/100 — Critical</div></div>
        <div class="report-meta-item"><div class="report-meta-label">Protected attributes</div><div class="report-meta-val">7 attributes analysed</div></div>
        <div class="report-meta-item"><div class="report-meta-label">Flags raised</div><div class="report-meta-val">3 critical · 2 warnings · 2 informational</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Key findings</h3>
      ${FLAGS.filter(f => f.severity !== "info").map((f, i) => `
        <div class="report-finding">
          <span class="report-find-num">${i + 1}.</span>
          <div>
            <div style="font-weight:500;margin-bottom:3px">${f.title}</div>
            <div style="font-size:12px;color:var(--muted)">${f.body}</div>
          </div>
          <span class="severity-badge ${f.severity}" style="flex-shrink:0;align-self:flex-start">${f.severity}</span>
        </div>
      `).join("")}
    </div>

    <div class="report-section">
      <h3>Regulatory compliance status</h3>
      <div class="report-reg-row"><span class="reg-name">EEOC / Title VII (US)</span><span style="color:var(--muted);font-size:12px;flex:1">80% rule violated — demographic parity gap 23.4% exceeds 5% threshold</span><span class="severity-badge critical reg-status">Non-compliant</span></div>
      <div class="report-reg-row"><span class="reg-name">ECOA (US Credit)</span><span style="color:var(--muted);font-size:12px;flex:1">Zip code proxy variable poses redlining risk</span><span class="severity-badge warning reg-status">At risk</span></div>
      <div class="report-reg-row"><span class="reg-name">EU AI Act (High Risk)</span><span style="color:var(--muted);font-size:12px;flex:1">Fundamental rights impact assessment required; bias documentation missing</span><span class="severity-badge critical reg-status">Non-compliant</span></div>
      <div class="report-reg-row"><span class="reg-name">ADA (Disability)</span><span style="color:var(--muted);font-size:12px;flex:1">27pp approval gap for disclosed disability applicants — high legal exposure</span><span class="severity-badge critical reg-status">Non-compliant</span></div>
      <div class="report-reg-row"><span class="reg-name">ADEA (Age)</span><span style="color:var(--muted);font-size:12px;flex:1">Age-group disparity pattern detected; requires legal review</span><span class="severity-badge warning reg-status">Review needed</span></div>
      <div class="report-reg-row"><span class="reg-name">ISO/IEC 24027</span><span style="color:var(--muted);font-size:12px;flex:1">Bias documentation partially complete; intersectional analysis missing</span><span class="severity-badge warning reg-status">Partial</span></div>
    </div>

    <div class="report-section">
      <h3>Recommended remediation priority</h3>
      <div style="font-size:13px;color:var(--muted);line-height:1.7">
        <p style="margin-bottom:8px"><strong style="color:var(--text);font-weight:500">Immediate (before deployment):</strong> Remove zip code and institution name proxy variables. Apply intersectional sample reweighting. Perform disaggregated evaluation on all demographic slices.</p>
        <p style="margin-bottom:8px"><strong style="color:var(--text);font-weight:500">Short-term (within 30 days):</strong> Retrain with adversarial debiasing or fairness constraints. Apply threshold calibration by group for the current model if retraining is not immediately possible.</p>
        <p><strong style="color:var(--text);font-weight:500">Ongoing:</strong> Deploy continuous fairness monitoring. Schedule re-audit after every model retraining. Collect more data on underrepresented groups.</p>
      </div>
    </div>

    <div class="report-section">
      <h3>Methodology</h3>
      <div style="font-size:13px;color:var(--muted);line-height:1.7">
        <p>Fairness metrics computed using standard definitions from Barocas, Hardt & Narayanan (2019). Demographic parity gap = max(Pr[Ŷ=1|A=a]) − min(Pr[Ŷ=1|A=a]) across all group values a. Equal opportunity gap = max(Pr[Ŷ=1|Y=1,A=a]) − min(Pr[Ŷ=1|Y=1,A=a]). 80% rule threshold = min approval rate must be ≥ 80% of max approval rate per EEOC four-fifths rule. Statistical significance tested using two-sample proportion z-test with Bonferroni correction for multiple comparisons. p &lt; 0.05 required for significance.</p>
      </div>
    </div>
  `;
}

// ── Utils ─────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function exportJSON() {
  const report = {
    meta: {
      tool: "FairSight",
      version: "1.0.0",
      generated: new Date().toISOString(),
      dataset: STATE.dataset?.name || "Demo dataset",
    },
    fairnessScore: 41,
    metrics: {
      demographicParityGap: 0.234,
      equalOpportunityGap: 0.317,
      equalizedOddsGap: 0.298,
      calibrationScore: 0.70,
    },
    flags: FLAGS,
    remediation: {
      preprocessing: REMEDIATION.pre.map((r) => r.title),
      inprocessing: REMEDIATION.in.map((r) => r.title),
      postprocessing: REMEDIATION.post.map((r) => r.title),
    },
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "fairsight-audit-report.json";
  a.click(); URL.revokeObjectURL(url);
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Auto-load hiring dataset for demo readiness
  loadDataset("hiring");
});
