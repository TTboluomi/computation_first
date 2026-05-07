const weightLabels = {
  cnn: "CNN",
  vit: "ViT",
  syncnet: "SyncNet",
  rppg: "rPPG",
  flow: "光流",
  geometry_model: "几何"
};

function formatTime(ts) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString();
}

function renderWeights(config) {
  const container = document.getElementById("weightSliders");
  container.innerHTML = "";
  Object.entries(weightLabels).forEach(([key, label]) => {
    const value = Number(config.weights[key] || 0);
    const row = document.createElement("div");
    row.className = "threshold-row";
    row.innerHTML = `
      <label>${label} 权重 <input type="range" min="0.01" max="1" step="0.01" data-key="${key}" value="${value}"></label>
      <strong>${value.toFixed(2)}</strong>
    `;
    const slider = row.querySelector("input");
    slider.addEventListener("input", () => {
      row.querySelector("strong").textContent = Number(slider.value).toFixed(2);
    });
    container.appendChild(row);
  });

  document.getElementById("lowThreshold").value = config.risk_thresholds.low;
  document.getElementById("mediumThreshold").value = config.risk_thresholds.medium;
  document.getElementById("lowThresholdValue").textContent = Number(config.risk_thresholds.low).toFixed(2);
  document.getElementById("mediumThresholdValue").textContent = Number(config.risk_thresholds.medium).toFixed(2);
}

function renderModelAverages(modelAverages = {}) {
  const grid = document.getElementById("modelAverageGrid");
  grid.innerHTML = Object.entries(weightLabels).map(([key, label]) => `
    <div class="model-score-card">
      <span>${label} 平均分</span>
      <strong>${Number(modelAverages[key] || 0).toFixed(2)}</strong>
    </div>
  `).join("");
}

function renderSummary(data) {
  document.getElementById("totalLogs").textContent = data.stats.total_logs || 0;
  document.getElementById("avgRisk").textContent = Number(data.stats.avg_risk || 0).toFixed(2);
  document.getElementById("highCount").textContent = data.stats.high_count || 0;
  document.getElementById("sessionCount").textContent = data.sessions.length;

  renderModelAverages(data.model_averages || {});

  const sessionsTable = document.getElementById("sessionsTable");
  sessionsTable.innerHTML = data.sessions.map((item) => `
    <tr>
      <td>${item.session_id.slice(0, 10)}</td>
      <td>${item.label || "-"}</td>
      <td>${Number(item.last_risk_score || 0).toFixed(2)}</td>
      <td><span class="mini-pill ${item.last_risk_level === "HIGH" ? "risk-high" : item.last_risk_level === "MEDIUM" ? "risk-medium" : "risk-low"}">${item.last_risk_level}</span></td>
      <td>${item.snapshot_count}</td>
      <td><button class="ghost-btn table-btn" data-session-id="${item.session_id}">删除</button></td>
    </tr>
  `).join("");

  const logsTable = document.getElementById("logsTable");
  logsTable.innerHTML = data.recent_logs.map((item) => `
    <tr>
      <td>${formatTime(item.created_at)}</td>
      <td>${item.session_id.slice(0, 8)}</td>
      <td>${Number(item.risk_score).toFixed(2)}</td>
      <td>${Number(item.model_scores?.cnn || 0).toFixed(2)}</td>
      <td>${Number(item.model_scores?.vit || 0).toFixed(2)}</td>
      <td>${Number(item.model_scores?.syncnet || 0).toFixed(2)}</td>
      <td>${Number(item.model_scores?.rppg || 0).toFixed(2)}</td>
      <td>${Number(item.model_scores?.flow || 0).toFixed(2)}</td>
      <td>${Number(item.model_scores?.geometry_model || 0).toFixed(2)}</td>
    </tr>
  `).join("");

  document.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", () => deleteSession(button.dataset.sessionId));
  });
}

async function loadSummary() {
  const response = await fetch("/api/admin/summary");
  const data = await response.json();
  renderWeights(data.config);
  renderSummary(data);
}

async function saveConfig() {
  const weights = {};
  document.querySelectorAll("#weightSliders input").forEach((input) => {
    weights[input.dataset.key] = Number(input.value);
  });
  const payload = {
    weights,
    risk_thresholds: {
      low: Number(document.getElementById("lowThreshold").value),
      medium: Number(document.getElementById("mediumThreshold").value)
    }
  };
  const response = await fetch("/api/admin/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  renderWeights(data.config);
}

async function deleteSession(sessionId) {
  const confirmed = window.confirm(`确认删除会话 ${sessionId.slice(0, 8)} 及其检测记录吗？`);
  if (!confirmed) return;
  await fetch(`/api/admin/sessions/${sessionId}`, { method: "DELETE" });
  await loadSummary();
}

document.getElementById("refreshBtn").addEventListener("click", loadSummary);
document.getElementById("saveConfigBtn").addEventListener("click", saveConfig);
document.getElementById("lowThreshold").addEventListener("input", (event) => {
  document.getElementById("lowThresholdValue").textContent = Number(event.target.value).toFixed(2);
});
document.getElementById("mediumThreshold").addEventListener("input", (event) => {
  document.getElementById("mediumThresholdValue").textContent = Number(event.target.value).toFixed(2);
});

loadSummary().catch(console.error);
setInterval(() => loadSummary().catch(console.error), 5000);
