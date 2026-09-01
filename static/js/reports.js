(() => {
  const report = document.querySelector(".report-kpis, .report-table");
  if (!report) return;
  document.querySelectorAll("[data-report-filter]").forEach((control) => {
    control.addEventListener("change", () => {
      const query = control.value.toLowerCase();
      document.querySelectorAll("[data-report-row]").forEach((row) => {
        row.hidden = query && !row.textContent.toLowerCase().includes(query);
      });
    });
  });

  const live = document.querySelector("[data-report-live]");
  const monitor = document.querySelector("[data-live-monitor]");
  const statusUrl = live?.dataset.statusUrl || monitor?.dataset.statusUrl;
  if (!statusUrl) return;

  const valueNodes = (key) => document.querySelectorAll(`[data-live-value="${key}"]`);
  const setValue = (key, value) => valueNodes(key).forEach((node) => {
    node.textContent = value;
  });
  const renderRuns = (runs) => {
    const list = document.querySelector("[data-live-run-list]");
    const count = document.querySelector("[data-live-run-count]");
    if (!list || !count) return;
    count.textContent = runs.length ? `${runs.length} active` : "No active jobs";
    list.replaceChildren();
    if (!runs.length) {
      const empty = document.createElement("div");
      empty.className = "live-empty";
      empty.innerHTML = "<span>✓</span><div><strong>No training jobs are running</strong><small>New uploads, experiments, and predictions will appear here automatically.</small></div>";
      list.append(empty);
      return;
    }
    runs.forEach((run) => {
      const row = document.createElement("div");
      row.className = "live-run-row";
      const pulse = document.createElement("span");
      pulse.className = "live-run-pulse";
      const detail = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = run.label;
      const dataset = document.createElement("small");
      dataset.textContent = run.dataset;
      detail.append(title, dataset);
      const stage = document.createElement("span");
      stage.className = "live-run-stage";
      stage.textContent = run.progress?.label || "Training in progress";
      const percent = document.createElement("b");
      percent.textContent = `${Math.max(0, Math.min(100, Number(run.progress?.percent) || 0))}%`;
      row.append(pulse, detail, stage, percent);
      list.append(row);
    });
  };
  const refresh = async () => {
    try {
      const response = await fetch(statusUrl, { headers: { "X-Requested-With": "XMLHttpRequest" }, cache: "no-store" });
      if (!response.ok) throw new Error("Live status unavailable");
      const state = await response.json();
      setValue("datasets", state.datasets);
      setValue("runs", state.runs);
      setValue("completed", state.completed);
      setValue("predictions", state.predictions);
      setValue("best_accuracy", state.best_accuracy === null ? "—" : `${state.best_accuracy}%`);
      setValue("best_model", state.best_model || "No completed model");
      renderRuns(state.active_runs || []);
      const connection = document.querySelector("[data-live-connection]");
      if (connection) connection.textContent = state.active_runs?.length ? `${state.active_runs.length} live job${state.active_runs.length === 1 ? "" : "s"}` : "Live workspace view";
      const updated = document.querySelector("[data-live-updated]");
      if (updated) updated.textContent = `Updated ${new Date(state.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    } catch (_error) {
      const connection = document.querySelector("[data-live-connection]");
      if (connection) connection.textContent = "Live updates paused";
    }
  };
  refresh();
  window.setInterval(refresh, 8000);
})();
