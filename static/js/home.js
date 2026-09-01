(() => {
  const page = document.querySelector(".overview-page");
  if (!page) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const scanSamples = [
    { image: "/static/images/radar/scan-cancer.jpg", title: "Skin cancer / tumor", type: "PRIMARY FOCUS", focus: "52% 47%" },
    { image: "/static/images/radar/scan-plant.jpg", title: "Crop / plant disease", type: "OPTIONAL SUPPORT", focus: "50% 52%" },
    { image: "/static/images/radar/scan-animal-new.png", title: "Animal classification", type: "OPTIONAL SUPPORT", focus: "50% 43%" },
    { image: "/static/images/radar/scan-food.jpg", title: "Food classification", type: "OPTIONAL SUPPORT", focus: "52% 48%" },
    { image: "/static/images/radar/scan-xray.jpg", title: "X-ray imaging", type: "OPTIONAL SUPPORT", focus: "50% 47%" },
    { image: "/static/images/radar/scan-others-new.png", title: "Other labeled images", type: "OPTIONAL SUPPORT", focus: "50% 46%" }
  ];
  const scanVisual = page.querySelector("#scanVisual");
  const scanImage = page.querySelector("#scanImage");
  const scanType = page.querySelector("#scanType");
  const scanTitle = page.querySelector("#scanTitle");
  const scanIndex = page.querySelector("#scanIndex");
  const trayItems = [...page.querySelectorAll(".scan-tray-item")];
  const scanImageReady = new Map(scanSamples.map((sample) => [sample.image, new Promise((resolve, reject) => {
    const preload = new Image();
    preload.onload = resolve;
    preload.onerror = reject;
    preload.src = sample.image;
  })]));
  let scanPosition = 0;
  let scanTimer;
  let renderToken = 0;
  const renderScan = () => {
    const sample = scanSamples[scanPosition];
    const token = ++renderToken;
    scanVisual.classList.remove("is-scanning");
    scanImageReady.get(sample.image).then(() => {
      if (token !== renderToken) return;
      scanImage.src = sample.image;
      scanImage.alt = `${sample.title} classification sample`;
      scanImage.style.setProperty("--scan-focus", sample.focus);
      scanType.textContent = sample.type;
      scanTitle.textContent = sample.title;
      scanIndex.textContent = `${String(scanPosition + 1).padStart(2, "0")} / ${String(scanSamples.length).padStart(2, "0")}`;
      trayItems.forEach((item, index) => item.classList.toggle("is-active", index === scanPosition));
      window.requestAnimationFrame(() => {
        if (token === renderToken) scanVisual.classList.add("is-scanning");
      });
    }).catch(() => console.error(`Unable to load scan sample: ${sample.image}`));
  };
  const advanceScan = () => {
    scanPosition = (scanPosition + 1) % scanSamples.length;
    renderScan();
  };
  if (scanVisual && scanImage && scanType && scanTitle && scanIndex) {
    renderScan();
    if (!reduceMotion) scanTimer = window.setInterval(advanceScan, 4200);
  }

   const radarDataElement = document.querySelector("#radarData");
   let radarData = radarDataElement ? JSON.parse(radarDataElement.textContent) : null;
   const sampleCount = document.querySelector("#radarSampleCount");
   const radarClassCount = document.querySelector("#radarClassCount");
   const radarDomainName = document.querySelector("#radarDomainName");
   const radarDomainDescription = document.querySelector("#radarDomainDescription");
   const radarTimestamp = document.querySelector("#radarTimestamp");
   const radarFootnote = document.querySelector("#radarFootnote");
   const renderRadarDomain = (domainKey) => {
     const domain = radarData?.domains?.[domainKey];
     if (!domain) return;
     radarDomainName.textContent = domain.name;
     radarDomainDescription.textContent = domain.description;
     sampleCount.textContent = domain.dataset_count;
     radarClassCount.textContent = domain.class_count;
     radarTimestamp.textContent = radarData.synced_label;
     radarFootnote.textContent = radarData.footnote;
     page.querySelectorAll(".dataset-node").forEach((node) => {
       const nodeDomain = node.dataset.radarNode;
       node.classList.toggle("is-empty", Boolean(nodeDomain && !radarData.domains[nodeDomain]?.dataset_count));
     });
   };
   const setActiveDomain = (filter) => {
    page.querySelectorAll(".dataset-filter").forEach((item) => {
      const active = item === filter;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-selected", active ? "true" : "false");
    });
     renderRadarDomain(filter.dataset.domain);
   };
   page.querySelectorAll(".dataset-filter").forEach((filter) => filter.addEventListener("click", () => setActiveDomain(filter)));
   const initialFilter = page.querySelector(`.dataset-filter[data-domain="${radarData?.primary_domain || "health"}"]`);
   if (initialFilter) setActiveDomain(initialFilter);
   const radarEndpoint = page.dataset.radarEndpoint;
   const refreshRadar = async () => {
     if (!radarEndpoint) return;
     try {
       const response = await fetch(radarEndpoint, { headers: { "X-Requested-With": "XMLHttpRequest" } });
       if (!response.ok) throw new Error(`Radar request failed: ${response.status}`);
       radarData = await response.json();
       page.querySelectorAll(".dataset-filter").forEach((filter) => {
         const count = radarData.domains[filter.dataset.domain]?.dataset_count ?? 0;
         filter.querySelector("b").textContent = String(count).padStart(2, "0");
       });
       const selectedFilter = page.querySelector(".dataset-filter.is-active") || initialFilter;
       if (selectedFilter) setActiveDomain(selectedFilter);
     } catch (error) {
       console.warn("Live dataset radar refresh unavailable.", error);
     }
   };
   if (radarEndpoint) window.setInterval(refreshRadar, 15000);

    const overviewEndpoint = page.dataset.overviewEndpoint;
    const overviewValueNodes = (key) => page.querySelectorAll(`[data-overview-value="${key}"]`);
    const setOverviewValue = (key, value) => overviewValueNodes(key).forEach((node) => { node.textContent = value; });
    const setOverviewStyle = (key, value) => page.querySelectorAll(`[data-overview-style="${key}"]`).forEach((node) => { node.style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`; });
    const renderOverviewActivity = (state) => {
      const list = page.querySelector("[data-overview-activity-list]");
      const count = page.querySelector("[data-overview-activity-count]");
      if (!list || !count) return;
      count.textContent = `${state.activity.length} recent event${state.activity.length === 1 ? "" : "s"}`;
      list.replaceChildren();
      if (!state.activity.length) {
        const empty = document.createElement("div");
        empty.className = "overview-activity-empty";
        empty.innerHTML = "<span>✓</span><p>Your latest workspace actions will appear here automatically.</p>";
        list.append(empty);
        return;
      }
      const colors = { Dataset: "activity-green", Training: "activity-blue", Prediction: "activity-orange" };
      state.activity.forEach((item) => {
        const row = document.createElement("div");
        row.className = "overview-activity-row";
        const dot = document.createElement("span");
        dot.className = `activity-dot ${colors[item.kind] || "activity-blue"}`;
        const detail = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = item.title;
        const meta = document.createElement("small");
        meta.textContent = `${item.kind} · ${item.detail}`;
        detail.append(title, meta);
        row.append(dot, detail);
        list.append(row);
      });
    };
    const refreshOverview = async () => {
      if (!overviewEndpoint) return;
      try {
        const response = await fetch(overviewEndpoint, { headers: { "X-Requested-With": "XMLHttpRequest" }, cache: "no-store" });
        if (!response.ok) throw new Error("Overview status unavailable");
        const state = await response.json();
        ["dataset_count", "ready_dataset_count", "training_count", "active_training_count", "prediction_count", "workspace_health", "workspace_health_detail"].forEach((key) => setOverviewValue(key, state[key]));
        setOverviewStyle("training_completion", state.training_completion);
        setOverviewStyle("prediction_activity", state.prediction_activity);
        const activeLabel = page.querySelector("[data-overview-active-label]");
        const activeDetail = page.querySelector("[data-overview-active-detail]");
        if (activeLabel) activeLabel.textContent = state.active_training_count ? `${state.active_training_count} job${state.active_training_count === 1 ? "" : "s"} running` : "No active jobs";
        if (activeDetail) activeDetail.textContent = state.active_runs[0] ? `${state.active_runs[0].label} · ${state.active_runs[0].stage}` : "Start an experiment when your dataset is ready.";
        const predictionLabel = page.querySelector("[data-overview-prediction-label]");
        const predictionDetail = page.querySelector("[data-overview-prediction-detail]");
        if (predictionLabel) predictionLabel.textContent = state.latest_prediction?.label || "No prediction yet";
        if (predictionDetail) predictionDetail.textContent = state.latest_prediction ? `${state.latest_prediction.confidence}% confidence · review recommended` : "Upload an image to create the first record.";
        const liveLabel = page.querySelector("[data-overview-live-label]");
        if (liveLabel) liveLabel.textContent = state.active_training_count ? `${state.active_training_count} job${state.active_training_count === 1 ? "" : "s"} live` : "Updating live";
        const updated = page.querySelector("[data-overview-updated]");
        if (updated) updated.textContent = `Updated ${new Date(state.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
        renderOverviewActivity(state);
      } catch (_error) {
        const liveLabel = page.querySelector("[data-overview-live-label]");
        if (liveLabel) liveLabel.textContent = "Live updates paused";
      }
    };
    refreshOverview();
    if (overviewEndpoint) window.setInterval(refreshOverview, 8000);

  const helpSteps = [...page.querySelectorAll(".help-step")];
  const helpNext = document.querySelector("#helpNext");
  const helpBack = document.querySelector("#helpBack");
  const helpLabel = document.querySelector("#helpStepLabel");
  const helpBar = document.querySelector("#helpProgressBar");
  let currentStep = 1;
  const renderStep = () => {
    helpSteps.forEach((step) => step.classList.toggle("is-hidden", Number(step.dataset.step) !== currentStep));
    helpLabel.textContent = `${currentStep} of ${helpSteps.length}`;
    helpBar.style.width = `${(currentStep / helpSteps.length) * 100}%`;
    helpBack.disabled = currentStep === 1;
    helpNext.innerHTML = currentStep === helpSteps.length ? "Start with data <span>→</span>" : "Next step <span>→</span>";
  };
  helpNext?.addEventListener("click", () => {
    if (currentStep < helpSteps.length) { currentStep += 1; renderStep(); }
    else window.location.href = "/datasets/upload/";
  });
  helpBack?.addEventListener("click", () => { if (currentStep > 1) { currentStep -= 1; renderStep(); } });
  renderStep();

  const modal = document.querySelector("#helpModal");
  const modalClose = modal?.querySelector(".help-modal-close");
  const helpOpen = document.querySelector("#helpOpen");
  let lastFocusedElement;
  const openModal = () => {
    if (!modal) return;
    lastFocusedElement = document.activeElement;
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => modalClose?.focus());
  };
  const closeModal = () => {
    if (!modal) return;
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    lastFocusedElement?.focus();
  };
  helpOpen?.addEventListener("click", openModal);
  document.querySelector("#helpReopen")?.addEventListener("click", openModal);
  modal?.querySelectorAll("[data-help-close]").forEach((item) => item.addEventListener("click", closeModal));
  modal?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [...modal.querySelectorAll("button, a[href], [tabindex]:not([tabindex='-1'])")]
      .filter((item) => !item.hasAttribute("disabled"));
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
})();
