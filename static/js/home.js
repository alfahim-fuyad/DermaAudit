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

  const domains = {
    health: { name: "Skin & health imaging", description: "Your workspace is optimized for labeled medical images and careful human review.", classes: "7" },
    environment: { name: "Plant & environment", description: "Image classification for crop health, field conditions, and environmental signals.", classes: "12" },
    vision: { name: "General visual research", description: "Compatible with other labeled image sets when classes and metadata are clear.", classes: "8" }
  };
  const sampleCount = document.querySelector("#radarSampleCount");
  page.querySelectorAll(".dataset-filter").forEach((filter) => filter.addEventListener("click", () => {
    page.querySelectorAll(".dataset-filter").forEach((item) => {
      const active = item === filter;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-selected", active ? "true" : "false");
    });
    const domain = domains[filter.dataset.domain];
    document.querySelector("#radarDomainName").textContent = domain.name;
    document.querySelector("#radarDomainDescription").textContent = domain.description;
    document.querySelector("#radarClassCount").textContent = domain.classes;
    document.querySelector("#radarTimestamp").textContent = "Signal refreshed";
    window.setTimeout(() => { document.querySelector("#radarTimestamp").textContent = "Synced just now"; }, 1800);
  }));

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
