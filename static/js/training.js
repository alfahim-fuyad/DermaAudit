(() => {
  const cards = [...document.querySelectorAll(".architecture-card")];
  const select = document.querySelector('select[name="architecture"]');
  if (!cards.length || !select) return;

  const enhanceSelect = (nativeSelect, id) => {
    const control = document.createElement("div");
    control.className = "select-control";
    control.dataset.selectControl = "";
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "select-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", id);
    const triggerValue = document.createElement("span");
    triggerValue.className = "select-trigger-value";
    trigger.append(triggerValue);

    const menu = document.createElement("div");
    menu.className = "select-menu";
    menu.id = id;
    menu.setAttribute("role", "listbox");
    menu.hidden = true;
    const optionButtons = [...nativeSelect.options].map((option, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "select-option";
      button.dataset.value = option.value;
      button.dataset.index = String(index);
      button.setAttribute("role", "option");
      button.innerHTML = `<span>${option.textContent}</span><b aria-hidden="true">✓</b>`;
      button.addEventListener("click", () => {
        nativeSelect.value = option.value;
        nativeSelect.dispatchEvent(new Event("change", { bubbles: true }));
        close();
        trigger.focus();
      });
      menu.append(button);
      return button;
    });

    const update = () => {
      const selected = nativeSelect.options[nativeSelect.selectedIndex] || nativeSelect.options[0];
      triggerValue.textContent = selected?.textContent || "";
      trigger.classList.toggle("is-placeholder", !nativeSelect.value);
      optionButtons.forEach((button) => {
        const active = button.dataset.value === nativeSelect.value;
        button.classList.toggle("is-selected", active);
        button.setAttribute("aria-selected", String(active));
      });
    };
    const close = () => {
      control.classList.remove("is-open");
      trigger.setAttribute("aria-expanded", "false");
      menu.hidden = true;
    };
    const open = () => {
      menu.hidden = false;
      control.classList.add("is-open");
      trigger.setAttribute("aria-expanded", "true");
    };
    const toggle = () => (control.classList.contains("is-open") ? close() : open());

    trigger.addEventListener("click", toggle);
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        close();
        return;
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
        return;
      }
      if (!control.classList.contains("is-open")) return;
      const current = Math.max(0, nativeSelect.selectedIndex);
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const direction = event.key === "ArrowDown" ? 1 : -1;
        const next = Math.min(optionButtons.length - 1, Math.max(0, current + direction));
        nativeSelect.selectedIndex = next;
        nativeSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        nativeSelect.selectedIndex = event.key === "Home" ? 0 : optionButtons.length - 1;
        nativeSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    nativeSelect.addEventListener("change", update);
    nativeSelect.classList.add("premium-native-select");
    nativeSelect.parentNode.insertBefore(control, nativeSelect);
    control.append(trigger, menu, nativeSelect);
    update();
  };

  enhanceSelect(document.querySelector('select[name="dataset"]'), "dataset-options");
  enhanceSelect(select, "architecture-options");
  document.addEventListener("click", (event) => {
    document.querySelectorAll(".select-control.is-open").forEach((control) => {
      if (!control.contains(event.target)) {
        control.classList.remove("is-open");
        const trigger = control.querySelector(".select-trigger");
        const menu = control.querySelector(".select-menu");
        trigger?.setAttribute("aria-expanded", "false");
        if (menu) menu.hidden = true;
      }
    });
  });

  const choose = (value) => {
    cards.forEach((card) => {
      const active = card.dataset.architecture === value;
      card.classList.toggle("is-selected", active);
      card.classList.toggle("selected", active);
      card.setAttribute("aria-pressed", String(active));
    });
  };
  cards.forEach((card, index) => {
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    const value = card.dataset.architecture || ["efficientnet_b0", "resnet50", "mobilenet_v3"][index];
    card.dataset.architecture = value;
    card.addEventListener("click", () => {
      select.value = value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select.value = value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
  });
  select.addEventListener("change", () => choose(select.value));
  choose(select.value);

  const form = document.getElementById("trainingForm");
  const submit = document.getElementById("trainingSubmit");
  const submitStatus = document.getElementById("trainingSubmitStatus");
  form?.addEventListener("submit", () => {
    if (submit?.disabled) return;
    if (submit) {
      submit.disabled = true;
      submit.classList.add("is-loading");
      submit.querySelector("span").textContent = "Starting…";
    }
    if (submitStatus) submitStatus.hidden = false;
  });

  const progressNodes = [
    ...document.querySelectorAll('[data-training-run][data-run-status="running"]'),
    ...document.querySelectorAll("[data-training-progress]"),
  ];
  const pollUrls = [...new Set(progressNodes.map((node) => node.dataset.statusUrl).filter(Boolean))];
  if (pollUrls.length) {
    let reloading = false;
    const renderProgress = (panel, state) => {
      const progress = state.progress || {};
      const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
      const steps = ["validate", "load", "prepare", "train", "evaluate", "checkpoint"];
      const isComplete = state.status === "completed" || progress.step === "complete";
      const currentIndex = isComplete ? steps.length : steps.indexOf(progress.step);
      panel.querySelector("[data-progress-fill]")?.style.setProperty("width", `${percent}%`);
      const percentNode = panel.querySelector("[data-progress-percent]");
      if (percentNode) percentNode.textContent = `${percent}%`;
      const stageNode = panel.querySelector("[data-progress-stage]");
      if (stageNode) stageNode.textContent = progress.label || "Training in progress";
      const detailNode = panel.querySelector("[data-progress-detail]");
      if (detailNode) detailNode.textContent = progress.detail || "Working through the next step.";
      const countNode = panel.querySelector("[data-progress-step-count]");
      if (countNode) countNode.textContent = isComplete
        ? "Complete"
        : currentIndex < 0 ? "Finishing up" : `Step ${currentIndex + 1} of ${steps.length}`;
      panel.querySelectorAll("[data-progress-step]").forEach((stepNode, index) => {
        stepNode.classList.toggle("is-complete", isComplete || (currentIndex >= 0 && index < currentIndex));
        stepNode.classList.toggle("is-active", !isComplete && (index === currentIndex || (currentIndex < 0 && index === 0)));
      });
    };

    const poll = async () => {
      const states = await Promise.all(pollUrls.map(async (url) => {
        try {
          const response = await fetch(url, {
            headers: { "X-Requested-With": "XMLHttpRequest" },
            cache: "no-store",
          });
          if (!response.ok) return "running";
          const state = await response.json();
          progressNodes.filter((node) => node.dataset.statusUrl === url).forEach((node) => {
            if (node.matches("[data-training-progress]")) renderProgress(node, state);
          });
          if (state.status !== "running" && !reloading) {
            reloading = true;
            const panels = progressNodes.filter((node) => node.dataset.statusUrl === url && node.matches("[data-training-progress]"));
            panels.forEach((panel) => {
              const complete = state.status === "completed";
              panel.classList.toggle("is-complete", complete);
              panel.classList.toggle("is-failed", !complete);
              const title = panel.querySelector("[data-progress-title]");
              if (title) title.textContent = complete ? "Training complete" : "Training needs attention";
              const badge = panel.querySelector(".progress-live-badge");
              if (badge) {
                badge.classList.toggle("is-complete", complete);
                badge.classList.toggle("is-failed", !complete);
                const badgeText = badge.querySelector("span");
                if (badgeText) badgeText.textContent = complete ? "COMPLETE" : "FAILED";
              }
              const message = panel.querySelector("[data-training-complete-message]");
              if (message) {
                message.hidden = false;
                message.textContent = complete
                  ? "✓ Training complete — checkpoint saved and ready for prediction."
                  : `Training needs attention — ${state.error || "review the run details and try again."}`;
                message.classList.toggle("is-error", !complete);
              }
              renderProgress(panel, state);
            });
            window.setTimeout(() => window.location.reload(), 4500);
          }
          return state.status;
        } catch (_error) {
          return "running";
        }
      }));
      if (states.some((state) => state === "running") && !reloading) {
        window.setTimeout(poll, 2000);
      }
    };
    window.setTimeout(poll, 1200);
  }
})();
