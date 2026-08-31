(() => {
  const page = document.querySelector(".overview-page");
  if (!page) return;

  page.querySelectorAll("[data-count]").forEach((element) => {
    const target = Number(element.dataset.count);
    if (!Number.isFinite(target)) return;
    element.textContent = target.toLocaleString();
  });

  page.querySelectorAll("[data-toggle-section]").forEach((button) => {
    const target = document.getElementById(button.dataset.toggleSection);
    if (!target) return;
    button.addEventListener("click", () => {
      const isHidden = target.hasAttribute("hidden");
      target.toggleAttribute("hidden", !isHidden);
      button.setAttribute("aria-expanded", String(isHidden));
    });
  });
})();
