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
})();
