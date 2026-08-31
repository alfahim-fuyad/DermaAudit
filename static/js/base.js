const menuToggle = document.getElementById("menuToggle");
const sidebar = document.getElementById("sidebar");
const notificationToggle = document.getElementById("notificationToggle");
const notificationPanel = document.getElementById("notificationPanel");

menuToggle?.addEventListener("click", () => sidebar?.classList.toggle("open"));

const closeNotifications = () => {
  if (!notificationPanel || !notificationToggle) return;
  notificationPanel.hidden = true;
  notificationToggle.setAttribute("aria-expanded", "false");
};

notificationToggle?.addEventListener("click", (event) => {
  event.stopPropagation();
  if (!notificationPanel) return;
  notificationPanel.hidden = !notificationPanel.hidden;
  notificationToggle.setAttribute("aria-expanded", String(!notificationPanel.hidden));
});

document.addEventListener("click", (event) => {
  if (sidebar?.classList.contains("open") && !sidebar.contains(event.target) && event.target !== menuToggle) {
    sidebar.classList.remove("open");
  }
  if (notificationPanel && !notificationPanel.hidden && !notificationPanel.contains(event.target) && event.target !== notificationToggle) {
    closeNotifications();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeNotifications();
});

setTimeout(() => document.querySelectorAll(".toast").forEach((toast) => toast.remove()), 5000);
