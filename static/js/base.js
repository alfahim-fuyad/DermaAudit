const menuToggle = document.getElementById("menuToggle");
const sidebar = document.getElementById("sidebar");
const notificationToggle = document.getElementById("notificationToggle");
const notificationPanel = document.getElementById("notificationPanel");

// Create backdrop for mobile sidebar
let backdrop = document.querySelector(".sidebar-backdrop");
if (!backdrop) {
  backdrop = document.createElement("div");
  backdrop.className = "sidebar-backdrop";
  backdrop.setAttribute("aria-hidden", "true");
  document.body.appendChild(backdrop);
}

const openSidebar = () => {
  sidebar?.classList.add("open");
  backdrop?.classList.add("is-visible");
  document.body.style.overflow = "hidden";
  menuToggle?.setAttribute("aria-expanded", "true");
};

const closeSidebar = () => {
  sidebar?.classList.remove("open");
  backdrop?.classList.remove("is-visible");
  document.body.style.overflow = "";
  menuToggle?.setAttribute("aria-expanded", "false");
};

const toggleSidebar = () => {
  if (sidebar?.classList.contains("open")) {
    closeSidebar();
  } else {
    openSidebar();
  }
};

menuToggle?.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleSidebar();
});

backdrop?.addEventListener("click", closeSidebar);

// Close sidebar when clicking nav links on mobile
sidebar?.querySelectorAll(".nav-link, .nav-link-logout, .button").forEach((link) => {
  link.addEventListener("click", () => {
    if (window.innerWidth <= 720) {
      closeSidebar();
    }
  });
});

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
  if (sidebar?.classList.contains("open") && !sidebar.contains(event.target) && event.target !== menuToggle && !backdrop.contains(event.target)) {
    closeSidebar();
  }
  if (notificationPanel && !notificationPanel.hidden && !notificationPanel.contains(event.target) && event.target !== notificationToggle) {
    closeNotifications();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeNotifications();
    closeSidebar();
  }
});

// Handle resize: close sidebar when going to desktop
window.addEventListener("resize", () => {
  if (window.innerWidth > 720) {
    closeSidebar();
  }
});

// Auto close toasts
setTimeout(() => document.querySelectorAll(".toast").forEach((toast) => toast.remove()), 5000);

// Add data-labels for mobile table cards
const addTableLabels = () => {
  if (window.innerWidth > 720) return;
  document.querySelectorAll(".data-table").forEach((table) => {
    const headRow = table.querySelector(".table-head");
    if (!headRow) return;
    const labels = [...headRow.querySelectorAll("span")].map(s => s.textContent.trim());
    table.querySelectorAll(".table-row:not(.table-head)").forEach((row) => {
      [...row.querySelectorAll("span")].forEach((cell, idx) => {
        if (labels[idx] && !cell.hasAttribute("data-label")) {
          cell.setAttribute("data-label", labels[idx]);
        }
      });
    });
  });
};
addTableLabels();
window.addEventListener("load", addTableLabels);
