(() => {
  const input = document.querySelector("#predictionFile");
  const zone = document.querySelector("#predictionDrop");
  const name = document.querySelector("#predictionName");
  const error = document.querySelector("#predictionClientError");
  const form = document.querySelector("#predictionForm");
  const submit = form?.querySelector(".prediction-submit");
  if (!input || !zone) return;

  const allowedTypes = new Set(["jpg", "jpeg", "jfif", "png", "webp", "bmp", "gif", "tif", "tiff"]);
  const showError = (message) => {
    if (!error) return;
    error.textContent = message;
    error.hidden = !message;
  };
  const showFile = (file) => {
    if (!file) {
      name.textContent = "";
      zone.classList.remove("has-file");
      return false;
    }
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!allowedTypes.has(extension)) {
      showError("Choose a JPG, PNG, WEBP, BMP, GIF, or TIFF image.");
      input.value = "";
      showFile(null);
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("This image is larger than the 10 MB limit.");
      input.value = "";
      showFile(null);
      return false;
    }
    showError("");
    name.textContent = file.name;
    zone.classList.add("has-file");
    return true;
  };
  input.addEventListener("change", () => showFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
  }));
  zone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    if (!showFile(file)) return;
    if (typeof DataTransfer !== "undefined") {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
    }
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  form?.addEventListener("submit", (event) => {
    if (!showFile(input.files[0])) {
      event.preventDefault();
      return;
    }
    submit?.classList.add("is-loading");
  });
})();
