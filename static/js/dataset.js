(() => {
  const fileInput = document.querySelector("#datasetFile");
  const dropzone = document.querySelector("#dropzone");
  const fileName = document.querySelector("#fileName");
  const form = document.querySelector("#datasetForm");
  const progressPanel = document.querySelector("#uploadProgress");
  const progressRing = document.querySelector("[data-upload-ring]");
  const progressPercent = document.querySelector("[data-upload-percent]");
  const progressLabel = document.querySelector("[data-upload-label]");
  const progressDetail = document.querySelector("[data-upload-detail]");
  const submit = form?.querySelector('button[type="submit"]');
  const MAX_ARCHIVE_BYTES = 7 * 1024 * 1024 * 1024;
  if (!fileInput || !dropzone) return;

  const updateProgress = (percent, label, detail) => {
    const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
    if (progressRing) progressRing.style.setProperty("--upload-progress", `${safePercent}%`);
    if (progressPercent) progressPercent.textContent = `${safePercent}%`;
    if (progressLabel) progressLabel.textContent = label;
    if (progressDetail) progressDetail.textContent = detail;
  };

  const restoreForm = () => {
    form?.setAttribute("data-uploading", "false");
    dropzone.classList.remove("is-uploading");
    if (submit) {
      submit.disabled = false;
      submit.classList.remove("is-loading");
      submit.querySelector("span")?.replaceChildren(document.createTextNode("Upload and profile"));
    }
  };

  const showUploadError = (label, detail) => {
    if (progressPanel) {
      progressPanel.hidden = false;
      progressPanel.classList.remove("is-complete", "is-processing");
      progressPanel.classList.add("is-error");
      updateProgress(0, label, detail);
    }
    dropzone.classList.add("is-error");
    restoreForm();
  };

  const showFile = (file) => {
    if (!file) return false;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      fileInput.value = "";
      fileName.textContent = "";
      dropzone.classList.remove("has-file");
      showUploadError("ZIP file required", "Choose a dataset with the .zip extension.");
      return false;
    }
    if (file.size > MAX_ARCHIVE_BYTES) {
      fileInput.value = "";
      fileName.textContent = "";
      dropzone.classList.remove("has-file");
      showUploadError("File is too large", "The maximum dataset upload size is 7 GB.");
      return false;
    }
    fileName.textContent = file.name;
    dropzone.classList.add("has-file");
    dropzone.classList.remove("is-error");
    if (progressPanel) {
      progressPanel.hidden = true;
      progressPanel.classList.remove("is-error", "is-complete", "is-processing");
    }
    return true;
  };

  fileInput.addEventListener("change", () => showFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    if (!showFile(file)) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
  });

  form?.addEventListener("submit", (event) => {
    if (!progressPanel || form.dataset.uploading === "true") return;
    event.preventDefault();
    if (!showFile(fileInput.files[0])) return;
    form.dataset.uploading = "true";
    progressPanel.hidden = false;
    progressPanel.classList.remove("is-error", "is-complete", "is-processing");
    dropzone.classList.add("is-uploading");
    dropzone.classList.remove("is-error");
    if (submit) {
      submit.disabled = true;
      submit.classList.add("is-loading");
      submit.querySelector("span")?.replaceChildren(document.createTextNode("Uploading…"));
    }
    updateProgress(0, "Starting upload", "Preparing the ZIP for secure upload.");

    const request = new XMLHttpRequest();
    request.open("POST", form.action || window.location.href);
    request.upload.addEventListener("progress", (uploadEvent) => {
      if (uploadEvent.lengthComputable) {
        updateProgress(
          (uploadEvent.loaded / uploadEvent.total) * 100,
          "Uploading dataset",
          "Your file is transferring securely. Keep this page open.",
        );
      } else {
        updateProgress(55, "Uploading dataset", "Transferring the ZIP file.");
      }
    });
    request.upload.addEventListener("load", () => {
      progressPanel.classList.add("is-processing");
      updateProgress(100, "Upload received", "Running validation, profiling, and auditing…");
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 400) {
        progressPanel.classList.remove("is-processing");
        updateProgress(100, "Upload complete", "Dataset processing finished. Opening the audit report…");
        progressPanel.classList.add("is-complete");
        window.setTimeout(() => {
          window.location.assign(request.responseURL || "/datasets/");
        }, 450);
        return;
      }
      const errors = {
        400: ["Upload request rejected", "Check the ZIP structure and try again."],
        403: ["Upload session expired", "Refresh this page and try again."],
        413: ["File is too large", "The maximum dataset upload size is 7 GB."],
        500: ["Server could not process the dataset", "Try a smaller ZIP or check the dataset structure."],
      };
      const [label, detail] = errors[request.status] || [
        "Upload could not be completed",
        "Please check the file and try again.",
      ];
      showUploadError(label, detail);
    });
    request.addEventListener("error", () => showUploadError(
      "Upload interrupted",
      "Check your connection and try again.",
    ));
    request.addEventListener("timeout", () => showUploadError(
      "Upload timed out",
      "The connection took too long. Keep the ZIP under 7 GB and try again.",
    ));
    request.send(new FormData(form));
  });
})();
