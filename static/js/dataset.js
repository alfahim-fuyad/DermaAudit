(() => {
  const fileInput = document.querySelector("#datasetFile");
  const dropzone = document.querySelector("#dropzone");
  const fileName = document.querySelector("#fileName");
  if (!fileInput || !dropzone) return;

  const showFile = (file) => {
    if (!file) return;
    fileName.textContent = file.name;
    dropzone.classList.add("has-file");
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
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showFile(file);
  });
})();
