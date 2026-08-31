(() => {
  const input = document.querySelector("#predictionFile");
  const zone = document.querySelector("#predictionDrop");
  const name = document.querySelector("#predictionName");
  if (!input || !zone) return;

  const showFile = (file) => {
    if (!file) return;
    name.textContent = file.name;
    zone.classList.add("has-file");
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
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    showFile(file);
  });
})();
