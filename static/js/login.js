(() => {
  const form = document.querySelector('form[action*="login"], form[data-auth-form="login"]');
  if (!form) return;
  form.addEventListener("submit", () => {
    const submit = form.querySelector('button[type="submit"],button:not([type])');
    if (submit) submit.classList.add("is-loading");
  });
})();
