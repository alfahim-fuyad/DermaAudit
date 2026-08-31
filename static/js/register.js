(() => {
  const form = document.querySelector('form[action*="register"], form[data-auth-form="register"]');
  if (!form) return;
  form.addEventListener("submit", (event) => {
    const password = form.querySelector('input[name="password"]');
    const confirmation = form.querySelector('input[name="password_confirm"],input[name="password_confirmation"]');
    if (password && confirmation && password.value !== confirmation.value) {
      event.preventDefault();
      confirmation.setCustomValidity("Passwords must match.");
      confirmation.reportValidity();
      return;
    }
    if (confirmation) confirmation.setCustomValidity("");
    const submit = form.querySelector('button[type="submit"],button:not([type])');
    if (submit) submit.classList.add("is-loading");
  });
})();
