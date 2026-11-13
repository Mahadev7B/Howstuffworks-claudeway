(function () {
  const toggle = document.getElementById('owlToggle');
  const panel = document.getElementById('askPanel');
  const input = document.getElementById('question');
  function setOpen(open) {
    panel.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (open) (document.getElementById('askerEmailRow').hidden ? input : document.getElementById('askerEmail')).focus();
  }
  toggle.addEventListener('click', () => setOpen(panel.hidden));

  // The grown-up's email is kept only in this browser; no password, no account page.
  const KEY = 'asklilowl-email';
  const row = document.getElementById('askerEmailRow');
  const field = document.getElementById('askerEmail');
  const saved = document.getElementById('askerSaved');
  const savedEmail = document.getElementById('askerSavedEmail');
  function stored() { try { return localStorage.getItem(KEY) || ''; } catch (e) { return ''; } }
  function store(value) { try { value ? localStorage.setItem(KEY, value) : localStorage.removeItem(KEY); } catch (e) {} }
  function showSaved(value) {
    field.value = value;
    savedEmail.textContent = value;
    row.hidden = Boolean(value);
    field.required = !value;
    saved.hidden = !value;
  }
  showSaved(stored());
  document.getElementById('askerChange').addEventListener('click', () => {
    showSaved('');
    field.focus();
  });
  window.AskerEmail = {
    require() {
      const value = field.value.trim().toLowerCase();
      if (!value || !field.checkValidity()) {
        showSaved('');
        setOpen(true);
        field.focus();
        field.reportValidity();
        return '';
      }
      store(value);
      showSaved(value);
      return value;
    },
    update(data) {
      if (data.subscribed || typeof data.free_left !== 'number') return;
      savedEmail.textContent = field.value + (data.free_left === 1 ? ' (1 free question left)' : ' (' + data.free_left + ' free questions left)');
    },
    forget() { store(''); showSaved(''); },
  };
  panel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { setOpen(false); toggle.focus(); }
  });
  document.querySelectorAll('[data-rail-scroll]').forEach((button) => {
    button.addEventListener('click', () => {
      const rail = button.closest('.cine-row').querySelector('.cine-rail');
      rail.scrollBy({ left: Number(button.dataset.railScroll) * rail.clientWidth * 0.8, behavior: 'smooth' });
    });
  });
})();
