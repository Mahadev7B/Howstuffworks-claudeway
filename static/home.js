(function () {
  const toggle = document.getElementById('owlToggle');
  const panel = document.getElementById('askPanel');
  const input = document.getElementById('question');
  function setOpen(open) {
    panel.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (open) input.focus();
  }
  toggle.addEventListener('click', () => setOpen(panel.hidden));
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
