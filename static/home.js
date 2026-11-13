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
  const TOKEN_KEY = 'asklilowl-email-token';
  const verifyOn = panel.dataset.verify === '1';
  const verifyBox = document.getElementById('askerVerify');
  const verifyEmail = document.getElementById('askerVerifyEmail');
  const codeInput = document.getElementById('askerCode');
  const codeConfirm = document.getElementById('askerCodeConfirm');
  const codeResend = document.getElementById('askerCodeResend');
  const verifyStatus = document.getElementById('askerVerifyStatus');
  let pending = null;  // { email, resolve } while waiting for a code
  const CONSENT_KEY = 'asklilowl-parent-consent';
  const consent = document.getElementById('askerConsent');
  const row = document.getElementById('askerEmailRow');
  const field = document.getElementById('askerEmail');
  const saved = document.getElementById('askerSaved');
  const savedEmail = document.getElementById('askerSavedEmail');
  // An email is only remembered together with the parent's agreement to the privacy policy.
  function stored() {
    try { return localStorage.getItem(CONSENT_KEY) === 'yes' ? (localStorage.getItem(KEY) || '') : ''; } catch (e) { return ''; }
  }
  function store(value) {
    try {
      if (value) { localStorage.setItem(KEY, value); localStorage.setItem(CONSENT_KEY, 'yes'); }
      else { localStorage.removeItem(KEY); localStorage.removeItem(CONSENT_KEY); localStorage.removeItem(TOKEN_KEY); }
    } catch (e) {}
  }
  function showSaved(value) {
    field.value = value;
    savedEmail.textContent = value;
    row.hidden = Boolean(value);
    field.required = !value;
    consent.required = !value;
    if (!value) consent.checked = false;
    saved.hidden = !value;
  }
  showSaved(stored());
  document.getElementById('askerChange').addEventListener('click', () => {
    store('');
    showSaved('');
    field.focus();
  });

  // A confirmed email is remembered as a signed token from the server.
  function storedToken(email) {
    try {
      const saved = JSON.parse(localStorage.getItem(TOKEN_KEY) || 'null');
      return saved && saved.email === email ? saved.token : '';
    } catch (e) { return ''; }
  }
  function storeToken(email, token) {
    try { token ? localStorage.setItem(TOKEN_KEY, JSON.stringify({ email, token })) : localStorage.removeItem(TOKEN_KEY); } catch (e) {}
  }
  async function post(url, body) {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return res.json().catch(() => ({ ok: false, error: 'Something went wrong. Please try again.' }));
  }
  async function sendCode(email) {
    verifyStatus.textContent = 'Sending a code…';
    const data = await post('/api/email/start', { email, parent_consent: true });
    if (data.ok && data.required === false) return 'not-needed';
    verifyStatus.textContent = data.ok ? '' : (data.error || 'We couldn\'t send the code. Please try again.');
    return data.ok ? 'sent' : 'failed';
  }
  function finish(result) {
    verifyBox.hidden = true;
    codeInput.value = '';
    verifyStatus.textContent = '';
    if (pending) { const done = pending.resolve; pending = null; done(result); }
  }
  async function confirmCode() {
    if (!pending) return;
    const code = codeInput.value.replace(/\D/g, '');
    if (code.length !== 6) { verifyStatus.textContent = 'Type the 6 digits from the email.'; codeInput.focus(); return; }
    codeConfirm.disabled = true;
    verifyStatus.textContent = 'Checking…';
    const data = await post('/api/email/verify', { email: pending.email, code });
    codeConfirm.disabled = false;
    if (data.ok && data.token) {
      storeToken(pending.email, data.token);
      finish({ email: pending.email, token: data.token });
    } else {
      verifyStatus.textContent = data.error || 'That code didn\'t work.';
      codeInput.select();
    }
  }
  codeConfirm.addEventListener('click', confirmCode);
  codeInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); confirmCode(); } });
  codeResend.addEventListener('click', async () => { if (pending) await sendCode(pending.email); if (pending) verifyStatus.textContent = 'A new code is on its way.'; });
  window.AskerEmail = {
    require() {
      const value = field.value.trim().toLowerCase();
      const emailOk = Boolean(value) && field.checkValidity();
      const agreed = row.hidden || consent.checked;
      if (!emailOk || !agreed) {
        row.hidden = false;
        saved.hidden = true;
        field.required = true;
        consent.required = true;
        setOpen(true);
        const problem = emailOk ? consent : field;
        problem.focus();
        problem.reportValidity();
        return '';
      }
      store(value);
      showSaved(value);
      return value;
    },
    update(data) {
      if (data.subscribed) return;
      const plural = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');
      if (typeof data.free_left === 'number') {
        savedEmail.textContent = field.value + ' (' + plural(data.free_left, 'free question') + ' left)';
      } else if (typeof data.new_left_today === 'number') {
        savedEmail.textContent = field.value + ' (free month: ' + plural(data.new_left_today, 'new question') + ' left for now)';
      }
    },
    forget() { store(''); showSaved(''); },
    forgetToken() { storeToken('', ''); },
    // Resolves to { email, token } once the email is entered, agreed and (when needed) confirmed; null if not.
    async confirm() {
      const email = this.require();
      if (!email) return null;
      if (!verifyOn) return { email, token: '' };
      const token = storedToken(email);
      if (token) return { email, token };
      if (pending) finish(null);
      setOpen(true);
      verifyEmail.textContent = email;
      verifyBox.hidden = false;
      const result = await sendCode(email);
      if (result === 'not-needed') { verifyBox.hidden = true; return { email, token: '' }; }
      codeInput.focus();
      return new Promise((resolve) => { pending = { email, resolve }; });
    },
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
