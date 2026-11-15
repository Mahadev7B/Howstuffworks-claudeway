// Subscription waitlist question for the grown-up. Shared by the home page and lesson pages.
(function () {
  const CHOICE_KEY = 'asklilowl-waitlist-choice';
  const CHOICES = [
    { id: 'subscribe', label: '⭐ I want unlimited lessons for $4.99 a month. Add me to the waitlist.' },
    { id: 'browsing', label: "👀 I'm just trying it out and may not come back." },
    { id: 'price', label: "💬 I like it, but $4.99 a month is more than I'd pay." },
  ];
  const THANKS = {
    subscribe: "You're on the waitlist! We'll email you when subscriptions open.",
    browsing: 'Thanks for trying AskLilOwl! The library lessons are always free to watch.',
    price: 'Thanks for telling us. It helps us choose the right price.',
  };

  function read(key) { try { return localStorage.getItem(key) || ''; } catch (e) { return ''; } }
  function asker() {
    if (read('asklilowl-parent-consent') !== 'yes') return null;
    const email = read('asklilowl-email');
    if (!email) return null;
    let token = '';
    try {
      const saved = JSON.parse(read('asklilowl-email-token') || 'null');
      if (saved && saved.email === email) token = saved.token;
    } catch (e) {}
    return { email, token };
  }
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }

  window.Waitlist = {
    answered() { return read(CHOICE_KEY); },
    remember(choice) { try { if (choice) localStorage.setItem(CHOICE_KEY, choice); } catch (e) {} },
    thanks(choice) { return THANKS[choice] || ''; },
    // Fill `box` with the question. `intro` is shown above it; `allowSkip` adds "Maybe later".
    render(box, { intro = '', allowSkip = true } = {}) {
      const who = asker();
      box.replaceChildren();
      if (!who) { box.hidden = true; return false; }
      box.hidden = false;
      if (intro) box.appendChild(el('p', 'waitlist-intro', intro));
      box.appendChild(el('p', 'waitlist-question', 'Quick question for the grown-up: when subscriptions open, which sounds like you?'));
      const list = el('div', 'waitlist-choices');
      const status = el('p', 'waitlist-status');
      status.setAttribute('role', 'status');
      CHOICES.forEach((choice) => {
        const button = el('button', 'waitlist-choice', choice.label);
        button.type = 'button';
        button.addEventListener('click', async () => {
          list.querySelectorAll('button').forEach((b) => { b.disabled = true; });
          status.textContent = 'Saving…';
          try {
            const res = await fetch('/api/waitlist', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ email: who.email, parent_consent: true, email_token: who.token, choice: choice.id }),
            });
            const data = await res.json().catch(() => ({}));
            if (!data.ok) throw new Error(data.error || 'Please try again.');
            window.Waitlist.remember(choice.id);
            box.replaceChildren(el('p', 'waitlist-thanks', THANKS[choice.id]));
          } catch (e) {
            status.textContent = 'Could not save your answer: ' + e.message;
            list.querySelectorAll('button').forEach((b) => { b.disabled = false; });
          }
        });
        list.appendChild(button);
      });
      box.appendChild(list);
      box.appendChild(status);
      if (allowSkip) {
        const skip = el('button', 'waitlist-skip', 'Maybe later');
        skip.type = 'button';
        skip.addEventListener('click', () => { box.hidden = true; });
        box.appendChild(skip);
      }
      return true;
    },
  };
})();
