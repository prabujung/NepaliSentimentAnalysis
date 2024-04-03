// prevent from storing state from previous session
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
  }