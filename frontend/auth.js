// Metis — auth page logic (register / login).
// Relative path: the backend serves this frontend itself, so API and page
// are always same-origin (see backend/main.py's static mount).
const API_BASE = "/api";

async function api(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function showForm(which) {
  document.getElementById("loginForm").classList.toggle("hidden", which !== "login");
  document.getElementById("registerForm").classList.toggle("hidden", which !== "register");
  document.getElementById("showLoginBtn").classList.toggle("active", which === "login");
  document.getElementById("showRegisterBtn").classList.toggle("active", which === "register");
}

function init() {
  document.getElementById("showLoginBtn").addEventListener("click", () => showForm("login"));
  document.getElementById("showRegisterBtn").addEventListener("click", () => showForm("register"));

  document.getElementById("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;
    const errorEl = document.getElementById("loginError");
    errorEl.textContent = "";

    try {
      const res = await api("/auth/login", { email, password });
      if (!res.success) {
        errorEl.textContent = res.error || "Login failed.";
        return;
      }
      localStorage.setItem("metisUserEmail", email);
      if (res.profile) {
        localStorage.setItem("metisPatientProfile", JSON.stringify(res.profile));
        window.location.href = "dashboard.html";
      } else {
        // Registered before but never completed the metrics form
        window.location.href = "intake.html";
      }
    } catch (err) {
      errorEl.textContent = "Could not reach the server. Is the backend running on port 8000?";
    }
  });

  document.getElementById("registerForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = document.getElementById("registerEmail").value.trim();
    const password = document.getElementById("registerPassword").value;
    const errorEl = document.getElementById("registerError");
    errorEl.textContent = "";

    try {
      const res = await api("/auth/register", { email, password });
      if (!res.success) {
        errorEl.textContent = res.error || "Registration failed.";
        return;
      }
      localStorage.setItem("metisUserEmail", email);
      localStorage.removeItem("metisPatientProfile"); // fresh account, no metrics yet
      window.location.href = "intake.html";
    } catch (err) {
      errorEl.textContent = "Could not reach the server. Is the backend running on port 8000?";
    }
  });
}

document.addEventListener("DOMContentLoaded", init);
