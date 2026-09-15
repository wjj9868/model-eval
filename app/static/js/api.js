/* @author: ztwz 前端基础工具：请求封装、DOM 帮助函数、Toast、模态框 */

/** HTML 转义，防 XSS */
function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** 统一请求封装：成功返回 JSON，失败弹 toast */
async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, opts);
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {
    /* 空响应 */
  }
  if (!resp.ok) {
    const msg =
      data && data.detail
        ? typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
        : `请求失败（HTTP ${resp.status}）`;
    toast(msg, "error");
    throw new Error(msg);
  }
  return data;
}

const toast = (() => {
  const stack = () => document.getElementById("toasts");
  return function (message, type = "", timeout = 2600) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    const icon =
      type === "success" ? "check-circle" : type === "error" ? "alert-circle" : "info";
    el.innerHTML = `<i data-lucide="${icon}"></i><span>${esc(message)}</span>`;
    stack().appendChild(el);
    window.lucide && lucide.createIcons({ attach: el });
    setTimeout(() => {
      el.style.transition = "opacity .3s";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 320);
    }, timeout);
  };
})();

const modal = {
  open(bodyHtml, footHtml = "") {
    const mask = document.getElementById("modal-mask");
    document.getElementById("modal").innerHTML =
      `<div class="modal-body">${bodyHtml}</div>${footHtml ? `<div class="modal-foot">${footHtml}</div>` : ""}`;
    mask.hidden = false;
    window.lucide && lucide.createIcons({ attach: document.getElementById("modal") });
  },
  close() {
    document.getElementById("modal-mask").hidden = true;
    document.getElementById("modal").innerHTML = "";
  },
  mask() {
    return document.getElementById("modal-mask");
  },
};

function icon(name, cls = "") {
  return `<i data-lucide="${name}" class="${cls}"></i>`;
}

function busy(btnEl, busy) {
  if (!btnEl) return;
  btnEl.disabled = busy;
  btnEl.classList.toggle("loading", busy);
  if (busy) btnEl.dataset.origin = btnEl.innerHTML;
  else if (btnEl.dataset.origin) btnEl.innerHTML = btnEl.dataset.origin;
}

function statusBadge(status) {
  return `<span class="badge badge-${esc(status)}">${esc(status)}</span>`;
}

function providerBadge(provider, modelName) {
  const label = provider === "ollama" ? "Ollama" : "OpenAI 兼容";
  return `<span class="badge ${esc(provider)}">${esc(provider === "ollama" ? "Ollama" : "OpenAI")}</span>`;
}

function copyText(text, evt) {
  navigator.clipboard && navigator.clipboard.writeText(text || "");
  if (evt && evt.currentTarget) {
    evt.currentTarget.innerHTML = icon("check");
    setTimeout(() => {
      evt.currentTarget.innerHTML = icon("copy");
      window.lucide && lucide.createIcons({ attach: evt.currentTarget });
    }, 900);
  }
}

/** 简易确认弹窗 */
function confirmModal(title, message, onConfirm, confirmText = "确认") {
  modal.open(
    `<h3 style="margin:0 0 8px">${esc(title)}</h3><p style="margin:0;color:#71788a">${esc(message)}</p>`,
    `<button class="btn" onclick="modal.close()">取消</button>
     <button class="btn danger" id="confirm-btn">${esc(confirmText)}</button>`
  );
  document.getElementById("confirm-btn").onclick = () => {
    modal.close();
    onConfirm();
  };
}