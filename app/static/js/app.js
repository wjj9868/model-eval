/* @author: ztwz 前端主逻辑：路由 + 五视图（工作台/模型/用例/批次/新建）+ 轮询 */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /* ============ 全局态 ============ */
  let pollTimer = null;
  const viewState = {
    models: { page: 1, keyword: "" },
    prompts: { page: 1, keyword: "", category: "" },
    batches: { page: 1, status: "" },
    detail: { page: 1, status: "", pollKey: 0 },
    create: { step: 1, modelIds: new Set(), promptIds: new Set(), keyword: "", category: "" },
  };
  const TERMINAL = ["COMPLETED", "PARTIAL", "FAILED"];
  const fmt = (s) => (s || "").replace("T", " ").slice(0, 16);

  /* ============ 工具 ============ */
  function icons(root) {
    if (window.lucide) lucide.createIcons({ attach: root });
  }
  function stopPoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }
  function schedulePoll(fn, ms) {
    stopPoll();
    pollTimer = setInterval(fn, ms);
  }
  function pagerHtml(page, total, pageSize) {
    const pages = Math.max(1, Math.ceil(total / pageSize));
    if (pages <= 1) return "";
    return `<div class="pager">
      <button class="btn sm" data-page="${page - 1}" ${page <= 1 ? "disabled" : ""}>上一页</button>
      <span>第 ${page} / ${pages} 页 · 共 ${total} 条</span>
      <button class="btn sm" data-page="${page + 1}" ${page >= pages ? "disabled" : ""}>下一页</button>
    </div>`;
  }
  function wirePager(el, setPage) {
    $$("[data-page]", el).forEach((b) => (b.onclick = () => setPage(Number(b.dataset.page))));
  }
  function progressBar(p) {
    const running = Math.max(p.total_tasks - Math.max(p.completed_tasks, 0), 0);
    const ratio = p.total_tasks ? Math.round((p.succeed_tasks / p.total_tasks) * 100) : 0;
    return `<div style="display:flex;align-items:center;gap:10px">
      <div class="progress" style="width:150px">
        <div class="ok" style="flex:${p.succeed_tasks}"></div>
        <div class="bad" style="flex:${p.failed_tasks}"></div>
        <div class="runnig" style="flex:${running}"></div>
      </div>
      <span class="counts"><span class="s-ok"><b>${p.succeed_tasks}</b> 成功</span><span class="s-bad"><b>${p.failed_tasks}</b> 失败</span>${p.completed_tasks}/${p.total_tasks} · ${ratio}%</span>
    </div>`;
  }

  /* ============ 路由 ============ */
  function router() {
    const hash = location.hash.slice(1) || "/";
    stopPoll();
    const view = $("#view");
    navActive(hash);
    if (hash === "/" || hash === "") return viewDashboard(view);
    if (hash === "/models") return viewModels(view);
    if (hash === "/prompts") return viewPrompts(view);
    if (hash === "/batches") return viewBatches(view);
    if (hash === "/create") return viewCreate(view);
    const m = hash.match(/^\/batch\/(\d+)/);
    if (m) return viewBatchDetail(view, Number(m[1]));
    view.innerHTML = `<div class="empty">${icon("compass")}<div>页面不存在</div></div>`;
  }
  function navActive(hash) {
    $$(".nav a").forEach((a) => {
      const key = a.dataset.nav;
      let active = false;
      if (key === "dashboard") active = hash === "/";
      else if (key === "batches") active = hash === "/batches" || hash.startsWith("/batch/");
      else active = hash === `/${key}`;
      a.classList.toggle("active", active);
    });
  }
  window.addEventListener("hashchange", router);

  /* ============ 视图：工作台 ============ */
  async function viewDashboard(el) {
    el.innerHTML = `<div class="empty">加载中…</div>`;
    await renderDashboard(el);
    schedulePoll(() => {
      if ((location.hash.slice(1) || "/") === "/") renderDashboard(el);
    }, 8000);
  }

  async function renderDashboard(el) {
    const stats = await api("/api/stats");
    const rate = stats.success_rate === null ? "—" : Math.round(stats.success_rate * 100) + "%";
    el.innerHTML = `
      <div class="page-head">
        <div><h1 class="page-title">工作台</h1><div class="page-sub">开源模型对比评测 · 第一期为「同输入多模型输出对比」</div></div>
        <a href="#/create" class="btn primary">${icon("plus-circle")}新建评测</a>
      </div>
      <div class="stats-grid">
        <div class="card stat-card"><div class="label">${icon("cpu", "stat-icon")}启用模型</div><div class="value">${stats.model_count}<span class="unit">个</span></div></div>
        <div class="card stat-card"><div class="label">${icon("list-checks", "stat-icon")}启用用例</div><div class="value">${stats.prompt_count}<span class="unit">条</span></div></div>
        <div class="card stat-card"><div class="label">${icon("history", "stat-icon")}评测批次</div><div class="value">${stats.batch_count}<span class="unit">次</span></div></div>
        <div class="card stat-card"><div class="label">${icon("target", "stat-icon")}任务成功率</div><div class="value">${rate}</div><div class="hint">成功 ${stats.succeed_tasks} · 失败 ${stats.failed_tasks} · 共 ${stats.total_tasks} 任务</div></div>
      </div>
      <div class="card">
        <div class="page-head" style="padding:14px 18px;margin:0;border-bottom:1px solid var(--border)">
          <div><h3 style="margin:0;font-size:15px">最近评测批次</h3></div>
          <a href="#/batches" class="btn sm ghost">查看全部 ${icon("chevron-right")}</a>
        </div>
        ${stats.recent_batches.length ? `
        <table class="table">
          <thead><tr><th>名称</th><th>状态</th><th>进度</th><th>规模</th><th>创建时间</th></tr></thead>
          <tbody>${stats.recent_batches.map((b) => `
            <tr style="cursor:pointer" onclick="location.hash='#/batch/${b.id}'">
              <td><b>${esc(b.name)}</b>${b.remark ? `<div style="color:#aab0bf;font-size:12px">${esc(b.remark)}</div>` : ""}</td>
              <td>${statusBadge(b.status)}</td>
              <td>${progressBar(b)}</td>
              <td>${b.model_count} 模型 × ${b.prompt_count} 用例</td>
              <td style="color:#71788a">${esc(fmt(b.created_at))}</td>
            </tr>`).join("")}
          </tbody>
        </table>` : `<div class="empty">${icon("inbox")}<div>还没有评测批次，点击右上角「新建评测」开始第一次对比</div></div>`}
      </div>`;
    icons(el);
  }

  /* ============ 视图：模型管理 ============ */
  async function viewModels(el) {
    const st = viewState.models;
    el.innerHTML = `
      <div class="page-head">
        <div><h1 class="page-title">模型管理</h1><div class="page-sub">配置已部署的开源模型：Ollama 或 OpenAI 兼容服务（vLLM / LM Studio / Xinference 等；unsloth 训练后经 vLLM 暴露即该协议）</div></div>
        <button class="btn primary" id="add-model">${icon("plus")}新增模型</button>
      </div>
      <div class="filters" style="margin-bottom:16px">
        <div class="search-box"><input class="input" id="kw" placeholder="搜索名称 / 模型名" value="${esc(st.keyword)}" /></div>
        <button class="btn" id="search-btn">${icon("search")}搜索</button>
      </div>
      <div class="model-grid" id="model-grid"><div class="empty">加载中…</div></div>
      <div id="model-pager"></div>`;
    $("#add-model").onclick = () => modelFormModal();
    $("#search-btn").onclick = () => { st.keyword = $("#kw").value.trim(); st.page = 1; renderModels(el); };
    $("#kw").addEventListener("keydown", (e) => { if (e.key === "Enter") { st.keyword = e.target.value.trim(); st.page = 1; renderModels(el); } });
    icons(el);
    await renderModels(el);
  }

  async function renderModels(el) {
    const st = viewState.models;
    const data = await api(`/api/models?page=${st.page}&page_size=9&keyword=${encodeURIComponent(st.keyword || "")}`);
    const grid = $("#model-grid");
    grid.innerHTML = data.items.length ? data.items.map((m) => `
      <div class="card model-card">
        <div class="mc-head">
          <div style="min-width:0">
            <div class="mc-name">${esc(m.name)} ${!m.enabled ? `<span class="badge ollama">已停用</span>` : ""}</div>
            <div class="mc-url">${icon("link")}${esc(m.base_url)} ${providerBadge(m.provider)}</div>
          </div>
          <div>
            <span class="badge ollama">${esc(m.provider === "ollama" ? "Ollama" : "OpenAI")}</span>
            <label class="switch" title="启用/停用">
              <input type="checkbox" ${m.enabled ? "checked" : ""} data-toggle="${m.id}" />
            </label>
          </div>
        </div>
        <div class="mc-meta"><b>模型</b> ${esc(m.model_name)} · <b>max_tokens</b> ${m.max_tokens} · <b>温度</b> ${m.temperature} · <b>超时</b> ${m.timeout_seconds}s</div>
        ${m.system_prompt ? `<div class="mc-meta" style="color:#aab0bf">系统提示词：${esc(m.system_prompt.slice(0, 40))}</div>` : ""}
        <div class="mc-meta" id="test-result-${m.id}"></div>
        <div class="mc-actions">
          <button class="btn sm" data-test="${m.id}">${icon("zap")}测试连接</button>
          <button class="btn sm" data-edit="${m.id}">${icon("pencil")}编辑</button>
          <button class="btn sm danger" data-del="${m.id}">${icon("trash-2")}删除</button>
        </div>
      </div>`).join("") : `<div class="empty" style="grid-column:1/-1">${icon("cpu")}<div>还没有模型配置，点击「新增模型」接入第一个开源模型</div></div>`;
    $("#model-pager").innerHTML = pagerHtml(st.page, data.total, data.page_size);
    wirePager($("#model-pager"), (p) => { st.page = p; renderModels(el); });
    icons(el);

    $$("[data-toggle]", grid).forEach((cb) => (cb.onchange = async () => {
      await api(`/api/models/${cb.dataset.toggle}/enabled`, { method: "PATCH", body: { enabled: cb.checked } });
      toast(cb.checked ? "已启用" : "已停用", "success");
      renderModels(el);
    }));
    $$("[data-test]", grid).forEach((btn) => (btn.onclick = async () => {
      busy(btn, true);
      btn.innerHTML = icon("loader"); icons(el);
      const r = await api(`/api/models/${btn.dataset.test}/test`, { method: "POST" });
      busy(btn, false);
      $("#test-result-" + btn.dataset.test).innerHTML = r.success
        ? `<span style="color:var(--green)">${icon("check-circle")} 连通正常 · ${r.response_time_ms}ms · ${esc((r.sample || "").slice(0, 40))}</span>`
        : `<span style="color:var(--red);word-break:break-all">${icon("alert-circle")} ${esc(r.message)}</span>`;
      icons(el);
    }));
    $$("[data-edit]", grid).forEach((btn) => (btn.onclick = () => modelFormModal(Number(btn.dataset.edit))));
    $$("[data-del]", grid).forEach((btn) =>
      (btn.onclick = () => confirmModal("删除模型配置", "删除后不可恢复（已被评测批次引用的模型禁止删除）", async () => {
        await api(`/api/models/${btn.dataset.del}`, { method: "DELETE" });
        toast("已删除", "success");
        renderModels(el);
      })));
  }

  /* ============ 模型表单弹窗 ============ */
  async function modelFormModal(id) {
    let m = null;
    if (id) {
      const list = await api("/api/models?page=1&page_size=100");
      m = list.items.find((x) => x.id === id);
    }
    const d = m || {};
    modal.open(`
      <div class="form-row"><label>配置名称 *</label><input class="input" name="name" value="${esc(d.name || "")}" placeholder="例如：Qwen2.5-7B（本机）" /></div>
      <div class="form-grid">
        <div class="form-row"><label>接入协议 *</label>
          <select class="select" name="provider">
            <option value="ollama" ${d.provider === "openai" ? "" : "selected"}>Ollama</option>
            <option value="openai" ${d.provider === "openai" ? "selected" : ""}>OpenAI 兼容（vLLM 等）</option>
          </select></div>
        <div class="form-row"><label>模型名 *</label><input class="input" name="model_name" value="${esc(d.model_name || "")}" placeholder="如 qwen2.5:7b 或 /models/Qwen2.5-7B-Instruct" /></div>
      </div>
      <div class="form-row"><label>服务地址 *（服务根地址，协议路径自动拼接）</label><input class="input" name="base_url" value="${esc(d.base_url || "")}" placeholder="Ollama：http://127.0.0.1:11434/ ｜ OpenAI 兼容：http://127.0.0.1:8000/" />
        <div class="hint">Ollama 自动拼 /api/chat；OpenAI 兼容自动拼 /v1/chat/completions</div></div>
      <div class="form-row"><label>API Key（OpenAI 兼容接口需要）</label><input class="input" type="password" name="api_key" value="${esc(d.api_key || "")}" placeholder="sk-..." /></div>
      <div class="form-row"><label>系统提示词（可选）</label><textarea class="textarea" name="system_prompt" rows="2" placeholder="例如：你是严谨的评测助手">${esc(d.system_prompt || "")}</textarea></div>
      <div class="form-grid">
        <div class="form-row"><label>最大生成 Token</label><input class="input" type="number" name="max_tokens" value="${d.max_tokens ?? 1024}" min="1" max="8192" /></div>
        <div class="form-row"><label>采样温度</label><input class="input" type="number" step="0.1" name="temperature" value="${d.temperature ?? 0.7}" min="0" max="2" /></div>
      </div>
      <div class="form-grid">
        <div class="form-row"><label>超时（秒）</label><input class="input" type="number" name="timeout_seconds" value="${d.timeout_seconds ?? 300}" min="1" max="900" /></div>
        <div class="form-row"><label>备注</label><input class="input" name="remark" value="${esc(d.remark || "")}" placeholder="可选" /></div>
      </div>`,
      `<button class="btn" onclick="modal.close()">取消</button><button class="btn primary" id="save-model">${icon("save")}保存</button>`);
    icons($("#modal"));
    $("#save-model").onclick = async () => {
      const btn = $("#save-model");
      busy(btn, true);
      const payload = {
        name: $('[name="name"]').value.trim(),
        provider: $('[name="provider"]').value,
        base_url: $('[name="base_url"]').value.trim(),
        api_key: $('[name="api_key"]').value.trim() || null,
        model_name: $('[name="model_name"]').value.trim(),
        system_prompt: $('[name="system_prompt"]').value,
        max_tokens: Number($('[name="max_tokens"]').value),
        temperature: Number($('[name="temperature"]').value),
        timeout_seconds: Number($('[name="timeout_seconds"]').value),
        remark: $('[name="remark"]').value.trim() || null,
      };
      try {
        if (id) await api(`/api/models/${id}`, { method: "PUT", body: payload });
        else await api("/api/models", { method: "POST", body: payload });
        modal.close();
        toast("保存成功", "success");
        if ((location.hash.slice(1) || "/") === "/models") renderModels($("#view"));
      } catch (e) {
        busy(btn, false);
      }
    };
  }

  /* ============ 视图：测试用例 ============ */
  async function viewPrompts(el) {
    const st = viewState.prompts;
    const cats = await api("/api/prompts/categories");
    el.innerHTML = `
      <div class="page-head">
        <div><h1 class="page-title">测试用例</h1><div class="page-sub">评测输入，支持批量导入（粘贴文本 / txt / csv / jsonl）</div></div>
        <div style="display:flex;gap:8px">
          <button class="btn" id="batch-del-prompt">${icon("trash-2")}批量删除</button>
          <button class="btn" id="import-prompt">${icon("upload")}批量导入</button>
          <button class="btn primary" id="add-prompt">${icon("plus")}新建用例</button>
        </div>
      </div>
      <div class="filters" style="margin-bottom:16px">
        <div class="search-box"><input class="input" id="kw" placeholder="搜索内容" value="${esc(st.keyword)}" /></div>
        <select class="select" id="cat">
          <option value="">全部分类</option>
          ${cats.map((c) => `<option value="${esc(c)}" ${st.category === c ? "selected" : ""}>${esc(c)}</option>`).join("")}
        </select>
        <span class="page-sub" id="prompt-total"></span>
      </div>
      <div class="card" style="overflow:auto">
        <table class="table">
          <thead><tr>
            <th style="width:36px"><input type="checkbox" id="check-all" style="accent-color:var(--primary)" /></th>
            <th style="width:80px">分类</th><th>内容</th><th>创建时间</th><th style="width:110px">操作</th>
          </tr></thead>
          <tbody id="prompt-tbody"></tbody>
        </table>
        <div id="prompt-empty"></div>
      </div>
      <div id="prompt-pager"></div>`;
    $("#add-prompt").onclick = () => promptFormModal();
    $("#import-prompt").onclick = importModal;
    $("#batch-del-prompt").onclick = batchDeletePrompts;
    $("#kw").addEventListener("keydown", (e) => { if (e.key === "Enter") { st.keyword = e.target.value.trim(); st.page = 1; renderPrompts(el); } });
    $("#cat").onchange = () => { st.category = $("#cat").value; st.page = 1; renderPrompts(el); };
    icons(el);
    await renderPrompts(el);
  }

  async function renderPrompts(el) {
    const st = viewState.prompts;
    const data = await api(`/api/prompts?page=${st.page}&page_size=10&keyword=${encodeURIComponent(st.keyword || "")}&category=${encodeURIComponent(st.category || "")}`);
    $("#prompt-total").textContent = `共 ${data.total} 条`;
    const tbody = $("#prompt-tbody");
    tbody.innerHTML = data.items.map((p) => `
      <tr>
        <td><input type="checkbox" class="row-check" value="${p.id}" style="accent-color:var(--primary)" /></td>
        <td><span class="badge ollama">${esc(p.category)}</span></td>
        <td>
          <div style="max-width:520px;white-space:pre-wrap;word-break:break-word;max-height:64px;overflow:hidden">${esc(p.content)}</div>
          ${p.expected_answer ? `<div style="color:#aab0bf;font-size:12px">期望答案：${esc(p.expected_answer.slice(0, 50))}</div>` : ""}
        </td>
        <td style="color:#71788a">${esc(fmt(p.created_at))}</td>
        <td><div class="row-actions">
          <button class="btn sm" data-edit="${p.id}">${icon("pencil")}</button>
          <button class="btn sm danger" data-del="${p.id}">${icon("trash-2")}</button>
        </div></td>
      </tr>`).join("");
    $("#prompt-empty").innerHTML = data.items.length ? "" : `<div class="empty">${icon("inbox")}<div>还没有用例，点击「批量导入」快速添加</div></div>`;
    $("#prompt-pager").innerHTML = pagerHtml(st.page, data.total, data.page_size);
    wirePager($("#prompt-pager"), (p) => { st.page = p; renderPrompts(el); });
    $("#check-all").onchange = (e) => { $$(".row-check", tbody).forEach((c) => (c.checked = e.target.checked)); };
    $$("[data-edit]", tbody).forEach((b) => (b.onclick = () => promptFormModal(Number(b.dataset.edit))));
    $$("[data-del]", tbody).forEach((b) =>
      (b.onclick = () => confirmModal("删除用例", "删除后不可恢复（已被评测任务引用的用例禁止删除）", async () => {
        await api(`/api/prompts/${b.dataset.del}`, { method: "DELETE" });
        toast("已删除", "success"); renderPrompts(el);
      })));
    icons(el);
  }

  async function batchDeletePrompts() {
    const ids = $$(".row-check").filter((c) => c.checked).map((c) => Number(c.value));
    if (!ids.length) return toast("请先勾选用例", "error");
    confirmModal("批量删除", `将删除 ${ids.length} 条用例（被评测任务引用的将整体拒绝）`, async () => {
      await api("/api/prompts/batch-delete", { method: "POST", body: { ids } });
      toast("删除完成", "success");
      renderPrompts($("#view"));
    });
  }

  function promptFormModal(id) {
    let loaded = false;
    modal.open(`
      <div class="form-row"><label>用例内容 *</label><textarea class="textarea" name="content" rows="5" placeholder="输入要评测的问题"></textarea></div>
      <div class="form-grid">
        <div class="form-row"><label>分类</label><input class="input" name="category" value="默认" /></div>
        <div class="form-row"><label>期望答案（可选）</label><input class="input" name="expected_answer" placeholder="预留给后续自动评分" /></div>
      </div>`,
      `<button class="btn" onclick="modal.close()">取消</button><button class="btn primary" id="save-prompt">保存</button>`);
    icons($("#modal"));
    (async () => {
      const data = await api("/api/prompts?page=1&page_size=100");
      const p = data.items.find((x) => x.id === id);
      if (p) {
        $('[name="content"]').value = p.content;
        $('[name="category"]').value = p.category;
        $('[name="expected_answer"]').value = p.expected_answer || "";
        loaded = true;
      }
    })();
    $("#save-prompt").onclick = async () => {
      const body = {
        content: $('[name="content"]').value,
        category: $('[name="category"]').value.trim() || "默认",
        expected_answer: $('[name="expected_answer"]').value || null,
      };
      if (id) await api(`/api/prompts/${id}`, { method: "PUT", body });
      else await api("/api/prompts", { method: "POST", body });
      modal.close();
      toast("保存成功", "success");
      if ((location.hash.slice(1) || "/") === "/prompts") renderPrompts($("#view"));
    };
  }

  function importModal() {
    modal.open(`
      <div class="form-row"><label>导入方式</label>
        <div class="form-check" style="gap:14px">
          <label class="radio"><input type="radio" name="way" value="text" checked /> 粘贴文本</label>
          <label class="radio"><input type="radio" name="way" value="file" /> 上传文件</label>
        </div></div>
      <div class="form-row" id="text-wrap"><label>每行一条用例</label><textarea class="textarea" id="paste-text" rows="6" placeholder="第一行问题&#10;第二行问题&#10;..."></textarea></div>
      <div class="form-row" id="file-wrap" hidden><label>支持 .txt / .csv / .jsonl（csv 第二列为分类）</label><input class="input" type="file" id="import-file" accept=".txt,.csv,.json,.jsonl" /></div>
      <div class="form-row"><label>默认分类</label><input class="input" id="import-cat" value="默认" /></div>
      <div id="import-result"></div>`,
      `<button class="btn" onclick="modal.close()">关闭</button><button class="btn primary" id="do-import">开始导入</button>`);
    icons($("#modal"));
    $$('input[name="way"]', $("#modal")).forEach((r) => (r.onchange = () => {
      $("#text-wrap").hidden = r.value !== "text";
      $("#file-wrap").hidden = r.value !== "file";
    }));
    $("#do-import").onclick = async () => {
      const btn = $("#do-import");
      busy(btn, true);
      const way = $('input[name="way"]:checked').value;
      const cat = $("#import-cat").value.trim() || "默认";
      try {
        const fd = new FormData();
        fd.append("default_category", cat);
        if (way === "file") {
          const file = $("#import-file").files[0];
          if (!file) return toast("请选择文件", "error");
          fd.append("file", file);
        } else {
          const text = $("#paste-text").value;
          if (!text.trim()) return toast("请粘贴内容", "error");
          fd.append("text", text);
        }
        const r = await api("/api/prompts/import", { method: "POST", body: fd });
        $("#import-result").innerHTML = `<div style="color:var(--green);font-size:13px">${icon("check-circle")} 新增 ${r.imported_count} 条 · 跳过重复 ${r.skipped_count} 条 · 忽略无效 ${r.invalid_count} 条</div>`;
        icons($("#modal"));
        if (r.imported_count > 0 && (location.hash.slice(1) || "/") === "/prompts") renderPrompts($("#view"));
      } catch (e) { /* toast 已提示 */ } finally {
        busy(btn, false);
      }
    };
  }

  /* ============ 视图：评测批次列表 ============ */
  async function viewBatches(el) {
    const st = viewState.batches;
    el.innerHTML = `
      <div class="page-head">
        <div><h1 class="page-title">评测批次</h1><div class="page-sub">同输入多模型输出对比，点击批次查看结果</div></div>
        <a href="#/create" class="btn primary">${icon("plus-circle")}新建评测</a>
      </div>
      <div class="filters" style="margin-bottom:16px" id="batch-chips">${batchChipsHtml(st.status)}</div>
      <div class="card" style="overflow:auto">
        <table class="table">
          <thead><tr><th>名称</th><th>状态</th><th>进度</th><th>规模</th><th>创建时间</th></tr></thead>
          <tbody id="batch-tbody"></tbody>
        </table>
        <div id="batch-empty"></div>
      </div>
      <div id="batch-pager"></div>`;
    icons(el);
    await renderBatches(el);
  }

  function batchChipsHtml(current) {
    const chips = [["", "全部"], ["RUNNING", "执行中"], ["PENDING", "待处理"], ["COMPLETED", "全部成功"], ["PARTIAL", "部分失败"], ["FAILED", "全部失败"]];
    return chips.map(([v, label]) => `<button class="btn sm ${current === v ? "primary" : ""}" data-chip="${v}">${label}</button>`).join("");
  }

  async function renderBatches(el) {
    const st = viewState.batches;
    const params = new URLSearchParams({ page: st.page, page_size: 8 });
    if (st.status) params.set("status", st.status);
    const data = await api(`/api/batches?${params}`);
    $("#batch-chips").innerHTML = batchChipsHtml(st.status);
    $$("[data-chip]", $("#batch-chips")).forEach((b) => (b.onclick = () => { st.status = b.dataset.chip; st.page = 1; renderBatches(el); }));
    $("#batch-tbody").innerHTML = data.items.map((b) => `
      <tr style="cursor:pointer" onclick="location.hash='#/batch/${b.id}'">
        <td><b>${esc(b.name)}</b></td>
        <td>${statusBadge(b.status)}</td>
        <td>${progressBar(b)}</td>
        <td>${b.model_count} 模型 × ${b.prompt_count} 用例</td>
        <td style="color:#71788a">${esc(fmt(b.created_at))}</td>
      </tr>`).join("");
    $("#batch-empty").innerHTML = data.items.length ? "" : `<div class="empty">${icon("inbox")}<div>暂无批次</div></div>`;
    $("#batch-pager").innerHTML = pagerHtml(st.page, data.total, data.page_size);
    wirePager($("#batch-pager"), (p) => { st.page = p; renderBatches(el); });
    icons(el);
  }

  /* ============ 视图：批次详情（结果对比） ============ */
  function viewBatchDetail(el, id) {
    const st = viewState.detail;
    const key = ++viewState.detail.pollKey;
    el.innerHTML = `<div class="empty">加载中…</div>`;
    renderDetail(el, id, st);
    schedulePoll(() => {
      if (key !== viewState.detail.pollKey) { stopPoll(); return; }
      renderDetail(el, id, st, true);
    }, 3000);
  }

  async function renderDetail(el, id, st, silent) {
    const params = new URLSearchParams({ page: st.page, page_size: 10 });
    if (st.status) params.set("status", st.status);
    const b = await api(`/api/batches/${id}?${params}`);
    const running = !TERMINAL.includes(b.status);
    el.innerHTML = `
      <div class="page-head">
        <div style="display:flex;align-items:center;gap:12px">
          <button class="btn sm" onclick="location.hash='#/batches'">${icon("chevron-left")}返回</button>
          <div><h1 class="page-title">${esc(b.name)}</h1>
            <div class="batch-meta">
              <span>${icon("calendar")}${esc(fmt(b.created_at))}</span>
              <span>${icon("cpu")}${b.model_count} 个模型 ${icon("list-checks")}${b.prompt_count} 条用例</span>
              ${b.remark ? `<span>${esc(b.remark)}</span>` : ""}
            </div>
          </div>
        </div>
        ${b.failed_tasks ? `<button class="btn" id="retry-btn">${icon("rotate-ccw")}重试失败（${b.failed_tasks}）</button>` : ""}
      </div>
      <div class="card" style="padding:16px 18px;margin-bottom:16px">
        <div class="detail-progress">
          ${statusBadge(b.status)}
          <div class="progress">
            <div class="ok" style="flex:${b.succeed_tasks}"></div>
            <div class="bad" style="flex:${b.failed_tasks}"></div>
            <div class="runnig" style="flex:${Math.max(b.total_tasks - Math.max(b.completed_tasks, 0), 0)}"></div>
          </div>
          <div class="counts">
            <span class="s-ok"><b>${b.succeed_tasks}</b> 成功</span>
            <span class="s-bad"><b>${b.failed_tasks}</b> 失败</span>
            <span><b>${b.completed_tasks}/${b.total_tasks}</b> 已完成</span>
            ${running ? `<span style="color:#b45309">执行中…</span>` : ""}
          </div>
        </div>
      </div>
      <div class="filters" style="margin-bottom:14px">
        ${[["", "全部"], ["PENDING", "待处理"], ["RUNNING", "执行中"], ["SUCCESS", "成功"], ["FAILED", "失败"]]
          .map(([v, label]) => `<button class="btn sm ${st.status === v ? "primary" : ""}" data-chip="${v}">${label}</button>`).join("")}
        <span class="page-sub">当前筛选用例：<b>${b.matched_total}</b> 个</span>
      </div>
      ${b.prompts.map((g) => caseCardHtml(g)).join("") || `<div class="card empty" style="border:none">${icon("inbox")}<div>该筛选下暂无结果</div></div>`}
      <div id="detail-pager"></div>`;
    $("#detail-pager").innerHTML = pagerHtml(st.page, b.matched_total, 10);
    wirePager($("#detail-pager"), (p) => { st.page = p; renderDetail(el, id, st, true); });
    $$("[data-chip]", el).forEach((x) => (x.onclick = () => { st.status = x.dataset.chip; st.page = 1; renderDetail(el, id, st, true); }));
    const retry = $("#retry-btn");
    if (retry) retry.onclick = async () => {
      busy(retry, true);
      const r = await api(`/api/batches/${id}/retry-failed`, { method: "POST" });
      busy(retry, false);
      toast(`已重新投递 ${r.retried_count} 个失败任务`, "success");
      renderDetail(el, id, st, true);
    };
    $$(".case-toggle", el).forEach((x) => (x.onclick = () => {
      const inp = $("#case-input-" + x.dataset.toggle);
      inp.classList.toggle("collapsed");
      x.textContent = inp.classList.contains("collapsed") ? "展开" : "收起";
    }));
    $$(".copy-btn", el).forEach((x) => (x.onclick = (evt) => {
      const out = document.querySelector(`[data-copy-src="${x.dataset.copy}"]`);
      copyText(out ? out.innerText : "", evt);
    }));
    icons(el);
  }

  function caseCardHtml(group) {
    const collapsed = group.content.length > 120;
    return `
      <div class="card case-card">
        <div class="case-head">
          <div style="flex:1;min-width:0">
            <span class="badge ollama">${esc(group.category)}</span>
            <div class="case-input ${collapsed ? "collapsed" : ""}" id="case-input-${group.prompt_id}">${esc(group.content)}</div>
          </div>
          ${collapsed ? `<button class="case-toggle" data-toggle="${group.prompt_id}">展开</button>` : ""}
        </div>
        <div class="res-grid">
          ${group.results.map((r) => `
            <div class="res-col">
              <div class="res-col-head">
                <div style="display:flex;align-items:center;gap:6px">
                  <span class="model-name">${esc(r.model_name)}</span>
                  ${statusBadge(r.status)}
                </div>
                ${r.status === "SUCCESS" ? `<button class="copy-btn" data-copy="${group.prompt_id}-${r.task_id}" title="复制输出">${icon("copy")}</button>` : ""}
              </div>
              ${r.status === "SUCCESS" ? `<div class="res-output" data-copy-src="${group.prompt_id}-${r.task_id}">${esc(r.output || "")}</div>
                <div class="res-time" style="margin-top:8px">${icon("timer")}${r.response_time_ms}ms · ${r.attempt_count} 次尝试</div>`
              : r.status === "FAILED" ? `<div class="res-error">${esc(r.error || "执行失败")}</div>`
              : r.status === "RUNNING" ? `<div class="res-output" style="color:#b45309">生成中…</div>`
              : `<div class="res-output" style="color:#aab0bf">等待执行</div>`}
            </div>`).join("")}
        </div>
      </div>`;
  }

  /* ============ 视图：新建评测 ============ */
  async function viewCreate(el) {
    const st = viewState.create;
    st.step = 1;
    el.innerHTML = `<div class="empty">加载中…</div>`;
    const [models, prompts] = await Promise.all([
      api("/api/models?page=1&page_size=100"),
      api("/api/prompts?page=1&page_size=100"),
    ]);
    renderCreate(el, st, models.items.filter((m) => m.enabled), prompts);
  }

  function renderCreate(el, st, models, prompts) {
    const totalTasks = st.modelIds.size * st.promptIds.size;
    el.innerHTML = `
      <div class="page-head"><div><h1 class="page-title">新建评测</h1><div class="page-sub">选择模型与用例，系统将按「模型 × 用例」批量执行并对比输出</div></div></div>
      <div class="steps">
        <div class="step ${stepCls(1, st)}" data-step="1"><span class="num">${st.modelIds.size ? icon("check") : "1"}</span>选择模型（已选 ${st.modelIds.size}）</div>
        <div class="step ${stepCls(2, st)}" data-step="2"><span class="num">${st.promptIds.size ? icon("check") : "2"}</span>选择用例（已选 ${st.promptIds.size}）</div>
        <div class="step ${stepCls(3, st)}" data-step="3"><span class="num">3</span>确认并提交</div>
      </div>
      <div id="create-body">
        ${st.step === 1 ? stepModelsHtml(st, models) : ""}
        ${st.step === 2 ? stepPromptsHtml(st, prompts) : ""}
        ${st.step === 3 ? stepConfirmHtml() : ""}
      </div>
      <div class="summary-bar">
        <div class="total">任务总数 <strong>${totalTasks}</strong> 条（${st.modelIds.size} 模型 × ${st.promptIds.size} 用例）</div>
        <div style="display:flex;gap:8px">
          ${st.step > 1 ? `<button class="btn" id="prev-step">${icon("chevron-left")}上一步</button>` : ""}
          ${st.step < 3 ? `<button class="btn primary" id="next-step">下一步 ${icon("chevron-right")}</button>` : ""}
          ${st.step === 3 ? `<button class="btn primary" id="submit-batch">${icon("play")}开始评测</button>` : ""}
        </div>
      </div>`;
    icons(el);
    $("#next-step") && ($("#next-step").onclick = () => {
      if (st.step === 1 && !st.modelIds.size) return toast("请至少选择一个模型", "error");
      if (st.step === 2 && !st.promptIds.size) return toast("请至少选择一个用例", "error");
      st.step++;
      renderCreate(el, st, models, prompts);
    });
    $("#prev-step") && ($("#prev-step").onclick = () => { st.step--; renderCreate(el, st, models, prompts); });
    $("#submit-batch") && ($("#submit-batch").onclick = () => submitBatch());
    if (st.step === 1) bindStepModels(el, st, models, prompts);
    if (st.step === 2) bindStepPrompts(el, st, models, prompts);
  }

  function stepCls(n, st) {
    if (st.step === n) return "active";
    return (n === 1 && st.modelIds.size) || (n === 2 && st.promptIds.size) ? "done" : "";
  }

  function stepModelsHtml(st, models) {
    return `<div class="form-row"><label style="font-size:14px">勾选参与评测的模型</label>
      <div class="pick-grid">
        ${models.map((m) => `
          <div class="pick-item ${st.modelIds.has(m.id) ? "checked" : ""}" data-pick-model="${m.id}">
            <input type="checkbox" ${st.modelIds.has(m.id) ? "checked" : ""} />
            <div style="min-width:0">
              <div class="title">${esc(m.name)}</div>
              <div class="desc">${providerBadge(m.provider)} · ${esc(m.model_name)}</div>
            </div>
          </div>`).join("") || `<div class="empty" style="grid-column:1/-1">请先在「模型管理」中新增并启用模型</div>`}
      </div></div>`;
  }

  function bindStepModels(el, st, models, prompts) {
    $$("[data-pick-model]", el).forEach((item) => (item.onclick = () => {
      const id = Number(item.dataset.pickModel);
      if (st.modelIds.has(id)) st.modelIds.delete(id); else st.modelIds.add(id);
      item.classList.toggle("checked", st.modelIds.has(id));
      item.querySelector("input").checked = st.modelIds.has(id);
      refreshSummary(el, st);
    }));
  }

  function stepPromptsHtml(st, prompts) {
    const hit = prompts.items.filter((p) =>
      p.enabled && (!st.keyword || p.content.includes(st.keyword)) && (!st.category || p.category === st.category));
    return `
      <div class="filters" style="margin-bottom:14px">
        <div class="search-box"><input class="input" id="pk" placeholder="搜索用例内容" value="${esc(st.keyword || "")}" /></div>
        <select class="select" id="pc">
          <option value="">全部分类</option>
          ${Array.from(new Set(prompts.items.map((p) => p.category))).map((c) => `<option value="${esc(c)}" ${st.category === c ? "selected" : ""}>${esc(c)}</option>`).join("")}
        </select>
        <button class="btn sm" id="pick-all">${icon("check-check")}全选当前${prompts.total > 100 ? "（前100条）" : ""}</button>
        <button class="btn sm" id="pick-none">${icon("x-circle")}清空</button>
        <span class="page-sub">已选 <b>${st.promptIds.size}</b> 条</span>
      </div>
      <div class="card" style="overflow:auto;max-height:52vh">
        ${hit.length ? hit.map((p) => `
          <div class="pick-item ${st.promptIds.has(p.id) ? "checked" : ""}" data-pick-prompt="${p.id}" style="border-radius:0;border-left:none;border-right:none;border-top:none">
            <input type="checkbox" ${st.promptIds.has(p.id) ? "checked" : ""} />
            <div style="min-width:0">
              <div style="font-size:12px;color:#aab0bf">#${p.id} · ${esc(p.category)}</div>
              <div class="desc" style="-webkit-line-clamp:3;white-space:pre-wrap">${esc(p.content)}</div>
            </div>
          </div>`).join("") : `<div class="empty">${icon("inbox")}<div>无匹配用例，请先在「测试用例」中导入</div></div>`}
      </div>`;
  }

  function bindStepPrompts(el, st, models, prompts) {
    $("#pk").addEventListener("input", () => { st.keyword = $("#pk").value; refreshStep2(el, st, prompts); });
    $("#pc").onchange = () => { st.category = $("#pc").value; refreshStep2(el, st, prompts); };
    $("#pick-all").onclick = () => {
      prompts.items.filter((p) =>
        p.enabled && (!st.keyword || p.content.includes(st.keyword)) && (!st.category || p.category === st.category)
      ).forEach((p) => st.promptIds.add(p.id));
      refreshStep2(el, st, prompts);
    };
    $("#pick-none").onclick = () => { st.promptIds.clear(); refreshStep2(el, st, prompts); };
    $$("[data-pick-prompt]", el).forEach((item) => (item.onclick = () => {
      const id = Number(item.dataset.pickPrompt);
      if (st.promptIds.has(id)) st.promptIds.delete(id); else st.promptIds.add(id);
      item.classList.toggle("checked", st.promptIds.has(id));
      item.querySelector("input").checked = st.promptIds.has(id);
      refreshSummary(el, st);
    }));
  }

  function refreshStep2(el, st, prompts) {
    /* 重新渲染第二步（保留选中集合） */
    const body = $("#create-body");
    body.innerHTML = stepPromptsHtml(st, prompts);
    icons(el);
    bindStepPrompts(el, st, null, prompts);
    refreshSummary(el, st);
  }

  function refreshSummary(el, st) {
    const strong = $(".summary-bar .total strong", el);
    if (strong) strong.textContent = st.modelIds.size * st.promptIds.size;
    const labels = { 1: `选择模型（已选 ${st.modelIds.size}）`, 2: `选择用例（已选 ${st.promptIds.size}）` };
    [1, 2].forEach((n) => {
      const step = $(`.steps [data-step="${n}"]`, el);
      if (!step) return;
      const done = n === 1 ? !!st.modelIds.size : !!st.promptIds.size;
      step.classList.toggle("done", done);
      step.innerHTML = `<span class="num">${done ? icon("check") : n}</span>${labels[n]}`;
      icons(step);
    });
  }

  function stepConfirmHtml() {
    return `<div class="form-row"><label>批次名称 *</label><input class="input" id="batch-name" placeholder="例如：Qwen 与 Llama 首轮对比" /></div>
      <div class="form-row"><label>备注（可选）</label><input class="input" id="batch-remark" placeholder="本次评测说明" /></div>`;
  }

  async function submitBatch() {
    const name = $("#batch-name") ? $("#batch-name").value.trim() : "";
    if (!name) return toast("请填写批次名称", "error");
    const remark = $("#batch-remark") ? $("#batch-remark").value.trim() || null : null;
    const btn = $("#submit-batch");
    busy(btn, true);
    try {
      const created = await api("/api/batches", {
        method: "POST",
        body: {
          name,
          remark,
          model_ids: Array.from(viewState.create.modelIds),
          prompt_ids: Array.from(viewState.create.promptIds),
        },
      });
      toast("已开始评测、正在批量对比输出", "success");
      location.hash = `#/batch/${created.id}`;
      viewState.create = { step: 1, modelIds: new Set(), promptIds: new Set(), keyword: "", category: "" };
    } catch (e) {
      busy(btn, false);
    }
  }

  /* ============ 启动 ============ */
  window.addEventListener("DOMContentLoaded", () => {
    modal.mask().addEventListener("click", (e) => { if (e.target === modal.mask()) modal.close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.close(); });
    router();
  });
})();