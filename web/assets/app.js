(() => {
  "use strict";

  const OFFICIAL_HOSTS = new Set([
    "modelcontextprotocol.io",
    "docs.langchain.com",
    "docs.python.org",
    "json-schema.org",
    "docs.docker.com",
  ]);

  const INTENT_LABELS = {
    design: "设计 / 历史",
    implementation: "当前实现",
    official: "官方规范",
    comparison: "跨源比较",
    out_of_scope: "超出范围",
  };

  const ROLE_LABELS = {
    current_implementation: "当前实现",
    indexed_implementation: "索引实现候选",
    internal_design: "内部设计",
    internal_history: "历史记录",
    external_normative: "官方规范",
    supporting: "补充证据",
  };

  const WARNING_LABELS = {
    query_normalized_langchian_to_langchain: "已将 Langchian 按常见拼写错误纠正为 LangChain 后检索。",
    model_generation_failed_fallback_used: "大模型调用失败；本次仅保留检索证据，没有生成 AI 回答。",
    model_generation_empty_fallback_used: "大模型返回空内容；本次仅保留检索证据，没有生成 AI 回答。",
    model_generation_invalid_citations_fallback_used: "大模型引用未通过证据校验；本次仅保留检索证据，没有生成 AI 回答。",
  };

  const GENERATION_FAILURE_LABELS = {
    provider_authentication_failed: "DeepSeek 鉴权失败。请更新 DEEPSEEK_API_KEY 并重启服务。",
    provider_timeout: "大模型请求超时。检索结果仍然可用，可稍后重试生成。",
    provider_error: "大模型服务调用失败。检索结果仍然可用，请检查服务配置。",
    invalid_citations: "模型回答的引用未通过校验，因此没有展示该回答。",
    empty_response: "模型没有返回可展示的回答。",
  };

  const state = {
    mode: "answer",
    requestController: null,
    lastRequest: null,
    disclosureTimers: new Map(),
    navLastFocus: null,
  };

  const dom = {
    serviceStatus: document.querySelector("#service-status"),
    serviceStatusText: document.querySelector("#service-status-text"),
    healthButton: document.querySelector("#health-button"),
    metricBuild: document.querySelector("#metric-build"),
    metricFreshness: document.querySelector("#metric-freshness"),
    metricDocuments: document.querySelector("#metric-documents"),
    metricPartitions: document.querySelector("#metric-partitions"),
    metricLive: document.querySelector("#metric-live"),
    metricBranch: document.querySelector("#metric-branch"),
    metricRevision: document.querySelector("#metric-revision"),
    metricAnswerProvider: document.querySelector("#metric-answer-provider"),
    metricAnswerModel: document.querySelector("#metric-answer-model"),
    metricVectorBackend: document.querySelector("#metric-vector-backend"),
    metricVectorDetail: document.querySelector("#metric-vector-detail"),
    sidebar: document.querySelector("#app-sidebar"),
    navToggle: document.querySelector("#nav-toggle"),
    navClose: document.querySelector("#nav-close"),
    navOverlay: document.querySelector("#nav-overlay"),
    navLinks: [...document.querySelectorAll("[data-nav-section]")],
    workspaceActions: [...document.querySelectorAll("[data-workspace-mode]")],
    healthActions: [...document.querySelectorAll("[data-health-action]")],
    popoverLinks: [...document.querySelectorAll(".topbar-popover a[href^='#']")],
    disclosureTriggers: [...document.querySelectorAll("[data-disclosure-target]")],
    modeButtons: [...document.querySelectorAll(".mode-button")],
    form: document.querySelector("#query-form"),
    queryInput: document.querySelector("#query-input"),
    queryCount: document.querySelector("#query-count"),
    queryHint: document.querySelector("#query-hint"),
    topKInput: document.querySelector("#top-k-input"),
    topKDown: document.querySelector("#top-k-down"),
    topKUp: document.querySelector("#top-k-up"),
    topKSummary: document.querySelector("#top-k-summary"),
    tokenSettings: document.querySelector("#token-settings"),
    tokenInput: document.querySelector("#token-input"),
    toggleToken: document.querySelector("#toggle-token"),
    exampleMenu: document.querySelector("#example-menu"),
    formError: document.querySelector("#form-error"),
    submitButton: document.querySelector("#submit-button"),
    submitLabel: document.querySelector("#submit-button .button-label"),
    cancelButton: document.querySelector("#cancel-button"),
    retryButton: document.querySelector("#retry-button"),
    resultPanel: document.querySelector(".result-panel"),
    resultAnnouncer: document.querySelector("#result-announcer"),
    resultHeadingTitle: document.querySelector("#result-heading-title"),
    requestTime: document.querySelector("#request-time"),
    emptyState: document.querySelector("#empty-state"),
    loadingState: document.querySelector("#loading-state"),
    errorState: document.querySelector("#error-state"),
    errorTitle: document.querySelector("#error-title"),
    errorMessage: document.querySelector("#error-message"),
    responseState: document.querySelector("#response-state"),
    responseSummary: document.querySelector("#response-summary"),
    responseAlerts: document.querySelector("#response-alerts"),
    answerBlock: document.querySelector("#answer-block"),
    answerModeLabel: document.querySelector("#answer-mode-label"),
    answerModeCaption: document.querySelector("#answer-mode-caption"),
    answerText: document.querySelector("#answer-text"),
    evidenceLabel: document.querySelector("#evidence-label"),
    evidenceCount: document.querySelector("#evidence-count"),
    evidenceCaption: document.querySelector("#evidence-caption"),
    evidenceDrawer: document.querySelector("#evidence-drawer"),
    evidenceList: document.querySelector("#evidence-list"),
    examples: [...document.querySelectorAll("[data-question]")],
  };

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function disclosurePanel(trigger) {
    const id = trigger.getAttribute("aria-controls");
    return id ? document.getElementById(id) : null;
  }

  function closeDisclosure(trigger, restoreFocus = false) {
    const panel = disclosurePanel(trigger);
    if (!panel) return;
    window.clearTimeout(state.disclosureTimers.get(trigger));
    trigger.dataset.pinned = "false";
    trigger.setAttribute("aria-expanded", "false");
    panel.hidden = true;
    if (restoreFocus) {
      trigger.dataset.suppressFocusOpen = "true";
      trigger.focus();
      window.setTimeout(() => delete trigger.dataset.suppressFocusOpen, 0);
    }
  }

  function closeOtherDisclosures(activeTrigger) {
    dom.disclosureTriggers.forEach((trigger) => {
      if (trigger !== activeTrigger) closeDisclosure(trigger);
    });
  }

  function openDisclosure(trigger, pinned = false) {
    const panel = disclosurePanel(trigger);
    if (!panel) return;
    closeOtherDisclosures(trigger);
    window.clearTimeout(state.disclosureTimers.get(trigger));
    trigger.dataset.pinned = pinned ? "true" : trigger.dataset.pinned || "false";
    trigger.setAttribute("aria-expanded", "true");
    panel.hidden = false;
  }

  function scheduleDisclosureClose(trigger) {
    if (trigger.dataset.pinned === "true") return;
    window.clearTimeout(state.disclosureTimers.get(trigger));
    const timer = window.setTimeout(() => {
      const panel = disclosurePanel(trigger);
      const focusedWithin = panel && panel.contains(document.activeElement);
      if (!trigger.matches(":hover") && !panel?.matches(":hover") && !focusedWithin) {
        closeDisclosure(trigger);
      }
    }, 160);
    state.disclosureTimers.set(trigger, timer);
  }

  function setupDisclosures() {
    const hoverCapable = window.matchMedia("(hover: hover) and (pointer: fine)");
    dom.disclosureTriggers.forEach((trigger) => {
      const panel = disclosurePanel(trigger);
      if (!panel) return;
      trigger.dataset.pinned = "false";
      trigger.addEventListener("click", () => {
        const expanded = trigger.getAttribute("aria-expanded") === "true";
        if (expanded && trigger.dataset.pinned === "true") {
          closeDisclosure(trigger);
        } else {
          openDisclosure(trigger, true);
        }
        if (mobileNavigationActive() && trigger.closest("#app-sidebar")) {
          closeNavigation();
        }
      });
      trigger.addEventListener("focus", () => {
        if (trigger.dataset.suppressFocusOpen !== "true") openDisclosure(trigger);
      });
      panel.addEventListener("focusin", () => openDisclosure(trigger));
      panel.addEventListener("focusout", (event) => {
        if (!panel.contains(event.relatedTarget) && event.relatedTarget !== trigger) {
          scheduleDisclosureClose(trigger);
        }
      });
      if (hoverCapable.matches) {
        trigger.addEventListener("pointerenter", () => openDisclosure(trigger));
        trigger.addEventListener("pointerleave", () => scheduleDisclosureClose(trigger));
        panel.addEventListener("pointerenter", () => {
          window.clearTimeout(state.disclosureTimers.get(trigger));
        });
        panel.addEventListener("pointerleave", () => scheduleDisclosureClose(trigger));
      }
    });

    document.addEventListener("click", (event) => {
      dom.disclosureTriggers.forEach((trigger) => {
        const panel = disclosurePanel(trigger);
        if (!trigger.contains(event.target) && !panel?.contains(event.target)) {
          closeDisclosure(trigger);
        }
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      const openTrigger = dom.disclosureTriggers.find(
        (trigger) => trigger.getAttribute("aria-expanded") === "true",
      );
      if (openTrigger) {
        event.preventDefault();
        closeDisclosure(openTrigger, true);
      }
    });
  }

  function mobileNavigationActive() {
    return window.matchMedia("(max-width: 900px)").matches;
  }

  function syncNavigationAccessibility() {
    const hidden = mobileNavigationActive() && !document.body.classList.contains("nav-open");
    dom.sidebar.toggleAttribute("inert", hidden);
    if (hidden) dom.sidebar.setAttribute("aria-hidden", "true");
    else dom.sidebar.removeAttribute("aria-hidden");
  }

  function openNavigation() {
    if (!mobileNavigationActive()) return;
    state.navLastFocus = document.activeElement;
    document.body.classList.add("nav-open");
    dom.navOverlay.hidden = false;
    dom.navToggle.setAttribute("aria-expanded", "true");
    dom.navToggle.setAttribute("aria-label", "关闭导航");
    syncNavigationAccessibility();
    dom.navClose.focus();
  }

  function closeNavigation({ restoreFocus = false } = {}) {
    const wasOpen = document.body.classList.contains("nav-open");
    const focusWasInside = dom.sidebar.contains(document.activeElement);
    if (wasOpen && mobileNavigationActive() && (restoreFocus || focusWasInside)) {
      const previous = state.navLastFocus;
      const target = previous instanceof HTMLElement
        && previous.isConnected
        && !dom.sidebar.contains(previous)
        ? previous
        : dom.navToggle;
      target.focus();
    }
    document.body.classList.remove("nav-open");
    dom.navOverlay.hidden = true;
    dom.navToggle.setAttribute("aria-expanded", "false");
    dom.navToggle.setAttribute("aria-label", "打开导航");
    syncNavigationAccessibility();
    state.navLastFocus = null;
  }

  function trapNavigationFocus(event) {
    if (event.key !== "Tab" || !document.body.classList.contains("nav-open")) return;
    const focusable = [...dom.sidebar.querySelectorAll("a[href], button:not(:disabled)")]
      .filter((node) => !node.hidden && node.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function activateNavigationLink(link) {
    dom.navLinks.forEach((item) => {
      const active = item === link;
      item.classList.toggle("is-active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
  }

  function focusWorkspace(mode) {
    setMode(mode);
    const workspace = document.querySelector("#query-workspace");
    workspace.scrollIntoView({
      block: "start",
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    });
    window.setTimeout(() => dom.queryInput.focus({ preventScroll: true }), 280);
  }

  function token() {
    return dom.tokenInput.value.trim();
  }

  function redacted(message) {
    const secret = token();
    const text = String(message || "");
    return secret ? text.split(secret).join("[redacted]") : text;
  }

  async function requestJson(path, options = {}) {
    const controller = options.controller || new AbortController();
    const timeout = window.setTimeout(() => {
      if (!controller.signal.aborted) {
        controller.abort(new DOMException("请求超时", "TimeoutError"));
      }
    }, 120_000);
    const headers = { Accept: "application/json" };
    const suppliedToken = token();
    if (suppliedToken) headers.Authorization = `Bearer ${suppliedToken}`;
    if (options.body) headers["Content-Type"] = "application/json";

    try {
      const response = await fetch(path, {
        method: options.method || "GET",
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
        credentials: "same-origin",
        redirect: "error",
      });
      const raw = await response.text();
      let payload = null;
      if (raw) {
        try {
          payload = JSON.parse(raw);
        } catch (_error) {
          throw new ApiError(response.status, "服务返回了无效的 JSON 响应");
        }
      }
      if (!response.ok) {
        const detail = payload && payload.detail;
        const message = Array.isArray(detail)
          ? detail.map((item) => item.msg || "输入不合法").join("；")
          : detail || `请求失败（${response.status}）`;
        throw new ApiError(response.status, redacted(message));
      }
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new ApiError(response.status, "服务响应不是 JSON 对象");
      }
      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setServiceStatus(kind, text) {
    dom.serviceStatus.className = `status-badge status-${kind}`;
    dom.serviceStatusText.textContent = text;
  }

  function shortHash(value, size = 12) {
    const text = String(value || "");
    return text ? text.slice(0, size) : "—";
  }

  function renderHealth(payload) {
    const index = payload.index || {};
    const revision = payload.live_revision || {};
    const fresh = payload.index_fresh === true;
    setServiceStatus(fresh ? "ok" : "warning", fresh ? "索引新鲜" : "索引可能过期");

    dom.metricBuild.textContent = shortHash(index.build_id, 22);
    dom.metricFreshness.textContent = fresh
      ? "与当前 manifest 一致"
      : "静态知识可能落后";
    dom.metricDocuments.textContent = Number(index.document_count || 0).toLocaleString("zh-CN");
    dom.metricPartitions.textContent = `${Number(payload.partition_count || 0)} 个物理分区`;

    const liveChecks = [
      payload.live_code_enabled,
      payload.live_ast_enabled,
      payload.live_git_enabled,
    ];
    const liveCount = liveChecks.filter(Boolean).length;
    dom.metricLive.textContent = liveCount === 3 ? "全部可用" : `${liveCount}/3 可用`;
    dom.metricBranch.textContent = revision.branch || "—";
    const dirty = revision.dirty === true ? " · 含未提交修改" : "";
    dom.metricRevision.textContent = `${shortHash(revision.commit_sha)}${dirty}`;
    dom.metricAnswerProvider.textContent = payload.model_generation_enabled
      ? "DeepSeek 已配置"
      : "确定性";
    dom.metricAnswerModel.textContent = payload.answer_model || "本地证据摘要";
    const vector = index.vector_backend || {};
    const requested = String(vector.requested_mode || vector.configured_backend || "faiss");
    const active = String(vector.active_backend || "faiss");
    dom.metricVectorBackend.textContent = active.toUpperCase();
    if (vector.fallback_used === true) {
      dom.metricVectorDetail.textContent = `请求 ${requested} · 已降级 · ${vector.reason_code || "availability"}`;
    } else {
      dom.metricVectorDetail.textContent = `请求 ${requested} · 未降级`;
    }
  }

  function resetHealthMetrics(message) {
    dom.metricBuild.textContent = "—";
    dom.metricFreshness.textContent = message;
    dom.metricDocuments.textContent = "—";
    dom.metricPartitions.textContent = "— 个物理分区";
    dom.metricLive.textContent = "—";
    dom.metricBranch.textContent = "—";
    dom.metricRevision.textContent = "等待工作区信息";
    dom.metricAnswerProvider.textContent = "—";
    dom.metricAnswerModel.textContent = "等待服务配置";
    dom.metricVectorBackend.textContent = "—";
    dom.metricVectorDetail.textContent = "等待索引后端信息";
  }

  async function checkHealth(options = {}) {
    const focusOnAuth = options.focusOnAuth === true;
    dom.healthButton.disabled = true;
    dom.healthButton.textContent = "连接中…";
    setServiceStatus("pending", "加载服务中");
    try {
      const payload = await requestJson("/health");
      renderHealth(payload);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setServiceStatus("warning", "需要访问令牌");
        resetHealthMetrics("请在查询面板填写令牌");
        dom.tokenSettings.open = true;
        dom.tokenSettings.classList.add("needs-attention");
        if (focusOnAuth) dom.tokenInput.focus();
      } else {
        setServiceStatus("error", "服务不可用");
        resetHealthMetrics(redacted(error.message || "无法连接服务"));
      }
    } finally {
      dom.healthButton.disabled = false;
      dom.healthButton.textContent = "检查服务";
    }
  }

  function setMode(mode) {
    state.mode = mode;
    dom.modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    dom.submitLabel.textContent = mode === "answer" ? "检索并生成回答" : "仅检索证据";
    dom.queryHint.textContent =
      mode === "answer"
        ? "会自动检索并生成回答，无需先点“仅检索证据”"
        : "只返回证据列表，不生成回答";
    dom.resultHeadingTitle.textContent = mode === "answer" ? "回答结果" : "检索结果";
  }

  function clampTopK(value) {
    const parsed = Number.parseInt(String(value), 10);
    const valid = Number.isFinite(parsed) ? parsed : 5;
    return Math.min(20, Math.max(1, valid));
  }

  function validateQuery() {
    const query = dom.queryInput.value.trim();
    const topK = clampTopK(dom.topKInput.value);
    dom.topKInput.value = String(topK);
    dom.topKSummary.textContent = String(topK);
    if (!query) return { error: "请输入一个工程知识问题。" };
    if (query.length > 1000) return { error: "问题不能超过 1000 个字符。" };
    return { query, topK };
  }

  function showState(name) {
    dom.emptyState.hidden = name !== "empty";
    dom.loadingState.hidden = name !== "loading";
    dom.errorState.hidden = name !== "error";
    dom.responseState.hidden = name !== "response";
    dom.resultPanel.setAttribute("aria-busy", String(name === "loading"));
  }

  function setBusy(busy) {
    dom.submitButton.disabled = busy;
    dom.cancelButton.hidden = !busy;
    dom.modeButtons.forEach((button) => {
      button.disabled = busy;
    });
  }

  function errorCopy(error) {
    if (error instanceof ApiError) {
      if (error.status === 401) {
        return ["访问令牌无效", "请展开“本地访问令牌”，填写与服务端一致的令牌后重试。"];
      }
      if (error.status === 422) {
        return ["查询参数不合法", redacted(error.message)];
      }
      if (error.status === 503) {
        return ["知识服务尚未就绪", "请确认索引和 manifest 已存在；首次加载模型时可稍后重试。"];
      }
      return [`服务返回 ${error.status}`, redacted(error.message)];
    }
    if ((error && error.name === "TimeoutError") || error === "timeout") {
      return ["请求超时", "服务在 120 秒内未返回结果，请稍后重试。"];
    }
    if ((error && error.name === "AbortError") || error === "user") {
      return ["请求已取消", "没有产生新的检索结果。"];
    }
    return ["无法连接知识服务", redacted(error && error.message ? error.message : "请确认服务已启动。")];
  }

  function showError(error) {
    const [title, message] = errorCopy(error);
    dom.errorTitle.textContent = title;
    dom.errorMessage.textContent = message;
    showState("error");
    dom.resultAnnouncer.textContent = `${title}。${message}`;
    if (error instanceof ApiError && error.status === 401) {
      dom.tokenSettings.open = true;
      dom.tokenSettings.classList.add("needs-attention");
    }
  }

  function summaryItem(label, value, stateClass = "") {
    const card = element("div", `summary-item ${stateClass}`.trim());
    card.append(element("span", "", label), element("strong", "", value));
    return card;
  }

  function addAlert(kind, text) {
    dom.responseAlerts.append(element("div", `alert alert-${kind}`, text));
  }

  function safeOfficialUrl(value) {
    try {
      const parsed = new URL(String(value));
      const host = parsed.hostname.toLowerCase();
      const allowed = [...OFFICIAL_HOSTS].some(
        (candidate) => host === candidate || host.endsWith(`.${candidate}`),
      );
      return parsed.protocol === "https:" && allowed ? parsed.href : null;
    } catch (_error) {
      return null;
    }
  }

  function roleLabel(role) {
    return ROLE_LABELS[role] || role || "未分类证据";
  }

  function metadataText(item) {
    const values = [];
    if (item.symbol) values.push(`symbol: ${item.symbol}`);
    if (item.line_start) {
      const lines = item.line_end && item.line_end !== item.line_start
        ? `${item.line_start}-${item.line_end}`
        : String(item.line_start);
      values.push(`lines: ${lines}`);
    }
    if (item.retriever) values.push(`retriever: ${item.retriever}`);
    if (typeof item.score === "number") values.push(`score: ${item.score.toFixed(4)}`);
    if (item.authority) values.push(`authority: ${item.authority}`);
    return values;
  }

  function sourceNode(source) {
    const safeUrl = safeOfficialUrl(source);
    if (!safeUrl) return element("span", "evidence-source", source || "unknown source");
    const link = element("a", "evidence-source-link", source);
    link.href = safeUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  function renderEvidence(item, index) {
    const role = item.evidence_role || "supporting";
    const card = document.createElement("details");
    card.className = "evidence-card";
    card.open = index < 1;

    const summary = document.createElement("summary");
    const title = element("div", "evidence-title");
    const top = element("div", "evidence-title-top");
    top.append(element("span", `role-chip role-${role}`, roleLabel(role)));
    if (item.live_verified === true) top.append(element("span", "live-chip", "LIVE VERIFIED"));
    title.append(top, sourceNode(item.citation || item.source || "unknown source"));

    const meta = element("div", "evidence-meta");
    metadataText(item).forEach((value) => meta.append(element("span", "", value)));
    title.append(meta);
    const expand = element("span", "expand-symbol", "+");
    expand.setAttribute("aria-hidden", "true");
    summary.append(title, expand);
    card.append(summary);

    const rawContent = String(item.content || "").trim();
    const content = rawContent.length > 12_000
      ? `${rawContent.slice(0, 12_000)}\n\n[页面展示已截断]`
      : rawContent || "该结果没有可展示的文本摘录。";
    card.append(element("pre", "evidence-content", content));
    return card;
  }

  function validCitationId(value) {
    const id = String(value || "");
    return /^E[1-9]\d*$/.test(id) ? id : null;
  }

  function citationTargetId(citationId) {
    return `citation-${citationId}`;
  }

  function renderCitation(citation, index, usedIds) {
    const card = document.createElement("details");
    card.className = "evidence-card citation-only";
    card.open = index === 0;
    const citationId = validCitationId(citation.citation_id);
    if (citationId && !usedIds.has(citationId)) {
      usedIds.add(citationId);
      card.id = citationTargetId(citationId);
    } else {
      card.id = `citation-card-${index + 1}`;
    }

    const summary = document.createElement("summary");
    const title = element("div", "evidence-title");
    const top = element("div", "evidence-title-top");
    const role = citation.evidence_role || "supporting";
    if (citationId) top.append(element("span", "citation-id-chip", citationId));
    top.append(element("span", `role-chip role-${role}`, roleLabel(role)));
    if (citation.live_verified === true) top.append(element("span", "live-chip", "LIVE VERIFIED"));
    title.append(top, sourceNode(citation.source || "unknown source"));
    const meta = element("div", "evidence-meta");
    const fields = metadataText(citation);
    if (citation.revision) fields.push(`revision: ${shortHash(citation.revision)}`);
    fields.forEach((value) => meta.append(element("span", "", value)));
    title.append(meta);
    const expand = element("span", "expand-symbol", "+");
    expand.setAttribute("aria-hidden", "true");
    summary.append(title, expand);
    card.append(summary);

    const details = element("div", "citation-details");
    details.append(
      element("span", "", `证据角色：${roleLabel(role)}`),
      element("span", "", `来源类型：${citation.authority || citation.corpus || "未标注"}`),
    );
    if (citation.branch) details.append(element("span", "", `分支：${citation.branch}`));
    if (citation.source_version) {
      details.append(element("span", "", `来源版本：${citation.source_version}`));
    }
    card.append(details);
    const rawContent = String(citation.content || "").trim();
    if (rawContent) {
      const content = rawContent.length > 12_000
        ? `${rawContent.slice(0, 12_000)}\n\n[页面展示已截断]`
        : rawContent;
      card.append(element("pre", "evidence-content", content));
    }
    return card;
  }

  function focusCitation(citationId) {
    const target = document.getElementById(citationTargetId(citationId));
    if (!target) return;
    dom.evidenceDrawer.open = true;
    if (target instanceof HTMLDetailsElement) target.open = true;
    document.querySelectorAll(".citation-target").forEach((node) => {
      node.classList.remove("citation-target");
    });
    target.classList.add("citation-target");
    target.scrollIntoView({
      block: "center",
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    });
    const focusTarget = target.querySelector("summary") || target;
    focusTarget.focus({ preventScroll: true });
    window.setTimeout(() => target.classList.remove("citation-target"), 1800);
  }

  function appendAnswerInline(parent, value, allowedCitationIds) {
    const text = String(value || "").replace(
      /(\[E[1-9]\d*\])(?:[ \t]*\1)+/g,
      "$1",
    );
    const tokenPattern = /(\[E[1-9]\d*\]|\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
    let cursor = 0;
    let match = tokenPattern.exec(text);
    while (match) {
      if (match.index > cursor) parent.append(document.createTextNode(text.slice(cursor, match.index)));
      const tokenValue = match[0];
      if (tokenValue.startsWith("[E")) {
        const citationId = tokenValue.slice(1, -1);
        if (allowedCitationIds.has(citationId)) {
          const link = element("a", "citation-ref", citationId);
          link.href = `#${citationTargetId(citationId)}`;
          link.setAttribute("aria-label", `查看回答引用 ${citationId}`);
          link.addEventListener("click", (event) => {
            event.preventDefault();
            focusCitation(citationId);
          });
          parent.append(link);
        } else {
          parent.append(document.createTextNode(tokenValue));
        }
      } else if (tokenValue.startsWith("**")) {
        parent.append(element("strong", "", tokenValue.slice(2, -2)));
      } else {
        parent.append(element("code", "", tokenValue.slice(1, -1)));
      }
      cursor = tokenPattern.lastIndex;
      match = tokenPattern.exec(text);
    }
    if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
  }

  function renderGroundedAnswer(rawAnswer, citations) {
    dom.answerText.replaceChildren();
    const answer = String(rawAnswer || "没有生成回答。");
    const clipped = answer.length > 6_000
      ? `${answer.slice(0, 6_000)}\n\n[页面展示已截断，请查看下方引用]`
      : answer;
    const allowedCitationIds = new Set(
      (citations || []).map((item) => validCitationId(item.citation_id)).filter(Boolean),
    );
    let currentList = null;
    let currentListType = null;

    clipped.split(/\r?\n/).forEach((line) => {
      const heading = line.match(/^(#{2,4})\s+(.+)$/);
      const bullet = line.match(/^[-*]\s+(.+)$/);
      const numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (!line.trim()) {
        currentList = null;
        currentListType = null;
        return;
      }
      if (heading) {
        currentList = null;
        currentListType = null;
        const node = document.createElement(heading[1].length === 2 ? "h3" : "h4");
        appendAnswerInline(node, heading[2], allowedCitationIds);
        dom.answerText.append(node);
        return;
      }
      if (bullet || numbered) {
        const listType = numbered ? "ol" : "ul";
        if (!currentList || currentListType !== listType) {
          currentList = document.createElement(listType);
          currentListType = listType;
          dom.answerText.append(currentList);
        }
        const item = document.createElement("li");
        appendAnswerInline(item, (bullet || numbered)[1], allowedCitationIds);
        currentList.append(item);
        return;
      }
      currentList = null;
      currentListType = null;
      const paragraph = document.createElement("p");
      appendAnswerInline(paragraph, line, allowedCitationIds);
      dom.answerText.append(paragraph);
    });
  }

  function generationDetails(payload) {
    const generation = payload.generation && typeof payload.generation === "object"
      ? payload.generation
      : {};
    return {
      mode: generation.mode || generation.type || generation.status || payload.generation_mode || "evidence_only",
      provider: generation.provider || payload.generation_provider || "deterministic",
      model: generation.model || payload.generation_model || null,
      attempted: generation.attempted === true,
      succeeded: generation.succeeded === true,
      failureCode: generation.failure_code || null,
      retrievedCount: Number.isInteger(generation.retrieved_count) ? generation.retrieved_count : null,
      contextCount: Number.isInteger(generation.context_count) ? generation.context_count : null,
    };
  }

  function arrayField(payload, preferred, legacy) {
    return Array.isArray(payload[preferred]) ? payload[preferred] : (payload[legacy] || []);
  }

  function renderGenerationStatus(title, caption, message) {
    dom.answerModeLabel.textContent = title;
    dom.answerModeCaption.textContent = caption;
    dom.answerText.replaceChildren(element("p", "generation-status", message));
  }

  function renderResponse(payload, mode, elapsedMs) {
    const isRetrieve = mode === "retrieve";
    const sufficient = isRetrieve ? payload.sufficient_evidence === true : payload.refused !== true;
    const intent = INTENT_LABELS[payload.intent] || payload.intent || "未知";
    const generation = generationDetails(payload);
    const generationLabel = generation.mode === "model"
      ? (generation.provider === "deepseek" ? "DeepSeek" : (generation.provider || "模型"))
      : generation.mode === "deterministic_fallback" || generation.mode === "fallback"
        ? "生成失败"
        : generation.mode === "refusal"
          ? "拒答"
          : isRetrieve
            ? "未生成"
            : "仅证据";

    dom.responseSummary.replaceChildren(
      summaryItem("路由", intent),
      summaryItem(
        "证据状态",
        sufficient ? "充分" : isRetrieve ? "不足" : "已拒答",
        sufficient ? "summary-good" : "summary-bad",
      ),
      summaryItem(
        "生成方式",
        isRetrieve ? "未生成（仅检索）" : generationLabel,
      ),
      summaryItem("请求耗时", `${elapsedMs.toFixed(0)} ms`),
    );
    dom.requestTime.textContent = `${elapsedMs.toFixed(0)} ms`;

    dom.responseAlerts.replaceChildren();
    if (payload.refusal_reason) addAlert("error", `拒答原因：${payload.refusal_reason}`);
    if (isRetrieve && !sufficient && (payload.results || []).length) {
      addAlert("warning", "以下内容仅是候选线索，不能作为该问题的充分证明。 ");
    }
    (payload.warnings || []).forEach((warning) => {
      if (generation.failureCode && String(warning).startsWith("model_generation_")) return;
      addAlert("warning", WARNING_LABELS[warning] || warning);
    });
    if (!isRetrieve && generation.failureCode && GENERATION_FAILURE_LABELS[generation.failureCode]) {
      addAlert(
        generation.failureCode === "provider_authentication_failed" ? "error" : "warning",
        GENERATION_FAILURE_LABELS[generation.failureCode],
      );
    }
    if (isRetrieve && payload.live_revision && payload.live_revision.dirty === true) {
      const revision = payload.live_revision;
      addAlert(
        "info",
        `实时结果来自 ${revision.branch || "当前分支"}，commit ${shortHash(revision.commit_sha)}；工作区包含未提交修改。`,
      );
    }

    const retrievedEvidence = arrayField(payload, "retrieved_evidence", "results");
    const generationContext = Array.isArray(payload.generation_context)
      ? payload.generation_context
      : retrievedEvidence;
    const answerCitations = arrayField(payload, "answer_citations", "citations");
    const items = isRetrieve
      ? retrievedEvidence
      : (retrievedEvidence.length ? retrievedEvidence : answerCitations);
    dom.answerBlock.hidden = isRetrieve;
    if (isRetrieve) {
      dom.answerText.replaceChildren();
    } else {
      if (generation.mode === "model") {
        dom.answerModeLabel.textContent = "大模型综合回答";
        dom.answerModeCaption.textContent = `召回 ${retrievedEvidence.length} 条，筛选 ${generationContext.length} 条上下文，由 ${generationLabel} 综合`;
        renderGroundedAnswer(payload.answer, answerCitations);
      } else if (generation.mode === "deterministic_fallback" || generation.mode === "fallback") {
        renderGenerationStatus(
          "大模型生成暂不可用",
          "已保留本次 Top-K 检索结果",
          "系统没有把原始片段拼接成答案。请展开下方证据查看检索结果，或检查模型配置后重试。",
        );
      } else if (generation.mode === "refusal" || payload.refused === true) {
        renderGenerationStatus(
          "证据不足，已拒答",
          "没有使用无关证据强行生成",
          "当前知识库没有找到足以支持答案的证据。请换一种问法，或补充相应知识来源。",
        );
      } else {
        renderGenerationStatus(
          "仅完成证据检索",
          "本次未调用大模型",
          "Top-K 相关片段已经返回。请展开下方证据查看原文；如需自然语言综合回答，请启用模型生成。",
        );
      }
    }
    dom.evidenceCount.textContent = String(items.length);
    dom.evidenceLabel.textContent = "检索到的 Top-K 证据";
    dom.evidenceCaption.textContent = isRetrieve
      ? "展开查看按综合排名返回的原始片段"
      : generation.mode === "model"
        ? `展开查看 ${retrievedEvidence.length} 条召回结果；其中 ${generationContext.length} 条送入模型`
        : "展开查看本次召回的原始片段与引用来源";
    dom.evidenceDrawer.open = false;
    dom.evidenceList.replaceChildren();
    if (!items.length) {
      dom.evidenceList.append(
        element("div", "no-evidence", "没有可展示的证据。请查看拒答原因或修改问题。"),
      );
    } else {
      const usedCitationIds = new Set();
      items.forEach((item, index) => {
        dom.evidenceList.append(
          isRetrieve || !item.citation_id
            ? renderEvidence(item, index)
            : renderCitation(item, index, usedCitationIds),
        );
      });
    }
    showState("response");
    dom.resultAnnouncer.textContent = isRetrieve
      ? `检索完成，共 ${items.length} 条证据。`
      : payload.refused === true
        ? "回答已安全拒绝，请查看拒答原因。"
        : `回答完成，生成方式为 ${generationLabel}，包含 ${items.length} 条检索证据。`;
    if (window.matchMedia("(max-width: 900px)").matches) {
      dom.resultHeadingTitle.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      });
    }
  }

  async function submitQuery(request = null) {
    dom.formError.hidden = true;
    const validated = request || validateQuery();
    if (validated.error) {
      dom.formError.textContent = validated.error;
      dom.formError.hidden = false;
      dom.queryInput.focus();
      return;
    }

    const activeRequest = {
      query: validated.query,
      topK: validated.topK,
      mode: validated.mode || state.mode,
    };
    state.lastRequest = activeRequest;
    const controller = new AbortController();
    state.requestController = controller;
    setBusy(true);
    showState("loading");
    dom.requestTime.textContent = "处理中";
    const started = performance.now();

    try {
      const payload = await requestJson(`/${activeRequest.mode}`, {
        method: "POST",
        body: { query: activeRequest.query, top_k: activeRequest.topK },
        controller,
      });
      if (activeRequest.mode === "retrieve" && payload.schema_version !== "engineering-retrieval/v1") {
        throw new ApiError(502, "检索响应版本不受支持");
      }
      if (activeRequest.mode === "answer" && payload.schema_version !== "engineering-answer/v2") {
        throw new ApiError(502, "回答响应版本不受支持");
      }
      renderResponse(payload, activeRequest.mode, performance.now() - started);
      setServiceStatus("ok", "服务在线");
    } catch (error) {
      showError(error);
    } finally {
      if (state.requestController === controller) state.requestController = null;
      setBusy(false);
    }
  }

  dom.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  dom.queryInput.addEventListener("input", () => {
    dom.queryCount.textContent = String(dom.queryInput.value.length);
    dom.formError.hidden = true;
  });

  dom.queryInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      dom.form.requestSubmit();
    }
  });

  dom.examples.forEach((button) => {
    button.addEventListener("click", () => {
      dom.queryInput.value = button.dataset.question || "";
      dom.queryCount.textContent = String(dom.queryInput.value.length);
      dom.exampleMenu.open = false;
      dom.queryInput.focus();
    });
  });

  dom.topKDown.addEventListener("click", () => {
    const value = clampTopK(clampTopK(dom.topKInput.value) - 1);
    dom.topKInput.value = String(value);
    dom.topKSummary.textContent = String(value);
  });
  dom.topKUp.addEventListener("click", () => {
    const value = clampTopK(clampTopK(dom.topKInput.value) + 1);
    dom.topKInput.value = String(value);
    dom.topKSummary.textContent = String(value);
  });
  dom.topKInput.addEventListener("change", () => {
    const value = clampTopK(dom.topKInput.value);
    dom.topKInput.value = String(value);
    dom.topKSummary.textContent = String(value);
  });

  dom.toggleToken.addEventListener("click", () => {
    const show = dom.tokenInput.type === "password";
    dom.tokenInput.type = show ? "text" : "password";
    dom.toggleToken.textContent = show ? "隐藏" : "显示";
    dom.toggleToken.setAttribute("aria-pressed", String(show));
    dom.toggleToken.setAttribute("aria-label", show ? "隐藏本地访问令牌" : "显示本地访问令牌");
  });

  dom.tokenInput.addEventListener("input", () => {
    dom.tokenSettings.classList.remove("needs-attention");
  });

  dom.tokenInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      checkHealth({ focusOnAuth: false });
    }
  });

  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuery();
  });

  dom.cancelButton.addEventListener("click", () => {
    if (state.requestController && !state.requestController.signal.aborted) {
      state.requestController.abort(new DOMException("请求已取消", "AbortError"));
    }
  });

  dom.retryButton.addEventListener("click", () => {
    if (state.lastRequest) {
      setMode(state.lastRequest.mode);
      submitQuery(state.lastRequest);
    }
  });

  dom.healthButton.addEventListener("click", () => checkHealth({ focusOnAuth: true }));

  dom.navToggle.addEventListener("click", () => {
    if (document.body.classList.contains("nav-open")) {
      closeNavigation({ restoreFocus: true });
    } else {
      openNavigation();
    }
  });
  dom.navClose.addEventListener("click", () => closeNavigation({ restoreFocus: true }));
  dom.navOverlay.addEventListener("click", () => closeNavigation({ restoreFocus: true }));

  dom.navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      activateNavigationLink(link);
      closeNavigation();
    });
  });

  dom.workspaceActions.forEach((button) => {
    button.addEventListener("click", () => focusWorkspace(button.dataset.workspaceMode));
  });

  dom.healthActions.forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await checkHealth({ focusOnAuth: false });
        document.querySelector("#knowledge-overview").scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        });
      } finally {
        button.disabled = false;
      }
    });
  });

  dom.popoverLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const panel = link.closest(".topbar-popover");
      const trigger = dom.disclosureTriggers.find(
        (item) => item.getAttribute("aria-controls") === panel?.id,
      );
      if (trigger) closeDisclosure(trigger);
      closeNavigation();
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("nav-open")) {
      event.preventDefault();
      closeNavigation({ restoreFocus: true });
      return;
    }
    trapNavigationFocus(event);
  });

  window.addEventListener("resize", () => {
    if (!mobileNavigationActive()) closeNavigation();
    else syncNavigationAccessibility();
  });

  setupDisclosures();
  syncNavigationAccessibility();
  dom.toggleToken.setAttribute("aria-pressed", "false");
  dom.toggleToken.setAttribute("aria-label", "显示本地访问令牌");
  setMode("answer");
  showState("empty");
  checkHealth({ focusOnAuth: false });
})();
