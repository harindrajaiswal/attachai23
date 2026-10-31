/**
 * ORBIT AI — ADMIN DASHBOARD CONTROLLER (dashboard.js)
 *
 * Connects every view, form, modal, filter, and button in /admin to real
 * FastAPI backend endpoints (/api/admin/*).
 * Enforces CSRF header protection and never fakes success on backend failure.
 */
(function () {
  "use strict";

  var csrfToken = document.body.getAttribute("data-csrf-token") || "";
  var publicBaseUrl = document.body.getAttribute("data-public-base-url") || window.location.origin;

  var appState = {
    overview: null,
    tenants: [],
    demoPublicKeys: {},
    selectedTenantId: "",
  };

  function escapeHtml(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function apiRequest(path, options) {
    var opts = options || {};
    var headers = Object.assign({}, opts.headers || {});
    if (csrfToken) {
      headers["x-csrf-token"] = csrfToken;
    }
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
    }
    opts.headers = headers;
    opts.credentials = "same-origin";

    var res = await fetch(path, opts);
    if (res.status === 401 && !path.includes("/api/admin/login")) {
      window.location.href = "/admin/login";
      throw new Error("Session expired. Redirecting to login.");
    }
    var contentType = res.headers.get("content-type") || "";
    var data = contentType.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      var errMsg = (data && data.error) || (data && data.detail) || "Operation failed (HTTP " + res.status + ")";
      throw new Error(errMsg);
    }
    return data;
  }

  // --- 1. LOGIN PAGE ---
  var loginForm = document.getElementById("admin-login-form");
  if (loginForm) {
    loginForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var keyInput = document.getElementById("admin-api-key-input");
      var errBox = document.getElementById("login-error-box");
      errBox.classList.add("hidden");
      try {
        await apiRequest("/api/admin/login", {
          method: "POST",
          json: { admin_api_key: keyInput.value.trim() },
        });
        window.location.href = "/admin";
      } catch (err) {
        errBox.textContent = err.message || "Authentication failed.";
        errBox.classList.remove("hidden");
      }
    });
    return;
  }

  // --- 2. DASHBOARD NAVIGATION ---
  var navButtons = document.querySelectorAll(".nav-item[data-target]");
  var views = document.querySelectorAll(".workspace-view");
  var sectionTitleEl = document.getElementById("current-section-title");

  function switchView(targetId) {
    views.forEach(function (v) {
      v.classList.toggle("hidden", v.id !== targetId);
    });
    navButtons.forEach(function (btn) {
      var isMatch = btn.getAttribute("data-target") === targetId;
      btn.classList.toggle("active", isMatch);
      if (isMatch && sectionTitleEl) {
        sectionTitleEl.textContent = btn.textContent.trim();
      }
    });

    if (targetId === "view-knowledge") loadKnowledgeFiles();
    if (targetId === "view-conversations") loadConversations();
    if (targetId === "view-leads") loadLeads();
    if (targetId === "view-prompts") loadPromptsForTenant();
    if (targetId === "view-widget") populateWidgetForm();
    if (targetId === "view-analytics") loadAnalytics();
    if (targetId === "view-settings") loadSettingsAndNotifications();
  }

  navButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      switchView(btn.getAttribute("data-target"));
    });
  });

  var logoutBtn = document.getElementById("btn-admin-logout");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async function () {
      await apiRequest("/api/admin/logout", { method: "POST" });
      window.location.href = "/admin/login";
    });
  }

  // --- 3. LOAD OVERVIEW & TENANT SELECTORS ---
  async function loadOverview() {
    var data = await apiRequest("/api/admin/overview");
    appState.overview = data;
    appState.tenants = data.tenants || [];
    appState.demoPublicKeys = data.demo_public_keys || {};
    if (data.public_base_url) {
      publicBaseUrl = data.public_base_url;
    }
    if (!appState.selectedTenantId && appState.tenants.length > 0) {
      appState.selectedTenantId = appState.tenants[0].id;
    }

    var s = data.summary || {};
    document.getElementById("kpi-tenants").textContent =
      (s.active_tenants || 0) + " / " + (s.total_tenants || 0);
    document.getElementById("kpi-conversations").textContent = String(s.total_conversations || 0);
    document.getElementById("kpi-messages-sub").textContent =
      (s.total_messages || 0) + " messages persisted";
    document.getElementById("kpi-leads").textContent = String(s.total_leads || 0);
    document.getElementById("kpi-chunks").textContent = String(s.total_knowledge_chunks || 0);
    document.getElementById("kpi-files-sub").textContent =
      (s.ready_knowledge_files || 0) + " ready documents (128d)";

    // Render provider summary
    var provList = document.getElementById("overview-providers-list");
    if (provList && data.diagnostics) {
      var d = data.diagnostics;
      var items = [
        { name: "Database", info: d.database },
        { name: "Vector Search (pgvector 128d)", info: d.vector },
        { name: "Private Storage Bucket", info: d.storage },
        { name: "Embeddings (128d)", info: d.embeddings },
        { name: "Chat LLM", info: d.llm },
        { name: "Notification Outbox", info: d.notifications },
      ];
      provList.innerHTML = items
        .map(function (it) {
          var ok = it.info && it.info.ok;
          var label = (it.info && (it.info.provider || it.info.provider_type || it.info.model)) || "";
          return (
            '<div class="list-row">' +
            "<span><strong>" + escapeHtml(it.name) + "</strong> · <span class='meta-inline'>" + escapeHtml(label) + "</span></span>" +
            '<span class="' + (ok ? "status-text-nominal" : "status-text-error") + '">' +
            (ok ? "Ready" : "Failed") +
            "</span></div>"
          );
        })
        .join("");
    }

    // Render recent audit logs
    var auditList = document.getElementById("overview-audit-list");
    if (auditList) {
      var audits = data.recent_audit_logs || [];
      if (!audits.length) {
        auditList.innerHTML = '<div class="meta-inline">No audit events recorded yet.</div>';
      } else {
        auditList.innerHTML = audits
          .slice(0, 6)
          .map(function (a) {
            return (
              '<div class="list-row">' +
              "<span><strong>" + escapeHtml(a.action) + "</strong> · <span class='meta-inline'>by " + escapeHtml(a.actor) + "</span></span>" +
              '<span class="meta-inline font-mono">' + escapeHtml(String(a.created_at || "").slice(0, 19).replace("T", " ")) + "</span>" +
              "</div>"
            );
          })
          .join("");
      }
    }

    syncTenantSelects();
    renderTenantsTable();
  }

  function syncTenantSelects() {
    var filterSelectIds = [
      "knowledge-tenant-select",
      "conv-tenant-select",
      "leads-tenant-select",
      "analytics-tenant-select",
    ];
    filterSelectIds.forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      var current = el.value;
      el.innerHTML =
        '<option value="">All Tenants</option>' +
        appState.tenants
          .map(function (t) {
            return '<option value="' + escapeHtml(t.id) + '">' + escapeHtml(t.business_name) + " (" + escapeHtml(t.slug) + ")</option>";
          })
          .join("");
      el.value = current;
    });

    var requiredSelectIds = ["upload-tenant-id", "prompt-tenant-select", "widget-tenant-select"];
    requiredSelectIds.forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      var current = el.value || appState.selectedTenantId;
      el.innerHTML = appState.tenants
        .map(function (t) {
          return '<option value="' + escapeHtml(t.id) + '">' + escapeHtml(t.business_name) + " (" + escapeHtml(t.slug) + ")</option>";
        })
        .join("");
      if (current && appState.tenants.some(function (t) { return t.id === current; })) {
        el.value = current;
      }
    });
  }

  // --- 4. CLIENTS / TENANTS TABLE & ACTIONS ---
  function showRawApiKey(rawKey) {
    var banner = document.getElementById("raw-key-banner");
    var codeEl = document.getElementById("raw-key-value");
    if (banner && codeEl && rawKey) {
      codeEl.textContent = rawKey;
      banner.classList.remove("hidden");
    }
  }

  var dismissKeyBtn = document.getElementById("btn-dismiss-key-banner");
  if (dismissKeyBtn) {
    dismissKeyBtn.addEventListener("click", function () {
      document.getElementById("raw-key-banner").classList.add("hidden");
    });
  }

  var copyKeyBtn = document.getElementById("btn-copy-raw-key");
  if (copyKeyBtn) {
    copyKeyBtn.addEventListener("click", function () {
      var txt = document.getElementById("raw-key-value").textContent;
      if (navigator.clipboard && txt) navigator.clipboard.writeText(txt);
      copyKeyBtn.textContent = "Copied!";
      setTimeout(function () { copyKeyBtn.textContent = "Copy Key"; }, 1500);
    });
  }

  function renderTenantsTable() {
    var tbody = document.getElementById("tenants-table-body");
    if (!tbody) return;
    if (!appState.tenants.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="meta-inline">No tenants created yet. Click "Create Tenant" to add one.</td></tr>';
      return;
    }

    tbody.innerHTML = appState.tenants
      .map(function (t) {
        var domains = (t.domains || []).join(", ") || "All origins";
        var statusClass = t.enabled ? "status-text-nominal" : "status-text-error";
        var statusLabel = t.enabled ? "Active" : "Disabled";
        return (
          "<tr>" +
          "<td><strong>" + escapeHtml(t.business_name) + "</strong><div class='meta-inline font-mono'>" + escapeHtml(t.slug) + " · " + escapeHtml(t.id.slice(0, 8)) + "</div></td>" +
          "<td>" + escapeHtml(t.assistant_name) + "<div class='meta-inline'>" + escapeHtml(t.company_name) + "</div></td>" +
          "<td class='font-mono meta-inline'>" + escapeHtml(domains) + "</td>" +
          '<td><span class="' + statusClass + '">' + statusLabel + "</span></td>" +
          '<td class="num-col">top_k=' + escapeHtml(t.top_k) + " · min=" + escapeHtml(t.min_similarity) + "</td>" +
          '<td class="actions-col">' +
          '<div class="action-row" style="justify-content:flex-end;">' +
          '<button type="button" class="btn btn-secondary btn-sm" data-action="edit-tenant" data-id="' + escapeHtml(t.id) + '">Edit</button>' +
          '<button type="button" class="btn btn-secondary btn-sm" data-action="rotate-key" data-id="' + escapeHtml(t.id) + '">Rotate Key</button>' +
          '<button type="button" class="btn btn-secondary btn-sm" data-action="toggle-tenant" data-id="' + escapeHtml(t.id) + '" data-enabled="' + (t.enabled ? "1" : "0") + '">' + (t.enabled ? "Disable" : "Enable") + "</button>" +
          '<button type="button" class="btn btn-secondary btn-sm" data-action="duplicate-tenant" data-id="' + escapeHtml(t.id) + '">Duplicate</button>' +
          '<button type="button" class="btn btn-danger btn-sm" data-action="delete-tenant" data-id="' + escapeHtml(t.id) + '">Delete</button>' +
          "</div></td></tr>"
        );
      })
      .join("");
  }

  var tenantsTableBody = document.getElementById("tenants-table-body");
  if (tenantsTableBody) {
    tenantsTableBody.addEventListener("click", async function (ev) {
      var btn = ev.target.closest("button[data-action]");
      if (!btn) return;
      var action = btn.getAttribute("data-action");
      var tenantId = btn.getAttribute("data-id");

      if (action === "edit-tenant") {
        var tenant = appState.tenants.find(function (t) { return t.id === tenantId; });
        if (tenant) openTenantModal(tenant);
      } else if (action === "rotate-key") {
        var res = await apiRequest("/api/admin/clients/" + tenantId + "/keys/rotate", { method: "POST" });
        showRawApiKey(res.rawApiKey);
        await loadOverview();
      } else if (action === "toggle-tenant") {
        var currentlyEnabled = btn.getAttribute("data-enabled") === "1";
        await apiRequest("/api/admin/clients/" + tenantId, {
          method: "PATCH",
          json: { enabled: !currentlyEnabled },
        });
        await loadOverview();
      } else if (action === "duplicate-tenant") {
        var dupRes = await apiRequest("/api/admin/clients/" + tenantId + "/duplicate", { method: "POST" });
        showRawApiKey(dupRes.rawApiKey);
        await loadOverview();
      } else if (action === "delete-tenant") {
        await apiRequest("/api/admin/clients/" + tenantId, { method: "DELETE" });
        await loadOverview();
      }
    });
  }

  // Tenant Modal
  var tenantModal = document.getElementById("modal-tenant");
  var tenantForm = document.getElementById("form-tenant-modal");

  function openTenantModal(tenant) {
    var errEl = document.getElementById("tenant-modal-error");
    errEl.classList.add("hidden");
    document.getElementById("modal-tenant-title").textContent = tenant ? "Edit Tenant" : "Create New Tenant";
    document.getElementById("tenant-edit-id").value = tenant ? tenant.id : "";
    document.getElementById("tenant-business-name").value = tenant ? tenant.business_name : "";
    document.getElementById("tenant-slug").value = tenant ? tenant.slug : "";
    document.getElementById("tenant-company-name").value = tenant ? tenant.company_name : "";
    document.getElementById("tenant-assistant-name").value = tenant ? tenant.assistant_name : "Test Assistant";
    document.getElementById("tenant-domains").value = tenant && tenant.domains ? tenant.domains.join(", ") : "example.com, localhost";
    document.getElementById("tenant-welcome-msg").value = tenant ? tenant.welcome_message : "Hello! How can I help you today?";
    if (tenantModal && tenantModal.showModal) {
      tenantModal.showModal();
    } else if (tenantModal) {
      tenantModal.setAttribute("open", "open");
    }
  }

  function closeTenantModal() {
    if (tenantModal && tenantModal.close) tenantModal.close();
    else if (tenantModal) tenantModal.removeAttribute("open");
  }

  ["btn-new-tenant-top", "btn-open-create-tenant"].forEach(function (id) {
    var b = document.getElementById(id);
    if (b) {
      b.addEventListener("click", function () {
        switchView("view-clients");
        openTenantModal(null);
      });
    }
  });

  ["btn-close-tenant-modal", "btn-cancel-tenant-modal"].forEach(function (id) {
    var b = document.getElementById(id);
    if (b) b.addEventListener("click", closeTenantModal);
  });

  var bNameInput = document.getElementById("tenant-business-name");
  var slugInput = document.getElementById("tenant-slug");
  if (bNameInput && slugInput) {
    bNameInput.addEventListener("input", function () {
      if (!document.getElementById("tenant-edit-id").value) {
        slugInput.value = bNameInput.value
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
      }
    });
  }

  if (tenantForm) {
    tenantForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var editId = document.getElementById("tenant-edit-id").value;
      var errEl = document.getElementById("tenant-modal-error");
      errEl.classList.add("hidden");
      var payload = {
        business_name: document.getElementById("tenant-business-name").value.trim(),
        slug: document.getElementById("tenant-slug").value.trim(),
        company_name: document.getElementById("tenant-company-name").value.trim(),
        assistant_name: document.getElementById("tenant-assistant-name").value.trim(),
        domains: document.getElementById("tenant-domains").value.trim(),
        welcome_message: document.getElementById("tenant-welcome-msg").value.trim(),
      };
      try {
        if (editId) {
          await apiRequest("/api/admin/clients/" + editId, { method: "PATCH", json: payload });
        } else {
          var created = await apiRequest("/api/admin/clients", { method: "POST", json: payload });
          if (created.rawApiKey) showRawApiKey(created.rawApiKey);
        }
        closeTenantModal();
        await loadOverview();
      } catch (err) {
        errEl.textContent = err.message;
        errEl.classList.remove("hidden");
      }
    });
  }

  // --- 5. KNOWLEDGE BASE ---
  async function loadKnowledgeFiles() {
    var tenantId = document.getElementById("knowledge-tenant-select").value;
    var qs = tenantId ? "?tenant_id=" + encodeURIComponent(tenantId) : "";
    var files = await apiRequest("/api/admin/knowledge" + qs);
    var tbody = document.getElementById("knowledge-table-body");
    if (!files.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="meta-inline">No documents indexed yet.</td></tr>';
      return;
    }
    var tenantMap = {};
    appState.tenants.forEach(function (t) { tenantMap[t.id] = t.business_name; });

    tbody.innerHTML = files
      .map(function (f) {
        var stClass =
          f.status === "ready"
            ? "status-text-nominal"
            : f.status === "failed"
            ? "status-text-error"
            : "status-text-warn";
        return (
          "<tr>" +
          "<td><strong>" + escapeHtml(f.original_filename) + "</strong>" +
          (f.error_message ? "<div class='status-text-error meta-inline'>" + escapeHtml(f.error_message) + "</div>" : "") +
          "</td>" +
          "<td>" + escapeHtml(tenantMap[f.tenant_id] || f.tenant_id.slice(0, 8)) + "</td>" +
          '<td><span class="' + stClass + '">' + escapeHtml(f.status) + "</span> · v" + escapeHtml(f.active_version) + "</td>" +
          '<td class="num-col">' + escapeHtml(f.file_size) + " B</td>" +
          '<td class="num-col">' + escapeHtml(f.chunk_count) + "</td>" +
          '<td class="actions-col">' +
          '<button type="button" class="btn btn-secondary btn-sm" data-kaction="rebuild" data-tid="' + escapeHtml(f.tenant_id) + '" data-fid="' + escapeHtml(f.id) + '">Rebuild</button> ' +
          '<button type="button" class="btn btn-danger btn-sm" data-kaction="delete" data-tid="' + escapeHtml(f.tenant_id) + '" data-fid="' + escapeHtml(f.id) + '">Delete</button>' +
          "</td></tr>"
        );
      })
      .join("");
  }

  var kTenantSelect = document.getElementById("knowledge-tenant-select");
  if (kTenantSelect) kTenantSelect.addEventListener("change", loadKnowledgeFiles);

  var kForm = document.getElementById("knowledge-upload-form");
  if (kForm) {
    kForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var tenantId = document.getElementById("upload-tenant-id").value;
      var fileInput = document.getElementById("upload-file-input");
      var msgBox = document.getElementById("upload-status-msg");
      msgBox.classList.add("hidden");
      if (!fileInput.files || !fileInput.files[0]) return;

      var fd = new FormData();
      fd.append("tenant_id", tenantId);
      fd.append("file", fileInput.files[0]);

      try {
        var uploaded = await apiRequest("/api/admin/knowledge/upload", {
          method: "POST",
          body: fd,
        });
        msgBox.className = "alert alert-success";
        msgBox.textContent =
          "Indexed '" + uploaded.original_filename + "' (" + uploaded.chunk_count + " chunks, version v" + uploaded.active_version + ").";
        msgBox.classList.remove("hidden");
        fileInput.value = "";
        await loadKnowledgeFiles();
        await loadOverview();
      } catch (err) {
        msgBox.className = "alert alert-error";
        msgBox.textContent = err.message;
        msgBox.classList.remove("hidden");
      }
    });
  }

  var kTableBody = document.getElementById("knowledge-table-body");
  if (kTableBody) {
    kTableBody.addEventListener("click", async function (ev) {
      var btn = ev.target.closest("button[data-kaction]");
      if (!btn) return;
      var act = btn.getAttribute("data-kaction");
      var tid = btn.getAttribute("data-tid");
      var fid = btn.getAttribute("data-fid");
      if (act === "rebuild") {
        await apiRequest("/api/admin/knowledge/" + tid + "/" + fid + "/rebuild", { method: "POST" });
        await loadKnowledgeFiles();
      } else if (act === "delete") {
        await apiRequest("/api/admin/knowledge/" + tid + "/" + fid, { method: "DELETE" });
        await loadKnowledgeFiles();
        await loadOverview();
      }
    });
  }

  // --- 6. CONVERSATIONS ---
  async function loadConversations() {
    var tid = document.getElementById("conv-tenant-select").value;
    var search = document.getElementById("conv-search-input").value.trim();
    var params = new URLSearchParams();
    if (tid) params.set("tenant_id", tid);
    if (search) params.set("search", search);
    var list = await apiRequest("/api/admin/conversations?" + params.toString());
    var container = document.getElementById("conversations-list");
    if (!list.length) {
      container.innerHTML = '<div class="meta-inline">No conversations found.</div>';
      return;
    }
    container.innerHTML = list
      .map(function (c) {
        return (
          '<button type="button" class="session-item" data-cid="' + escapeHtml(c.id) + '" data-tid="' + escapeHtml(c.tenant_id) + '">' +
          "<div><strong>Session " + escapeHtml(c.session_id.slice(0, 18)) + "</strong></div>" +
          "<div class='meta-inline'>" + escapeHtml(c.message_count || 0) + " messages · " + escapeHtml(String(c.updated_at || "").slice(0, 16).replace("T", " ")) + "</div>" +
          (c.last_message ? "<div class='meta-inline' style='margin-top:4px;'>" + escapeHtml(c.last_message.slice(0, 70)) + "</div>" : "") +
          "</button>"
        );
      })
      .join("");

    var first = list[0];
    if (first) inspectConversation(first.tenant_id, first.id);
  }

  async function inspectConversation(tenantId, convId) {
    var detail = await apiRequest("/api/admin/conversations/" + tenantId + "/" + convId);
    document.getElementById("conv-detail-title").textContent = "Session: " + detail.session_id;
    document.getElementById("conv-detail-meta").textContent = "ID: " + detail.id.slice(0, 12);
    var sumBox = document.getElementById("conversation-summary-box");
    if (detail.summary) {
      sumBox.textContent = detail.summary;
      sumBox.classList.remove("hidden");
    } else {
      sumBox.classList.add("hidden");
    }
    var pane = document.getElementById("conversation-messages-pane");
    var msgs = detail.messages || [];
    pane.innerHTML = msgs
      .map(function (m) {
        var srcs = (m.sources || [])
          .map(function (s) { return s.source + "#" + s.chunkIndex; })
          .join(", ");
        return (
          '<div class="transcript-msg ' + escapeHtml(m.role) + '">' +
          "<div class='meta-inline'><strong>" + escapeHtml(m.role.toUpperCase()) + "</strong> · Support: " + escapeHtml(m.support_level) +
          (srcs ? " · Sources: " + escapeHtml(srcs) : "") +
          "</div>" +
          "<div style='margin-top:4px;'>" + escapeHtml(m.content) + "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  var convListEl = document.getElementById("conversations-list");
  if (convListEl) {
    convListEl.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button[data-cid]");
      if (!btn) return;
      inspectConversation(btn.getAttribute("data-tid"), btn.getAttribute("data-cid"));
    });
  }
  ["conv-tenant-select", "conv-search-input"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener("input", loadConversations);
  });

  // --- 7. LEADS CRM ---
  var currentLeads = [];
  async function loadLeads() {
    var tid = document.getElementById("leads-tenant-select").value;
    var status = document.getElementById("leads-status-select").value;
    var search = document.getElementById("leads-search-input").value.trim();
    var params = new URLSearchParams();
    if (tid) params.set("tenant_id", tid);
    if (status) params.set("status", status);
    if (search) params.set("search", search);

    currentLeads = await apiRequest("/api/admin/leads?" + params.toString());
    var tbody = document.getElementById("leads-table-body");
    if (!currentLeads.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="meta-inline">No leads match current filter.</td></tr>';
      return;
    }
    tbody.innerHTML = currentLeads
      .map(function (l) {
        return (
          "<tr>" +
          "<td><strong>" + escapeHtml(l.name) + "</strong><div class='meta-inline'>" + escapeHtml(l.email) + (l.phone ? " · " + escapeHtml(l.phone) : "") + "</div></td>" +
          "<td>" + escapeHtml(l.requirement || l.message || "—") +
          (l.internal_notes ? "<div class='meta-inline'>Note: " + escapeHtml(l.internal_notes) + "</div>" : "") +
          "</td>" +
          "<td><strong>" + escapeHtml(l.status) + "</strong></td>" +
          "<td>" + escapeHtml(l.assignee || "Unassigned") + "</td>" +
          "<td class='meta-inline font-mono'>" + escapeHtml(l.notification_status || "sent") + "</td>" +
          '<td class="num-col">' + escapeHtml(String(l.created_at || "").slice(0, 10)) + "</td>" +
          '<td class="actions-col"><button type="button" class="btn btn-secondary btn-sm" data-lead-id="' + escapeHtml(l.id) + '">Update</button></td>' +
          "</tr>"
        );
      })
      .join("");
  }

  ["leads-tenant-select", "leads-status-select", "leads-search-input"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener("input", loadLeads);
  });

  var exportCsvBtn = document.getElementById("btn-export-leads-csv");
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener("click", function () {
      var tid = document.getElementById("leads-tenant-select").value;
      var status = document.getElementById("leads-status-select").value;
      var search = document.getElementById("leads-search-input").value.trim();
      var params = new URLSearchParams();
      if (tid) params.set("tenant_id", tid);
      if (status) params.set("status", status);
      if (search) params.set("search", search);
      window.location.href = "/api/admin/leads/export?" + params.toString();
    });
  }

  var leadModal = document.getElementById("modal-lead");
  var leadsTbody = document.getElementById("leads-table-body");
  if (leadsTbody) {
    leadsTbody.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button[data-lead-id]");
      if (!btn) return;
      var lead = currentLeads.find(function (l) { return l.id === btn.getAttribute("data-lead-id"); });
      if (!lead) return;
      document.getElementById("lead-edit-id").value = lead.id;
      document.getElementById("lead-edit-tenant-id").value = lead.tenant_id;
      document.getElementById("lead-edit-status").value = lead.status;
      document.getElementById("lead-edit-assignee").value = lead.assignee || "";
      document.getElementById("lead-edit-notes").value = lead.internal_notes || "";
      if (leadModal && leadModal.showModal) leadModal.showModal();
      else if (leadModal) leadModal.setAttribute("open", "open");
    });
  }

  function closeLeadModal() {
    if (leadModal && leadModal.close) leadModal.close();
    else if (leadModal) leadModal.removeAttribute("open");
  }
  ["btn-close-lead-modal", "btn-cancel-lead-modal"].forEach(function (id) {
    var b = document.getElementById(id);
    if (b) b.addEventListener("click", closeLeadModal);
  });

  var leadFormModal = document.getElementById("form-lead-modal");
  if (leadFormModal) {
    leadFormModal.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var lid = document.getElementById("lead-edit-id").value;
      var tid = document.getElementById("lead-edit-tenant-id").value;
      await apiRequest("/api/admin/leads/" + tid + "/" + lid, {
        method: "PATCH",
        json: {
          status: document.getElementById("lead-edit-status").value,
          assignee: document.getElementById("lead-edit-assignee").value.trim(),
          internal_notes: document.getElementById("lead-edit-notes").value.trim(),
        },
      });
      closeLeadModal();
      await loadLeads();
    });
  }

  // --- 8. PROMPT & RAG STUDIO ---
  async function loadPromptsForTenant() {
    var tid = document.getElementById("prompt-tenant-select").value;
    if (!tid) return;
    var tenant = appState.tenants.find(function (t) { return t.id === tid; });
    if (tenant) {
      document.getElementById("prompt-system-textarea").value = tenant.system_prompt || "";
      document.getElementById("prompt-fallback-textarea").value = tenant.fallback_answer || "";
    }
    var versions = await apiRequest("/api/admin/prompts/" + tid);
    var vList = document.getElementById("prompt-versions-list");
    vList.innerHTML = versions
      .map(function (v) {
        return (
          '<div class="list-row">' +
          "<span><strong>v" + escapeHtml(v.version_number) + "</strong> · " + escapeHtml(v.note || "Saved prompt") +
          "<div class='meta-inline'>" + escapeHtml(v.system_prompt.slice(0, 75)) + "...</div></span>" +
          '<button type="button" class="btn btn-secondary btn-sm" data-restore-v="' + escapeHtml(v.version_number) + '">Restore</button>' +
          "</div>"
        );
      })
      .join("");
  }

  var promptTenantSelect = document.getElementById("prompt-tenant-select");
  if (promptTenantSelect) promptTenantSelect.addEventListener("change", loadPromptsForTenant);

  var promptForm = document.getElementById("prompt-editor-form");
  if (promptForm) {
    promptForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var tid = document.getElementById("prompt-tenant-select").value;
      var msgEl = document.getElementById("prompt-save-msg");
      try {
        var ver = await apiRequest("/api/admin/prompts/" + tid, {
          method: "POST",
          json: {
            system_prompt: document.getElementById("prompt-system-textarea").value.trim(),
            fallback_answer: document.getElementById("prompt-fallback-textarea").value.trim(),
            note: document.getElementById("prompt-version-note").value.trim(),
          },
        });
        msgEl.className = "alert alert-success";
        msgEl.textContent = "Saved prompt version v" + ver.version_number + ".";
        msgEl.classList.remove("hidden");
        await loadOverview();
        await loadPromptsForTenant();
      } catch (err) {
        msgEl.className = "alert alert-error";
        msgEl.textContent = err.message;
        msgEl.classList.remove("hidden");
      }
    });
  }

  var promptVersionsList = document.getElementById("prompt-versions-list");
  if (promptVersionsList) {
    promptVersionsList.addEventListener("click", async function (ev) {
      var btn = ev.target.closest("button[data-restore-v]");
      if (!btn) return;
      var tid = document.getElementById("prompt-tenant-select").value;
      var vNum = btn.getAttribute("data-restore-v");
      await apiRequest("/api/admin/prompts/" + tid + "/restore/" + vNum, { method: "POST" });
      await loadOverview();
      await loadPromptsForTenant();
    });
  }

  var previewForm = document.getElementById("prompt-preview-form");
  if (previewForm) {
    previewForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var tid = document.getElementById("prompt-tenant-select").value;
      var q = document.getElementById("prompt-preview-question").value.trim();
      var out = document.getElementById("prompt-preview-result");
      var res = await apiRequest("/api/admin/prompts/" + tid + "/preview", {
        method: "POST",
        json: {
          message: q,
          system_prompt: document.getElementById("prompt-system-textarea").value.trim(),
          fallback_answer: document.getElementById("prompt-fallback-textarea").value.trim(),
        },
      });
      out.className = "summary-callout";
      out.innerHTML =
        "<div><strong>Support Level:</strong> " + escapeHtml(res.support_level) +
        " · <strong>Grounding Passed:</strong> " + (res.grounding_passed ? "Yes" : "Fallback Triggered") + "</div>" +
        "<div style='margin-top:8px;'>" + escapeHtml(res.answer) + "</div>";
      out.classList.remove("hidden");
    });
  }

  // --- 9. WIDGET CONFIG & EMBED CODE ---
  function populateWidgetForm() {
    var tid = document.getElementById("widget-tenant-select").value;
    var tenant = appState.tenants.find(function (t) { return t.id === tid; });
    if (!tenant) return;
    var w = tenant.widget_config || {};
    document.getElementById("wg-assistant-name").value = tenant.assistant_name || "AI Assistant";
    document.getElementById("wg-primary-color").value = w.primary_color || "#2563eb";
    document.getElementById("wg-theme-mode").value = w.theme_mode || "light";
    document.getElementById("wg-position").value = w.position || "bottom-right";
    document.getElementById("wg-width").value = w.width || 380;
    document.getElementById("wg-height").value = w.height || 560;
    document.getElementById("wg-radius").value = w.border_radius !== undefined ? w.border_radius : 14;
    document.getElementById("wg-welcome").value = tenant.welcome_message || "";
    document.getElementById("wg-placeholder").value = w.input_placeholder || "Ask a question...";
    document.getElementById("wg-questions").value = (w.suggested_questions || []).join("\n");

    var keyPreview = appState.demoPublicKeys[tenant.id] || "TENANT_PUBLIC_API_KEY";
    var embedSnippet =
      "<script\n" +
      '  src="' + publicBaseUrl + '/static/js/widget.js"\n' +
      '  data-client-id="' + tenant.id + '"\n' +
      '  data-api-key="' + keyPreview + '"\n' +
      "  defer>\n" +
      "</script>";
    document.getElementById("widget-embed-code").textContent = embedSnippet;
    document.getElementById("link-test-in-demo").href =
      "/demo.html?client_id=" + encodeURIComponent(tenant.id);
  }

  var wgTenantSelect = document.getElementById("widget-tenant-select");
  if (wgTenantSelect) wgTenantSelect.addEventListener("change", populateWidgetForm);

  var wgForm = document.getElementById("widget-config-form");
  if (wgForm) {
    wgForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      var tid = document.getElementById("widget-tenant-select").value;
      var msgEl = document.getElementById("widget-save-msg");
      try {
        await apiRequest("/api/admin/clients/" + tid, {
          method: "PATCH",
          json: {
            assistant_name: document.getElementById("wg-assistant-name").value.trim(),
            welcome_message: document.getElementById("wg-welcome").value.trim(),
            widget_config: {
              primary_color: document.getElementById("wg-primary-color").value.trim(),
              theme_mode: document.getElementById("wg-theme-mode").value,
              position: document.getElementById("wg-position").value,
              width: Number(document.getElementById("wg-width").value),
              height: Number(document.getElementById("wg-height").value),
              border_radius: Number(document.getElementById("wg-radius").value),
              input_placeholder: document.getElementById("wg-placeholder").value.trim(),
              suggested_questions: document
                .getElementById("wg-questions")
                .value.split("\n")
                .map(function (s) { return s.trim(); })
                .filter(Boolean),
            },
          },
        });
        msgEl.className = "alert alert-success";
        msgEl.textContent = "Widget configuration saved and active on /v1/config.";
        msgEl.classList.remove("hidden");
        await loadOverview();
        populateWidgetForm();
      } catch (err) {
        msgEl.className = "alert alert-error";
        msgEl.textContent = err.message;
        msgEl.classList.remove("hidden");
      }
    });
  }

  var copyEmbedBtn = document.getElementById("btn-copy-embed");
  if (copyEmbedBtn) {
    copyEmbedBtn.addEventListener("click", function () {
      var code = document.getElementById("widget-embed-code").textContent;
      if (navigator.clipboard && code) navigator.clipboard.writeText(code);
      copyEmbedBtn.textContent = "Copied Embed Script!";
      setTimeout(function () { copyEmbedBtn.textContent = "Copy Embed Script"; }, 1500);
    });
  }

  // --- 10. ANALYTICS ---
  async function loadAnalytics() {
    var tid = document.getElementById("analytics-tenant-select").value;
    var sDate = document.getElementById("analytics-start-date").value;
    var eDate = document.getElementById("analytics-end-date").value;
    var params = new URLSearchParams();
    if (tid) params.set("tenant_id", tid);
    if (sDate) params.set("start_date", sDate);
    if (eDate) params.set("end_date", eDate);

    var data = await apiRequest("/api/admin/overview?" + params.toString());
    var s = data.summary || {};
    var sb = s.support_breakdown || {};
    document.getElementById("an-total-requests").textContent = String(s.total_requests || 0);
    document.getElementById("an-usage-units").textContent = String(s.total_usage_units || 0);
    document.getElementById("an-avg-latency").textContent = (s.avg_latency_ms || 0) + " ms";
    document.getElementById("an-p95-latency").textContent = "p95: " + (s.p95_latency_ms || 0) + " ms";
    document.getElementById("an-support-ratio").textContent =
      (sb.SUPPORTED || 0) + " / " + (sb.PARTIAL || 0) + " / " + (sb.UNKNOWN || 0);

    var dailyBody = document.getElementById("analytics-daily-body");
    var daily = data.daily_series || [];
    dailyBody.innerHTML = daily.length
      ? daily
          .map(function (d) {
            return (
              "<tr>" +
              "<td class='font-mono'>" + escapeHtml(d.date) + "</td>" +
              '<td class="num-col">' + escapeHtml(d.conversations) + "</td>" +
              '<td class="num-col">' + escapeHtml(d.messages) + "</td>" +
              '<td class="num-col">' + escapeHtml(d.leads) + "</td>" +
              '<td class="num-col">' + escapeHtml(d.usage_units) + "</td>" +
              '<td class="num-col">' + escapeHtml(d.avg_latency_ms) + " ms</td>" +
              "</tr>"
            );
          })
          .join("")
      : '<tr><td colspan="6" class="meta-inline">No activity recorded for selected period.</td></tr>';

    var tenantBody = document.getElementById("analytics-tenant-body");
    var tStats = data.tenant_stats || [];
    tenantBody.innerHTML = tStats
      .map(function (ts) {
        return (
          "<tr>" +
          "<td><strong>" + escapeHtml(ts.business_name) + "</strong></td>" +
          '<td class="num-col">' + escapeHtml(ts.conversations) + "</td>" +
          '<td class="num-col">' + escapeHtml(ts.messages) + "</td>" +
          '<td class="num-col">' + escapeHtml(ts.leads) + "</td>" +
          '<td class="num-col">' + escapeHtml(ts.knowledge_chunks) + "</td>" +
          '<td class="num-col">' + escapeHtml(ts.avg_latency_ms) + " ms</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  var applyAnBtn = document.getElementById("btn-apply-analytics-filter");
  if (applyAnBtn) applyAnBtn.addEventListener("click", loadAnalytics);

  // --- 11. PROVIDER DIAGNOSTICS & OUTBOX ---
  async function loadSettingsAndNotifications() {
    var res = await apiRequest("/api/admin/providers/test", { method: "POST" });
    var d = res.diagnostics || {};
    var grid = document.getElementById("providers-diagnostics-grid");
    var cards = [
      { title: "Database Provider", obj: d.database },
      { title: "Vector Search (pgvector)", obj: d.vector },
      { title: "Private File Storage", obj: d.storage },
      { title: "Embeddings Provider", obj: d.embeddings },
      { title: "Chat LLM Provider", obj: d.llm },
      { title: "Notifications & TTS", obj: d.notifications },
    ];
    grid.innerHTML = cards
      .map(function (c) {
        var o = c.obj || {};
        var ok = Boolean(o.ok);
        var details = Object.keys(o)
          .filter(function (k) { return k !== "ok"; })
          .map(function (k) {
            return "<div class='meta-inline'><strong>" + escapeHtml(k) + ":</strong> <span class='font-mono'>" + escapeHtml(String(o[k])) + "</span></div>";
          })
          .join("");
        return (
          '<div class="diag-card">' +
          "<div style='display:flex;justify-content:space-between;margin-bottom:8px;'>" +
          "<strong>" + escapeHtml(c.title) + "</strong>" +
          '<span class="' + (ok ? "status-text-nominal" : "status-text-error") + '">' + (ok ? "Nominal" : "Error") + "</span>" +
          "</div>" +
          details +
          "</div>"
        );
      })
      .join("");

    var notifs = await apiRequest("/api/admin/notifications");
    var nBody = document.getElementById("notifications-table-body");
    nBody.innerHTML = notifs.length
      ? notifs
          .map(function (n) {
            return (
              "<tr>" +
              "<td><strong>" + escapeHtml(n.subject) + "</strong><div class='meta-inline'>" + escapeHtml(n.recipient) + "</div></td>" +
              "<td class='font-mono'>" + escapeHtml(n.channel) + "</td>" +
              "<td>" + escapeHtml(n.status) + "</td>" +
              '<td class="num-col">' + escapeHtml(n.attempts) + "</td>" +
              "<td class='meta-inline'>" + escapeHtml(n.last_error || "—") + "</td>" +
              '<td class="num-col">' + escapeHtml(String(n.updated_at || "").slice(0, 19).replace("T", " ")) + "</td>" +
              "</tr>"
            );
          })
          .join("")
      : '<tr><td colspan="6" class="meta-inline">No outbox notifications recorded yet.</td></tr>';
  }

  var testProvBtn = document.getElementById("btn-test-providers");
  if (testProvBtn) {
    testProvBtn.addEventListener("click", async function () {
      var banner = document.getElementById("provider-test-banner");
      try {
        var res = await apiRequest("/api/admin/providers/test", { method: "POST" });
        banner.className = "alert " + (res.ok ? "alert-success" : "alert-error");
        banner.textContent = res.ok
          ? "All configured providers passed live connectivity verification."
          : "One or more providers reported an error. Inspect diagnostics below.";
        banner.classList.remove("hidden");
        await loadSettingsAndNotifications();
      } catch (err) {
        banner.className = "alert alert-error";
        banner.textContent = err.message;
        banner.classList.remove("hidden");
      }
    });
  }

  var retryNotifBtn = document.getElementById("btn-retry-notifications");
  if (retryNotifBtn) {
    retryNotifBtn.addEventListener("click", async function () {
      await apiRequest("/api/admin/notifications/retry", { method: "POST" });
      await loadSettingsAndNotifications();
    });
  }

  var refreshBtn = document.getElementById("btn-refresh-overview");
  if (refreshBtn) refreshBtn.addEventListener("click", loadOverview);

  // Initialize dashboard
  loadOverview();
})();
