/**
 * ORBIT AI — EMBEDDABLE CUSTOMER WIDGET (widget.js)
 *
 * Vanilla browser JavaScript widget supporting:
 * - Single script-tag embed (`data-client-id`, `data-api-key`, optional `data-base-url`)
 * - Authenticated tenant config loading (`GET /v1/config`)
 * - Dynamic theme mode (light/dark), primary color, dimensions, radius, position
 * - Conversation & session persistence per tenant
 * - Validated RAG chat (`POST /v1/chat`) with retry on transient network failure
 * - Idempotent lead capture (`POST /v1/leads`) with proper scoped `leadSubmitted` state
 *   (Fixes Defect #1 and Defect #15)
 */
(function () {
  "use strict";

  function findWidgetScript() {
    if (document.currentScript && document.currentScript.getAttribute("data-api-key")) {
      return document.currentScript;
    }
    var scripts = document.querySelectorAll("script[data-api-key]");
    return scripts.length ? scripts[scripts.length - 1] : null;
  }

  function resolveBaseUrl(scriptEl) {
    if (scriptEl && scriptEl.getAttribute("data-base-url")) {
      return scriptEl.getAttribute("data-base-url").replace(/\/+$/, "");
    }
    if (scriptEl && scriptEl.src) {
      try {
        var u = new URL(scriptEl.src, window.location.origin);
        return u.origin;
      } catch (_e) {}
    }
    return window.location.origin;
  }

  function ensureStylesheet(baseUrl) {
    if (document.getElementById("orbit-widget-stylesheet")) return;
    var link = document.createElement("link");
    link.id = "orbit-widget-stylesheet";
    link.rel = "stylesheet";
    link.href = baseUrl + "/static/css/widget.css";
    document.head.appendChild(link);
  }

  function generateUuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0;
      var v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function createOrbitWidgetInstance(options) {
    var baseUrl = (options.baseUrl || window.location.origin).replace(/\/+$/, "");
    var clientId = options.clientId || "";
    var apiKey = options.apiKey || "";
    var mountContainer = options.container || null;
    var startOpen = Boolean(options.startOpen);

    // Properly declared instance state (Fixes Defect #1: no undeclared leadSubmitted variable)
    var state = {
      isOpen: startOpen,
      isMinimized: false,
      isLoading: false,
      leadFormVisible: false,
      leadSubmitted: false,
      leadIdempotencyKey: generateUuid(),
      sessionId: "",
      conversationId: "",
      config: null,
      messages: [],
    };

    var storagePrefix = "orbit_widget_" + (clientId || "default") + "_";
    try {
      state.sessionId = sessionStorage.getItem(storagePrefix + "session_id") || generateUuid();
      sessionStorage.setItem(storagePrefix + "session_id", state.sessionId);
      state.conversationId = sessionStorage.getItem(storagePrefix + "conversation_id") || "";
    } catch (_e) {
      state.sessionId = generateUuid();
    }

    ensureStylesheet(baseUrl);

    var existingRoot = document.getElementById("orbit-widget-live-root");
    if (existingRoot && !mountContainer) {
      existingRoot.remove();
    }

    var root = document.createElement("div");
    if (!mountContainer) {
      root.id = "orbit-widget-live-root";
    }
    root.className = "orbit-widget-root " + (mountContainer ? "" : "orbit-widget-fixed orbit-pos-bottom-right");
    if (mountContainer) {
      mountContainer.innerHTML = "";
      mountContainer.appendChild(root);
    } else {
      document.body.appendChild(root);
    }

    function applyThemeVariables(cfg) {
      var theme = (cfg && cfg.theme) || {};
      var primary = theme.primaryColor || "#2563eb";
      var mode = (theme.themeMode || "light").toLowerCase();
      var position = (theme.position || "bottom-right").toLowerCase();
      var width = (theme.width || 380) + "px";
      var height = (theme.height || 560) + "px";
      var radius = (theme.borderRadius !== undefined ? theme.borderRadius : 14) + "px";

      root.style.setProperty("--orbit-primary", primary);
      root.style.setProperty("--orbit-width", width);
      root.style.setProperty("--orbit-height", height);
      root.style.setProperty("--orbit-radius", radius);

      root.classList.toggle("orbit-theme-dark", mode === "dark");
      root.classList.toggle("orbit-theme-light", mode !== "dark");

      if (!mountContainer) {
        root.classList.toggle("orbit-pos-bottom-left", position === "bottom-left");
        root.classList.toggle("orbit-pos-bottom-right", position !== "bottom-left");
      }
    }

    async function fetchWithRetry(url, init, retries) {
      var attempts = 0;
      var maxAttempts = (retries || 1) + 1;
      while (attempts < maxAttempts) {
        attempts += 1;
        try {
          var res = await fetch(url, init);
          if (res.status >= 500 && attempts < maxAttempts) {
            await new Promise(function (r) { setTimeout(r, 250 * attempts); });
            continue;
          }
          return res;
        } catch (err) {
          if (attempts >= maxAttempts) throw err;
          await new Promise(function (r) { setTimeout(r, 250 * attempts); });
        }
      }
    }

    function render() {
      var cfg = state.config || {
        assistantName: "AI Assistant",
        companyName: "Customer Support",
        welcomeMessage: "Hello! How can I help you today?",
        leadCaptureEnabled: true,
        leadSettings: { customPrompt: "Leave your contact details so our team can follow up." },
        theme: {
          primaryColor: "#2563eb",
          themeMode: "light",
          inputPlaceholder: "Ask a question...",
          showBranding: true,
          suggestedQuestions: [],
        },
      };
      applyThemeVariables(cfg);

      root.innerHTML = "";

      if (state.isOpen) {
        var win = document.createElement("div");
        win.className = "orbit-chat-window" + (state.isMinimized ? " orbit-minimized" : "");
        win.setAttribute("role", "dialog");
        win.setAttribute("aria-label", cfg.assistantName || "Chat Assistant");

        // Header
        var header = document.createElement("div");
        header.className = "orbit-chat-header";

        var info = document.createElement("div");
        info.className = "orbit-header-info";

        var avatar = document.createElement("div");
        avatar.className = "orbit-avatar";
        var avatarUrl = cfg.theme && (cfg.theme.avatarUrl || cfg.theme.logoUrl);
        if (avatarUrl) {
          var img = document.createElement("img");
          img.src = avatarUrl;
          img.alt = cfg.assistantName;
          img.referrerPolicy = "no-referrer";
          img.onerror = function () {
            avatar.textContent = (cfg.assistantName || "A").charAt(0).toUpperCase();
          };
          avatar.appendChild(img);
        } else {
          avatar.textContent = (cfg.assistantName || "A").charAt(0).toUpperCase();
        }

        var titleWrap = document.createElement("div");
        var titleEl = document.createElement("div");
        titleEl.className = "orbit-assistant-title";
        titleEl.textContent = cfg.assistantName || "AI Assistant";
        var subEl = document.createElement("div");
        subEl.className = "orbit-company-subtitle";
        subEl.textContent = cfg.companyName || cfg.businessName || "";
        titleWrap.appendChild(titleEl);
        titleWrap.appendChild(subEl);

        info.appendChild(avatar);
        info.appendChild(titleWrap);

        var controls = document.createElement("div");
        controls.className = "orbit-header-controls";

        if (cfg.leadCaptureEnabled && !state.leadSubmitted) {
          var leadToggleBtn = document.createElement("button");
          leadToggleBtn.type = "button";
          leadToggleBtn.className = "orbit-icon-btn";
          leadToggleBtn.title = "Leave contact details";
          leadToggleBtn.setAttribute("aria-label", "Open Contact Form");
          leadToggleBtn.textContent = "✉";
          leadToggleBtn.onclick = function () {
            state.leadFormVisible = !state.leadFormVisible;
            render();
          };
          controls.appendChild(leadToggleBtn);
        }

        var minBtn = document.createElement("button");
        minBtn.type = "button";
        minBtn.className = "orbit-icon-btn";
        minBtn.setAttribute("aria-label", state.isMinimized ? "Expand chat" : "Minimize chat");
        minBtn.textContent = state.isMinimized ? "□" : "—";
        minBtn.onclick = function () {
          state.isMinimized = !state.isMinimized;
          render();
        };

        var closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "orbit-icon-btn";
        closeBtn.setAttribute("aria-label", "Close chat");
        closeBtn.textContent = "✕";
        closeBtn.onclick = function () {
          state.isOpen = false;
          render();
        };

        controls.appendChild(minBtn);
        controls.appendChild(closeBtn);
        header.appendChild(info);
        header.appendChild(controls);
        win.appendChild(header);

        if (!state.isMinimized) {
          var msgContainer = document.createElement("div");
          msgContainer.className = "orbit-messages-container";

          // Welcome message
          var welcomeBubble = document.createElement("div");
          welcomeBubble.className = "orbit-msg-bubble orbit-msg-assistant";
          welcomeBubble.textContent = cfg.welcomeMessage || "Hello! How can I help you today?";
          msgContainer.appendChild(welcomeBubble);

          state.messages.forEach(function (m) {
            var b = document.createElement("div");
            b.className =
              "orbit-msg-bubble " +
              (m.role === "user"
                ? "orbit-msg-user"
                : m.role === "error"
                ? "orbit-msg-error"
                : "orbit-msg-assistant");
            b.textContent = m.content;
            msgContainer.appendChild(b);
          });

          if (state.isLoading) {
            var typing = document.createElement("div");
            typing.className = "orbit-msg-bubble orbit-msg-assistant";
            typing.textContent = "Typing...";
            msgContainer.appendChild(typing);
          }

          // Inline Lead Capture Form (avoid duplicates; only shown if not yet submitted)
          if (state.leadFormVisible && !state.leadSubmitted && cfg.leadCaptureEnabled) {
            var leadCard = document.createElement("form");
            leadCard.className = "orbit-lead-card";
            leadCard.setAttribute("aria-label", "Lead Capture Form");

            var leadTitle = document.createElement("div");
            leadTitle.className = "orbit-lead-title";
            leadTitle.textContent =
              (cfg.leadSettings && cfg.leadSettings.customPrompt) ||
              "Leave your contact details so our team can assist you:";
            leadCard.appendChild(leadTitle);

            var nameInput = document.createElement("input");
            nameInput.type = "text";
            nameInput.className = "orbit-lead-input";
            nameInput.placeholder = "Your Name *";
            nameInput.required = true;
            nameInput.setAttribute("aria-label", "Your Name");

            var emailInput = document.createElement("input");
            emailInput.type = "email";
            emailInput.className = "orbit-lead-input";
            emailInput.placeholder = "Your Email *";
            emailInput.required = true;
            emailInput.setAttribute("aria-label", "Your Email");

            var phoneInput = document.createElement("input");
            phoneInput.type = "tel";
            phoneInput.className = "orbit-lead-input";
            phoneInput.placeholder = "Phone Number";
            phoneInput.setAttribute("aria-label", "Phone Number");

            var reqInput = document.createElement("input");
            reqInput.type = "text";
            reqInput.className = "orbit-lead-input";
            reqInput.placeholder = "How can we help?";
            reqInput.setAttribute("aria-label", "Requirement");

            var submitLeadBtn = document.createElement("button");
            submitLeadBtn.type = "submit";
            submitLeadBtn.className = "orbit-lead-submit";
            submitLeadBtn.textContent = "Send Contact Details";

            leadCard.appendChild(nameInput);
            leadCard.appendChild(emailInput);
            leadCard.appendChild(phoneInput);
            leadCard.appendChild(reqInput);
            leadCard.appendChild(submitLeadBtn);

            leadCard.onsubmit = function (ev) {
              ev.preventDefault();
              if (state.leadSubmitted) return;
              submitLead({
                name: nameInput.value.trim(),
                email: emailInput.value.trim(),
                phone: phoneInput.value.trim(),
                requirement: reqInput.value.trim(),
                message: reqInput.value.trim(),
              });
            };
            msgContainer.appendChild(leadCard);
          }

          win.appendChild(msgContainer);
          setTimeout(function () {
            msgContainer.scrollTop = msgContainer.scrollHeight;
          }, 10);

          // Suggested questions
          var suggestions = (cfg.theme && cfg.theme.suggestedQuestions) || [];
          if (state.messages.length === 0 && suggestions.length > 0) {
            var sugWrap = document.createElement("div");
            sugWrap.className = "orbit-suggestions";
            suggestions.forEach(function (qText) {
              var sBtn = document.createElement("button");
              sBtn.type = "button";
              sBtn.className = "orbit-suggestion-btn";
              sBtn.textContent = qText;
              sBtn.onclick = function () {
                sendMessage(qText);
              };
              sugWrap.appendChild(sBtn);
            });
            win.appendChild(sugWrap);
          }

          // Composer
          var composer = document.createElement("form");
          composer.className = "orbit-composer";
          var input = document.createElement("input");
          input.type = "text";
          input.className = "orbit-composer-input";
          input.placeholder = (cfg.theme && cfg.theme.inputPlaceholder) || "Ask a question...";
          input.setAttribute("aria-label", "Chat Message Input");
          var sendBtn = document.createElement("button");
          sendBtn.type = "submit";
          sendBtn.className = "orbit-composer-send";
          sendBtn.textContent = "Send";
          composer.appendChild(input);
          composer.appendChild(sendBtn);

          composer.onsubmit = function (ev) {
            ev.preventDefault();
            var val = input.value.trim();
            if (!val || state.isLoading) return;
            input.value = "";
            sendMessage(val);
          };
          win.appendChild(composer);

          if (!cfg.theme || cfg.theme.showBranding !== false) {
            var brandBar = document.createElement("div");
            brandBar.className = "orbit-branding-bar";
            brandBar.textContent = "Powered by Orbit AI";
            win.appendChild(brandBar);
          }
        }

        root.appendChild(win);
      }

      if (!mountContainer) {
        var launcher = document.createElement("button");
        launcher.type = "button";
        launcher.className = "orbit-launcher-btn";
        launcher.setAttribute("aria-label", state.isOpen ? "Close Support Chat" : "Open Support Chat");
        launcher.textContent = state.isOpen ? "✕" : "💬";
        launcher.onclick = function () {
          state.isOpen = !state.isOpen;
          state.isMinimized = false;
          render();
        };
        root.appendChild(launcher);
      }
    }

    async function loadConfig() {
      if (!apiKey) {
        state.messages.push({
          role: "error",
          content: "Widget is missing data-api-key configuration.",
        });
        render();
        return;
      }
      try {
        var headers = { "x-api-key": apiKey };
        if (clientId) headers["x-client-id"] = clientId;
        var res = await fetchWithRetry(baseUrl + "/v1/config", { method: "GET", headers: headers }, 1);
        var data = await res.json();
        if (!res.ok) {
          throw new Error((data && data.error) || "Failed to load widget configuration.");
        }
        state.config = data;
        render();
      } catch (err) {
        state.messages.push({
          role: "error",
          content: err.message || "Unable to connect to support service.",
        });
        render();
      }
    }

    async function sendMessage(text) {
      state.messages.push({ role: "user", content: text });
      state.isLoading = true;
      render();

      try {
        var headers = {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        };
        if (clientId) headers["x-client-id"] = clientId;
        var payload = {
          message: text,
          sessionId: state.sessionId,
        };
        if (state.conversationId) {
          payload.conversationId = state.conversationId;
        }

        var res = await fetchWithRetry(
          baseUrl + "/v1/chat",
          {
            method: "POST",
            headers: headers,
            body: JSON.stringify(payload),
          },
          1
        );
        var data = await res.json();
        state.isLoading = false;

        if (!res.ok) {
          state.messages.push({
            role: "error",
            content: (data && data.error) || "Unable to complete request.",
          });
          render();
          return;
        }

        if (data.conversationId) {
          state.conversationId = data.conversationId;
          try {
            sessionStorage.setItem(storagePrefix + "conversation_id", state.conversationId);
          } catch (_e) {}
        }
        state.messages.push({ role: "assistant", content: data.answer });

        // If the assistant returned a fallback answer and lead capture is enabled, offer the lead form once
        if (data.isFallback && state.config && state.config.leadCaptureEnabled && !state.leadSubmitted) {
          state.leadFormVisible = true;
        }
        render();
      } catch (err) {
        state.isLoading = false;
        state.messages.push({
          role: "error",
          content: "Network error while sending message. Please try again.",
        });
        render();
      }
    }

    async function submitLead(leadData) {
      if (state.leadSubmitted) return;
      try {
        var headers = {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "x-idempotency-key": state.leadIdempotencyKey,
        };
        if (clientId) headers["x-client-id"] = clientId;
        var body = {
          name: leadData.name,
          email: leadData.email,
          phone: leadData.phone || "",
          requirement: leadData.requirement || "",
          message: leadData.message || "",
          conversationId: state.conversationId || null,
          idempotencyKey: state.leadIdempotencyKey,
        };
        var res = await fetchWithRetry(
          baseUrl + "/v1/leads",
          {
            method: "POST",
            headers: headers,
            body: JSON.stringify(body),
          },
          1
        );
        var data = await res.json();
        if (!res.ok) {
          state.messages.push({
            role: "error",
            content: (data && data.error) || "Could not submit contact details.",
          });
          render();
          return;
        }
        state.leadSubmitted = true;
        state.leadFormVisible = false;
        state.messages.push({
          role: "assistant",
          content: "Thank you, " + leadData.name + "! Your contact details have been received and our team will reach out shortly.",
        });
        render();
      } catch (_err) {
        state.messages.push({
          role: "error",
          content: "Network error while submitting contact form. Please retry.",
        });
        render();
      }
    }

    render();
    loadConfig();

    return {
      open: function () { state.isOpen = true; state.isMinimized = false; render(); },
      close: function () { state.isOpen = false; render(); },
      reloadConfig: loadConfig,
      getState: function () { return state; },
    };
  }

  window.OrbitWidget = {
    init: createOrbitWidgetInstance,
  };

  var scriptEl = findWidgetScript();
  if (scriptEl) {
    var apiKey = scriptEl.getAttribute("data-api-key");
    var clientId = scriptEl.getAttribute("data-client-id") || "";
    var baseUrl = resolveBaseUrl(scriptEl);
    var autoOpen = scriptEl.getAttribute("data-start-open") === "true";
    if (apiKey) {
      createOrbitWidgetInstance({
        baseUrl: baseUrl,
        clientId: clientId,
        apiKey: apiKey,
        startOpen: autoOpen,
      });
    }
  }
})();
