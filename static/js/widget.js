/**
 * ORBIT AI — PREMIUM DYNAMIC CUSTOMER WIDGET (widget.js)
 *
 * Vanilla browser JavaScript widget supporting:
 * - Single script-tag embed (`data-client-id`, `data-api-key`, optional `data-base-url`)
 * - Authenticated tenant config loading (`GET /v1/config`)
 * - Dynamic theme mode (light/dark), primary color, dimensions, radius, position
 * - Conversation & session persistence + restoration (`GET /v1/conversations/{id}/messages`)
 * - Clear conversation with confirmation (`DELETE /v1/conversations/{id}`)
 * - Progressive SSE streaming (`POST /v1/chat/stream`) with fallback to `POST /v1/chat`
 * - Contextual processing stages ("Understanding your question...", "Searching relevant information...", "Preparing your answer...")
 * - Safe Markdown rendering (bold, lists, inline code, code blocks, links)
 * - Assistant Message Action Toolbar: Like, Dislike (with optional reason), Copy, and Share (`POST /v1/feedback`)
 * - Auto-expanding textarea (`Enter` to send, `Shift+Enter` for newline) & idempotent retry on failure
 * - Idempotent lead capture (`POST /v1/leads`) with proper scoped `leadSubmitted` state
 */
(function () {
  "use strict";

  var PROCESSING_STAGES = [
    "Understanding your question...",
    "Searching relevant information...",
    "Preparing your answer...",
  ];

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

  function escapeHtml(raw) {
    return String(raw || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatInlineMarkdown(escapedLine) {
    // Inline code `code`
    var out = escapedLine.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold **text**
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Safe links [label](https://...)
    out = out.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    return out;
  }

  function renderSafeMarkdownToHtml(rawText) {
    var safe = escapeHtml(rawText || "");
    if (!safe.trim()) return "";

    // Extract fenced code blocks ```...```
    var codeBlocks = [];
    safe = safe.replace(/```(?:[a-zA-Z0-9_-]+)?\n?([\s\S]*?)```/g, function (_m, codeContent) {
      var idx = codeBlocks.length;
      codeBlocks.push("<pre><code>" + codeContent.trim() + "</code></pre>");
      return "%%ORBIT_CODE_BLOCK_" + idx + "%%";
    });

    var lines = safe.split(/\r?\n/);
    var htmlParts = [];
    var inUl = false;
    var inOl = false;

    function closeLists() {
      if (inUl) {
        htmlParts.push("</ul>");
        inUl = false;
      }
      if (inOl) {
        htmlParts.push("</ol>");
        inOl = false;
      }
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) {
        closeLists();
        continue;
      }
      var codeMatch = line.match(/^%%ORBIT_CODE_BLOCK_(\d+)%%$/);
      if (codeMatch) {
        closeLists();
        htmlParts.push(codeBlocks[Number(codeMatch[1])] || "");
        continue;
      }
      var ulMatch = line.match(/^(?:[-*•])\s+(.*)$/);
      if (ulMatch) {
        if (inOl) {
          htmlParts.push("</ol>");
          inOl = false;
        }
        if (!inUl) {
          htmlParts.push("<ul>");
          inUl = true;
        }
        htmlParts.push("<li>" + formatInlineMarkdown(ulMatch[1]) + "</li>");
        continue;
      }
      var olMatch = line.match(/^\d+[.)]\s+(.*)$/);
      if (olMatch) {
        if (inUl) {
          htmlParts.push("</ul>");
          inUl = false;
        }
        if (!inOl) {
          htmlParts.push("<ol>");
          inOl = true;
        }
        htmlParts.push("<li>" + formatInlineMarkdown(olMatch[1]) + "</li>");
        continue;
      }
      closeLists();
      htmlParts.push("<p>" + formatInlineMarkdown(line) + "</p>");
    }
    closeLists();
    return htmlParts.join("");
  }

  async function copyTextToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_e) {}
    }
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      document.body.removeChild(ta);
      return true;
    } catch (_err) {
      document.body.removeChild(ta);
      return false;
    }
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
      processingStageIndex: 0,
      confirmingClear: false,
      leadFormVisible: false,
      leadSubmitted: false,
      leadIdempotencyKey: generateUuid(),
      sessionId: "",
      conversationId: "",
      config: null,
      messages: [],
      draftText: "",
      toastMessage: "",
      dislikeReasonOpenForId: null,
    };

    var stageTimer = null;
    var toastTimer = null;

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

    function showToast(msg) {
      state.toastMessage = msg;
      if (toastTimer) clearTimeout(toastTimer);
      render();
      toastTimer = setTimeout(function () {
        state.toastMessage = "";
        render();
      }, 2200);
    }

    function startProcessingStages() {
      state.processingStageIndex = 0;
      if (stageTimer) clearInterval(stageTimer);
      stageTimer = setInterval(function () {
        if (!state.isLoading) {
          clearInterval(stageTimer);
          stageTimer = null;
          return;
        }
        if (state.processingStageIndex < PROCESSING_STAGES.length - 1) {
          state.processingStageIndex += 1;
          var statusEl = root.querySelector(".orbit-processing-status");
          if (statusEl) {
            statusEl.textContent = PROCESSING_STAGES[state.processingStageIndex];
          }
        }
      }, 550);
    }

    function stopProcessingStages() {
      if (stageTimer) {
        clearInterval(stageTimer);
        stageTimer = null;
      }
      state.processingStageIndex = 0;
    }

    function applyThemeVariables(cfg) {
      var theme = (cfg && cfg.theme) || {};
      var primary = theme.primaryColor || "#2563eb";
      var mode = (theme.themeMode || "light").toLowerCase();
      var position = (theme.position || "bottom-right").toLowerCase();
      var width = (theme.width || 400) + "px";
      var height = (theme.height || 600) + "px";
      var radius = (theme.borderRadius !== undefined ? theme.borderRadius : 16) + "px";

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

    function buildAssistantToolbar(msgObj, msgIndex) {
      var toolbar = document.createElement("div");
      toolbar.className = "orbit-msg-toolbar";
      toolbar.setAttribute("role", "group");
      toolbar.setAttribute("aria-label", "Message actions");

      // 1. Like Button
      var likeBtn = document.createElement("button");
      likeBtn.type = "button";
      likeBtn.className =
        "orbit-action-btn" + (msgObj.feedbackType === "like" ? " orbit-action-active" : "");
      likeBtn.setAttribute("aria-label", "Like response");
      likeBtn.setAttribute("aria-pressed", msgObj.feedbackType === "like" ? "true" : "false");
      likeBtn.setAttribute("data-action", "like");
      likeBtn.textContent = msgObj.feedbackType === "like" ? "👍 Helpful" : "👍 Like";
      likeBtn.onclick = function () {
        submitMessageFeedback(msgObj, "like", null);
      };

      // 2. Dislike Button
      var dislikeBtn = document.createElement("button");
      dislikeBtn.type = "button";
      dislikeBtn.className =
        "orbit-action-btn" + (msgObj.feedbackType === "dislike" ? " orbit-action-active" : "");
      dislikeBtn.setAttribute("aria-label", "Dislike response");
      dislikeBtn.setAttribute("aria-pressed", msgObj.feedbackType === "dislike" ? "true" : "false");
      dislikeBtn.setAttribute("data-action", "dislike");
      dislikeBtn.textContent = msgObj.feedbackType === "dislike" ? "👎 Feedback sent" : "👎 Dislike";
      dislikeBtn.onclick = function () {
        state.dislikeReasonOpenForId = msgObj.id || String(msgIndex);
        submitMessageFeedback(msgObj, "dislike", msgObj.feedbackReason || null);
      };

      // 3. Copy Button
      var copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "orbit-action-btn";
      copyBtn.setAttribute("aria-label", "Copy response");
      copyBtn.setAttribute("data-action", "copy");
      copyBtn.textContent = msgObj._copied ? "✓ Copied" : "📋 Copy";
      copyBtn.onclick = async function () {
        var ok = await copyTextToClipboard(msgObj.content || "");
        if (ok) {
          msgObj._copied = true;
          showToast("Copied response to clipboard");
          setTimeout(function () {
            msgObj._copied = false;
            render();
          }, 1800);
        }
      };

      // 4. Share Button
      var shareBtn = document.createElement("button");
      shareBtn.type = "button";
      shareBtn.className = "orbit-action-btn";
      shareBtn.setAttribute("aria-label", "Share response");
      shareBtn.setAttribute("data-action", "share");
      shareBtn.textContent = msgObj._shared ? "✓ Shared" : "↗ Share";
      shareBtn.onclick = async function () {
        var prevUserQ = "";
        for (var k = msgIndex - 1; k >= 0; k--) {
          if (state.messages[k] && state.messages[k].role === "user") {
            prevUserQ = state.messages[k].content;
            break;
          }
        }
        var shareText = prevUserQ
          ? "Question: " + prevUserQ + "\n\nAnswer: " + (msgObj.content || "")
          : String(msgObj.content || "");

        if (navigator.share) {
          try {
            await navigator.share({
              title: (state.config && state.config.assistantName) || "Support Response",
              text: shareText,
            });
            msgObj._shared = true;
            showToast("Response shared");
            return;
          } catch (_e) {}
        }
        var copied = await copyTextToClipboard(shareText);
        if (copied) {
          msgObj._shared = true;
          showToast("Shareable answer copied to clipboard");
          setTimeout(function () {
            msgObj._shared = false;
            render();
          }, 1800);
        }
      };

      toolbar.appendChild(likeBtn);
      toolbar.appendChild(dislikeBtn);
      toolbar.appendChild(copyBtn);
      toolbar.appendChild(shareBtn);
      return toolbar;
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

      // Preserve scroll position check before re-rendering
      var prevContainer = root.querySelector(".orbit-messages-container");
      var wasNearBottom = true;
      if (prevContainer) {
        wasNearBottom =
          prevContainer.scrollHeight - prevContainer.scrollTop - prevContainer.clientHeight < 72;
      }

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
          img.alt = cfg.assistantName || "Assistant Avatar";
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

        if (state.messages.length > 0 || state.conversationId) {
          var clearBtn = document.createElement("button");
          clearBtn.type = "button";
          clearBtn.className = "orbit-icon-btn";
          clearBtn.title = "Clear conversation";
          clearBtn.setAttribute("aria-label", "Clear conversation");
          clearBtn.textContent = "↺";
          clearBtn.onclick = function () {
            state.confirmingClear = !state.confirmingClear;
            render();
          };
          controls.appendChild(clearBtn);
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

        // Toast notification
        if (state.toastMessage) {
          var toastEl = document.createElement("div");
          toastEl.className = "orbit-toast";
          toastEl.setAttribute("role", "status");
          toastEl.textContent = state.toastMessage;
          win.appendChild(toastEl);
        }

        // Clear conversation confirmation bar
        if (state.confirmingClear && !state.isMinimized) {
          var confirmBar = document.createElement("div");
          confirmBar.className = "orbit-confirm-bar";
          var confirmLbl = document.createElement("span");
          confirmLbl.textContent = "Start a fresh conversation?";
          var confirmActs = document.createElement("div");
          confirmActs.className = "orbit-confirm-actions";

          var confirmYes = document.createElement("button");
          confirmYes.type = "button";
          confirmYes.className = "orbit-btn-Confirm";
          confirmYes.setAttribute("aria-label", "Confirm clear conversation");
          confirmYes.textContent = "Clear";
          confirmYes.onclick = function () {
            clearConversation();
          };

          var confirmNo = document.createElement("button");
          confirmNo.type = "button";
          confirmNo.className = "orbit-btn-cancel";
          confirmNo.setAttribute("aria-label", "Cancel clear conversation");
          confirmNo.textContent = "Cancel";
          confirmNo.onclick = function () {
            state.confirmingClear = false;
            render();
          };

          confirmActs.appendChild(confirmYes);
          confirmActs.appendChild(confirmNo);
          confirmBar.appendChild(confirmLbl);
          confirmBar.appendChild(confirmActs);
          win.appendChild(confirmBar);
        }

        if (!state.isMinimized) {
          var msgContainer = document.createElement("div");
          msgContainer.className = "orbit-messages-container";
          msgContainer.setAttribute("role", "log");
          msgContainer.setAttribute("aria-live", "polite");

          // Welcome message
          var welcomeTurn = document.createElement("div");
          welcomeTurn.className = "orbit-msg-turn orbit-msg-turn-assistant";
          var welcomeBubble = document.createElement("div");
          welcomeBubble.className = "orbit-msg-bubble orbit-msg-assistant";
          welcomeBubble.innerHTML = renderSafeMarkdownToHtml(
            cfg.welcomeMessage || "Hello! How can I help you today?"
          );
          welcomeTurn.appendChild(welcomeBubble);
          msgContainer.appendChild(welcomeTurn);

          state.messages.forEach(function (m, idx) {
            if (m.role === "error") {
              var errBubble = document.createElement("div");
              errBubble.className = "orbit-msg-bubble orbit-msg-error";
              var errSpan = document.createElement("span");
              errSpan.textContent = m.content;
              errBubble.appendChild(errSpan);
              if (m.retryText) {
                var retryBtn = document.createElement("button");
                retryBtn.type = "button";
                retryBtn.className = "orbit-retry-btn";
                retryBtn.setAttribute("aria-label", "Retry sending message");
                retryBtn.textContent = "Retry";
                retryBtn.onclick = function () {
                  state.messages.splice(idx, 1);
                  sendMessage(m.retryText, m.retryMessageId, true);
                };
                errBubble.appendChild(retryBtn);
              }
              msgContainer.appendChild(errBubble);
              return;
            }

            var turnWrap = document.createElement("div");
            turnWrap.className =
              "orbit-msg-turn " +
              (m.role === "user" ? "orbit-msg-turn-user" : "orbit-msg-turn-assistant");

            var b = document.createElement("div");
            b.className =
              "orbit-msg-bubble " +
              (m.role === "user" ? "orbit-msg-user" : "orbit-msg-assistant");

            if (m.role === "assistant") {
              b.innerHTML = renderSafeMarkdownToHtml(m.content || "");
            } else {
              b.textContent = m.content || "";
            }
            turnWrap.appendChild(b);

            // Show action toolbar on completed assistant messages
            if (m.role === "assistant" && !m.streaming && m.content) {
              turnWrap.appendChild(buildAssistantToolbar(m, idx));

              var reasonKey = m.id || String(idx);
              if (state.dislikeReasonOpenForId === reasonKey && m.feedbackType === "dislike") {
                var reasonBox = document.createElement("form");
                reasonBox.className = "orbit-feedback-reason-box";
                var reasonInput = document.createElement("input");
                reasonInput.type = "text";
                reasonInput.className = "orbit-feedback-reason-input";
                reasonInput.placeholder = "Optional: What could be improved?";
                reasonInput.value = m.feedbackReason || "";
                reasonInput.setAttribute("aria-label", "Feedback reason");
                var reasonSubmit = document.createElement("button");
                reasonSubmit.type = "submit";
                reasonSubmit.className = "orbit-feedback-reason-submit";
                reasonSubmit.textContent = "Save";
                reasonBox.appendChild(reasonInput);
                reasonBox.appendChild(reasonSubmit);
                reasonBox.onsubmit = function (ev) {
                  ev.preventDefault();
                  var rText = reasonInput.value.trim();
                  state.dislikeReasonOpenForId = null;
                  submitMessageFeedback(m, "dislike", rText || null);
                };
                turnWrap.appendChild(reasonBox);
              }
            }

            msgContainer.appendChild(turnWrap);
          });

          if (state.isLoading) {
            var procCard = document.createElement("div");
            procCard.className = "orbit-processing-card";
            procCard.setAttribute("aria-label", "Assistant is preparing a response");
            var procStatus = document.createElement("div");
            procStatus.className = "orbit-processing-status";
            procStatus.textContent =
              PROCESSING_STAGES[state.processingStageIndex] || PROCESSING_STAGES[0];
            var procBars = document.createElement("div");
            procBars.className = "orbit-processing-bars";
            var bar1 = document.createElement("div");
            bar1.className = "orbit-skeleton-line";
            var bar2 = document.createElement("div");
            bar2.className = "orbit-skeleton-line short";
            procBars.appendChild(bar1);
            procBars.appendChild(bar2);
            procCard.appendChild(procStatus);
            procCard.appendChild(procBars);
            msgContainer.appendChild(procCard);
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
          if (wasNearBottom) {
            setTimeout(function () {
              msgContainer.scrollTop = msgContainer.scrollHeight;
            }, 15);
          }

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

          // Auto-expanding Composer (textarea with Enter to send, Shift+Enter for newline)
          var composer = document.createElement("form");
          composer.className = "orbit-composer";
          var input = document.createElement("textarea");
          input.rows = 1;
          input.className = "orbit-composer-input";
          input.placeholder = (cfg.theme && cfg.theme.inputPlaceholder) || "Ask a question...";
          input.setAttribute("aria-label", "Chat Message Input");
          input.value = state.draftText || "";

          var sendBtn = document.createElement("button");
          sendBtn.type = "submit";
          sendBtn.className = "orbit-composer-send";
          sendBtn.textContent = "Send";
          sendBtn.disabled = state.isLoading || !input.value.trim();

          input.oninput = function () {
            state.draftText = input.value;
            input.style.height = "auto";
            input.style.height = Math.min(input.scrollHeight, 112) + "px";
            sendBtn.disabled = state.isLoading || !input.value.trim();
          };

          input.onkeydown = function (ev) {
            if (ev.key === "Enter" && !ev.shiftKey) {
              ev.preventDefault();
              var val = input.value.trim();
              if (!val || state.isLoading) return;
              state.draftText = "";
              input.value = "";
              sendMessage(val);
            }
          };

          composer.appendChild(input);
          composer.appendChild(sendBtn);

          composer.onsubmit = function (ev) {
            ev.preventDefault();
            var val = input.value.trim();
            if (!val || state.isLoading) return;
            state.draftText = "";
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

    async function restoreConversationMessages() {
      if (!apiKey || !state.conversationId) return;
      try {
        var headers = { "x-api-key": apiKey };
        if (clientId) headers["x-client-id"] = clientId;
        var res = await fetch(
          baseUrl + "/v1/conversations/" + encodeURIComponent(state.conversationId) + "/messages",
          { method: "GET", headers: headers }
        );
        if (!res.ok) {
          if (res.status === 404) {
            state.conversationId = "";
            try {
              sessionStorage.removeItem(storagePrefix + "conversation_id");
            } catch (_e) {}
          }
          return;
        }
        var data = await res.json();
        if (data && Array.isArray(data.messages) && data.messages.length > 0) {
          state.messages = data.messages.map(function (m) {
            return {
              id: m.id,
              role: m.role,
              content: m.content,
              sources: m.sources || [],
              feedbackType: m.feedbackType || null,
              feedbackReason: m.feedbackReason || null,
            };
          });
          render();
        }
      } catch (_err) {}
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
        if (state.conversationId) {
          await restoreConversationMessages();
        }
      } catch (err) {
        state.messages.push({
          role: "error",
          content: err.message || "Unable to connect to support service.",
        });
        render();
      }
    }

    async function clearConversation() {
      var oldConvId = state.conversationId;
      state.confirmingClear = false;
      state.messages = [];
      state.conversationId = "";
      state.sessionId = generateUuid();
      state.dislikeReasonOpenForId = null;
      try {
        sessionStorage.setItem(storagePrefix + "session_id", state.sessionId);
        sessionStorage.removeItem(storagePrefix + "conversation_id");
      } catch (_e) {}
      render();

      if (oldConvId && apiKey) {
        try {
          var headers = { "x-api-key": apiKey };
          if (clientId) headers["x-client-id"] = clientId;
          await fetch(baseUrl + "/v1/conversations/" + encodeURIComponent(oldConvId), {
            method: "DELETE",
            headers: headers,
          });
        } catch (_err) {}
      }
      showToast("Conversation cleared");
    }

    async function submitMessageFeedback(msgObj, feedbackType, reason) {
      if (!msgObj || !msgObj.id || !state.conversationId) {
        msgObj.feedbackType = feedbackType;
        if (reason !== null) msgObj.feedbackReason = reason;
        showToast(feedbackType === "like" ? "Thanks for your feedback!" : "Feedback recorded");
        return;
      }
      msgObj.feedbackType = feedbackType;
      if (reason !== null) msgObj.feedbackReason = reason;
      render();

      try {
        var headers = {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        };
        if (clientId) headers["x-client-id"] = clientId;
        var res = await fetchWithRetry(
          baseUrl + "/v1/feedback",
          {
            method: "POST",
            headers: headers,
            body: JSON.stringify({
              conversationId: state.conversationId,
              messageId: msgObj.id,
              feedbackType: feedbackType,
              reason: reason,
            }),
          },
          1
        );
        if (res.ok) {
          showToast(feedbackType === "like" ? "Thanks for your feedback!" : "Feedback saved");
        }
      } catch (_err) {}
    }

    async function sendMessage(text, existingMessageId, isRetry) {
      var userMsgId = existingMessageId || generateUuid();
      if (!isRetry) {
        state.messages.push({ id: userMsgId, role: "user", content: text });
      }
      state.isLoading = true;
      startProcessingStages();
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
          messageId: userMsgId,
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
        stopProcessingStages();

        if (!res.ok) {
          state.messages.push({
            role: "error",
            content: (data && data.error) || "Unable to complete request.",
            retryText: text,
            retryMessageId: userMsgId,
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

        var fullAnswer = String(data.answer || "");
        var assistantMsg = {
          id: data.messageId || generateUuid(),
          role: "assistant",
          content: fullAnswer,
          sources: data.sources || [],
          feedbackType: null,
          feedbackReason: null,
          streaming: false,
        };
        state.messages.push(assistantMsg);

        // If the assistant returned a fallback answer and lead capture is enabled, offer the lead form once
        if (data.isFallback && state.config && state.config.leadCaptureEnabled && !state.leadSubmitted) {
          state.leadFormVisible = true;
        }
        render();
      } catch (_err) {
        state.isLoading = false;
        stopProcessingStages();
        state.messages.push({
          role: "error",
          content: "Network error while sending message. Please try again.",
          retryText: text,
          retryMessageId: userMsgId,
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
          id: generateUuid(),
          role: "assistant",
          content:
            "Thank you, **" +
            leadData.name +
            "**! Your contact details have been received and our team will reach out shortly.",
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
      clearConversation: clearConversation,
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
