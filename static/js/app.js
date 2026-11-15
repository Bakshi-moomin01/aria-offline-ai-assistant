/**
 * EduBot — Frontend JavaScript
 * Handles streaming chat, session state, benchmark runs, and explainability traces.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── State Variables ────────────────────────
  let currentSessionId = "default";
  let activeModel = "llama3.2:latest";
  let isGenerating = false;

  // ── DOM Elements ───────────────────────────
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const modelSelect = document.getElementById("model-select");
  const sessionList = document.getElementById("session-list");
  const newSessionBtn = document.getElementById("new-session-btn");
  const clearHistoryBtn = document.getElementById("clear-history-btn");
  const ollamaStatusDot = document.getElementById("ollama-status-dot");
  const ollamaStatusText = document.getElementById("ollama-status-text");

  // Tab navigation
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Benchmark elements
  const runBenchBtn = document.getElementById("run-bench-btn");
  const viewDashboardBtn = document.getElementById("view-dashboard-btn");
  const benchStatusMsg = document.getElementById("bench-status-msg");
  const benchJsonOutput = document.getElementById("benchmark-json-output");
  const statFastest = document.getElementById("stat-fastest");
  const statMemory = document.getElementById("stat-memory");
  const statQuality = document.getElementById("stat-quality");

  // Explainability elements
  const explainInput = document.getElementById("explain-input");
  const inspectTraceBtn = document.getElementById("inspect-trace-btn");
  const guardrailTraceView = document.getElementById("guardrail-trace-view");
  const promptInfluenceView = document.getElementById("prompt-influence-view");
  const promptChainView = document.getElementById("prompt-chain-view");
  const runBiasAuditBtn = document.getElementById("run-bias-audit-btn");
  const biasAuditView = document.getElementById("bias-audit-view");

  // About hardware info
  const hardwareInfoView = document.getElementById("hardware-info-view");

  // ── Tab Switching ──────────────────────────
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });

  // ── Health & Model Introspection ───────────
  async function checkHealthAndModels() {
    try {
      const res = await fetch("/health");
      const data = await res.json();

      if (data.ollama_online) {
        ollamaStatusDot.className = "status-indicator online";
        ollamaStatusText.textContent = "Ollama Local (Connected)";
      } else {
        ollamaStatusDot.className = "status-indicator offline";
        ollamaStatusText.textContent = "Ollama Disconnected (Run 'ollama serve')";
      }

      // Populate hardware info in About tab
      if (data.hardware) {
        hardwareInfoView.innerHTML = `
          <strong>OS:</strong> ${data.hardware.os} (${data.hardware.os_version})<br>
          <strong>CPU:</strong> ${data.hardware.cpu} (${data.hardware.cpu_cores_physical} cores / ${data.hardware.cpu_cores_logical} threads)<br>
          <strong>Total RAM:</strong> ${(data.hardware.ram_total_mb / 1024).toFixed(1)} GB (Used: ${data.hardware.ram_used_percent}%)<br>
          <strong>GPUs Detected:</strong> ${data.hardware.gpus.length > 0 ? data.hardware.gpus.map(g => `${g.name} (${(g.total_memory_mb/1024).toFixed(1)} GB VRAM)`).join(", ") : "No dedicated GPU detected (CPU mode)"}
        `;
      }

      // Refresh model list
      const mRes = await fetch("/models");
      const mData = await mRes.json();
      if (mData.available_models && mData.available_models.length > 0) {
        modelSelect.innerHTML = "";
        mData.available_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          if (m === activeModel) opt.selected = true;
          modelSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.warn("Health check error:", err);
      ollamaStatusDot.className = "status-indicator offline";
      ollamaStatusText.textContent = "Backend Offline";
    }
  }

  // Model selection change
  modelSelect.addEventListener("change", (e) => {
    activeModel = e.target.value;
  });

  // ── Chat Messaging (Streaming Support) ─────
  async function sendMessage(text) {
    if (!text || isGenerating) return;
    const query = text.trim();
    if (!query) return;

    isGenerating = true;
    chatInput.value = "";
    sendBtn.disabled = true;

    // 1. Append User Message
    appendMessage("user", query);

    // 2. Create Placeholder Assistant Message Bubble
    const assistantBubble = appendMessage("assistant", "...");
    const contentElem = assistantBubble.querySelector(".msg-content");
    contentElem.textContent = "";

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          message: query,
          model_name: activeModel,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        contentElem.innerHTML = formatMarkdown(fullText);
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }
    } catch (err) {
      console.error("Chat streaming error:", err);
      contentElem.textContent = `[Connection Error: ${err.message}. Please verify Ollama is running.]`;
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  }

  function appendMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role === "user" ? "user-msg" : "assistant-msg"}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "user" ? "👤" : "🎓";

    const content = document.createElement("div");
    content.className = "msg-content";
    content.innerHTML = formatMarkdown(text);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  function formatMarkdown(text) {
    // Basic formatting: bold, code blocks, lists
    let formatted = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
    formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");
    formatted = formatted.replace(/\n\n/g, "<br><br>");
    formatted = formatted.replace(/\n- /g, "<br>• ");
    return formatted;
  }

  // Send button & enter key handling
  sendBtn.addEventListener("click", () => sendMessage(chatInput.value));
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value);
    }
  });

  // Quick FAQ prompt buttons
  document.querySelectorAll(".quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q;
      sendMessage(q);
    });
  });

  // ── Session Management ─────────────────────
  newSessionBtn.addEventListener("click", () => {
    const sessionName = prompt("Enter a name for the new session:", `Session ${Date.now().toString().slice(-4)}`);
    if (!sessionName) return;

    currentSessionId = "sess_" + Date.now();
    const li = document.createElement("li");
    li.className = "session-item active";
    li.dataset.id = currentSessionId;
    li.textContent = sessionName;

    document.querySelectorAll(".session-item").forEach((s) => s.classList.remove("active"));
    sessionList.appendChild(li);

    // Reset chat messages
    chatMessages.innerHTML = `
      <div class="message assistant-msg">
        <div class="msg-avatar">🎓</div>
        <div class="msg-content">
          <p>Started a new session: <strong>${sessionName}</strong>. How may I assist you today?</p>
        </div>
      </div>
    `;
  });

  sessionList.addEventListener("click", (e) => {
    if (e.target.classList.contains("session-item")) {
      document.querySelectorAll(".session-item").forEach((s) => s.classList.remove("active"));
      e.target.classList.add("active");
      currentSessionId = e.target.dataset.id;
    }
  });

  clearHistoryBtn.addEventListener("click", async () => {
    if (confirm("Are you sure you want to clear conversation history for this session?")) {
      await fetch(`/sessions/${currentSessionId}`, { method: "DELETE" });
      chatMessages.innerHTML = `
        <div class="message assistant-msg">
          <div class="msg-avatar">🎓</div>
          <div class="msg-content">
            <p>Session memory cleared. Ask me any question about BrightMinds coaching!</p>
          </div>
        </div>
      `;
    }
  });

  // ── Benchmark Suite Handlers ───────────────
  runBenchBtn.addEventListener("click", async () => {
    const checked = Array.from(document.querySelectorAll("input[name='bench-model']:checked")).map(
      (c) => c.value
    );
    if (checked.length === 0) {
      alert("Please select at least one model to benchmark.");
      return;
    }

    runBenchBtn.disabled = true;
    benchStatusMsg.textContent = "Running benchmark on local hardware (this may take a few minutes)...";

    try {
      const res = await fetch("/benchmark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: checked, question_limit: 10 }),
      });
      const data = await res.json();
      benchStatusMsg.textContent = "Benchmark completed! Check summary below.";
      benchJsonOutput.textContent = JSON.stringify(data.summary || data, null, 2);
      loadLatestBenchmarkMetrics();
    } catch (err) {
      console.error("Benchmark execution error:", err);
      benchStatusMsg.textContent = `Benchmark failed: ${err.message}`;
    } finally {
      runBenchBtn.disabled = false;
    }
  });

  viewDashboardBtn.addEventListener("click", () => {
    window.open("/benchmark/report", "_blank");
  });

  async function loadLatestBenchmarkMetrics() {
    try {
      const res = await fetch("/benchmark/results");
      const data = await res.json();
      if (data.summary) {
        benchJsonOutput.textContent = JSON.stringify(data.summary, null, 2);
        // Find fastest model from summary
        let fastestModel = "-";
        let lowestLatency = Infinity;
        let bestQuality = "-";
        let maxQuality = -1;

        for (const [m, stats] of Object.entries(data.summary)) {
          if (stats.avg_latency_ms && stats.avg_latency_ms < lowestLatency) {
            lowestLatency = stats.avg_latency_ms;
            fastestModel = `${m} (${Math.round(lowestLatency)} ms)`;
          }
          if (stats.avg_quality_score && stats.avg_quality_score > maxQuality) {
            maxQuality = stats.avg_quality_score;
            bestQuality = `${m} (${maxQuality.toFixed(2)}/5.0)`;
          }
        }

        statFastest.textContent = fastestModel;
        statQuality.textContent = bestQuality;
        statMemory.textContent = "Logged in CSV";
      }
    } catch (err) {
      console.log("No previous benchmark loaded:", err);
    }
  }

  // ── Explainability Handlers ────────────────
  inspectTraceBtn.addEventListener("click", async () => {
    const q = explainInput.value.trim();
    if (!q) return;

    inspectTraceBtn.disabled = true;
    guardrailTraceView.textContent = "Inspecting...";
    promptInfluenceView.textContent = "Analyzing...";
    promptChainView.textContent = "Decomposing prompt chain...";

    try {
      const res = await fetch("/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          session_id: currentSessionId,
          model_name: activeModel,
        }),
      });

      const trace = await res.json();

      // Guardrail status
      const g = trace.guardrail_trace;
      const isAccepted = g.decision === "accepted";
      guardrailTraceView.innerHTML = `
        <p><strong>Decision:</strong> <span class="badge-tag ${isAccepted ? 'badge-success' : 'badge-danger'}">${g.decision.toUpperCase()}</span></p>
        <p><strong>Reason:</strong> ${g.reason || "Passed all scope checks."}</p>
        <p><strong>Blocked Keyword Hit:</strong> ${g.blocked_keyword_found ? "YES" : "NO"}</p>
        <p><strong>LLM Semantic Classifier:</strong> ${g.llm_classifier_result || "N/A"}</p>
      `;

      // Prompt influences & decision points
      promptInfluenceView.innerHTML = `
        <p><strong>Influences:</strong></p>
        <ul>${trace.system_prompt_influence.map((inf) => `<li>${inf}</li>`).join("")}</ul>
        <br>
        <p><strong>Decision Points:</strong></p>
        <ul>${trace.decision_points.map((dp) => `<li>${dp}</li>`).join("")}</ul>
      `;

      // Full prompt chain
      promptChainView.textContent = trace.full_prompt_chain;
    } catch (err) {
      console.error("Trace inspection error:", err);
      guardrailTraceView.textContent = "Error: " + err.message;
    } finally {
      inspectTraceBtn.disabled = false;
    }
  });

  runBiasAuditBtn.addEventListener("click", async () => {
    runBiasAuditBtn.disabled = true;
    biasAuditView.textContent = "Executing demographic invariance test suite on local model...";

    try {
      const res = await fetch(`/explain/audit?model_name=${encodeURIComponent(activeModel)}`);
      const data = await res.json();

      let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
        <thead>
          <tr style="border-bottom: 2px solid #e2e8f0; text-align: left;">
            <th style="padding: 6px;">Variant Test Case</th>
            <th style="padding: 6px;">Similarity</th>
            <th style="padding: 6px;">Fairness Flag</th>
          </tr>
        </thead>
        <tbody>`;

      data.bias_tests.forEach((b) => {
        html += `
          <tr style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 8px;"><strong>${b.variant_label}</strong><br><small style="color: #64748b;">${b.variant_query}</small></td>
            <td style="padding: 8px;">${b.similarity_score}</td>
            <td style="padding: 8px;"><span class="badge-tag ${b.flag.startsWith('PASS') ? 'badge-success' : 'badge-warning'}">${b.flag}</span></td>
          </tr>
        `;
      });

      html += "</tbody></table>";
      biasAuditView.innerHTML = html;
    } catch (err) {
      console.error("Bias audit error:", err);
      biasAuditView.textContent = "Error running bias audit: " + err.message;
    } finally {
      runBiasAuditBtn.disabled = false;
    }
  });

  // ── Initial Boot ───────────────────────────
  checkHealthAndModels();
  loadLatestBenchmarkMetrics();
});
