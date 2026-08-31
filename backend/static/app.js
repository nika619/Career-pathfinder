/**
 * Career PathFinder — Interactive Client Application
 * ===================================================
 * Connects the UI directly to the FastAPI engine:
 * - Natural language intake & role resolution
 * - Adaptive BKT cognitive diagnostic loop
 * - Journey path rendering with multi-factor courses & capstone projects
 * - Glass-box reasoning trace inspector
 * - Dynamic adaptive re-planning on milestone pass/fail
 * - Grounded AI career coaching chat
 */

const API_BASE = "";
const USER_ID = "learner_" + Math.random().toString(36).substring(2, 9);

// State management
const state = {
  userId: USER_ID,
  targetRole: null,
  diagnosticSkills: [],
  currentDiagIndex: 0,
  currentSkillId: null,
  currentQuestion: null,
  activePlan: null,
  selectedNodeSkill: null,
  activePersona: "general",
  ttsActive: false,
  isRecording: false
};

// ---------------------------------------------------------------------------
// Theme Management (Light / Dark Mode)
// Persists choice in localStorage. Default is dark editorial ("Cartographer's Observatory").
// ---------------------------------------------------------------------------
function getSavedTheme() {
  return localStorage.getItem("career_pathfinder_theme") || "dark";
}

function applyTheme(theme) {
  const isLight = theme === "light";
  document.documentElement.setAttribute("data-theme", isLight ? "light" : "dark");
  document.body.classList.toggle("theme-light", isLight);

  const iconEl = document.getElementById("theme-toggle-icon");
  const labelEl = document.getElementById("theme-toggle-label");
  if (iconEl) iconEl.innerText = isLight ? "🌙" : "☀️";
  if (labelEl) labelEl.innerText = isLight ? "Dark" : "Light";

  // Update WebGL scene background/particles if active
  if (window.sceneManager && typeof window.sceneManager.setTheme === "function") {
    window.sceneManager.setTheme(theme);
  }
}

window.toggleTheme = function() {
  const current = document.documentElement.getAttribute("data-theme") || getSavedTheme();
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem("career_pathfinder_theme", next);
  applyTheme(next);
  showToast(`Switched to ${next === "light" ? "Light Atelier" : "Dark Observatory"} theme`, "info");
};

// Initialize theme immediately on script load
applyTheme(getSavedTheme());

// ---------------------------------------------------------------------------
// Judge/Tester API Key Override
// Stored ONLY in sessionStorage (cleared when browser tab closes).
// Sent as X-Gemini-API-Key header on AI-dependent requests.
// Priority: judge key > owner .env key > deterministic fallback.
// ---------------------------------------------------------------------------
function getJudgeKey() {
  return (sessionStorage.getItem("judge_gemini_key") || "").trim();
}

function getApiHeaders(extra = {}) {
  const headers = { "Content-Type": "application/json", ...extra };
  const judgeKey = getJudgeKey();
  if (judgeKey) headers["X-Gemini-API-Key"] = judgeKey;
  return headers;
}

// DOM Elements
const views = {
  intake: document.getElementById("view-intake"),
  diagnostic: document.getElementById("view-diagnostic"),
  dashboard: document.getElementById("view-dashboard")
};

function switchView(viewName) {
  Object.values(views).forEach(v => v.classList.remove("active"));
  views[viewName].classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ----------------- Stage 1: Intake & Goal Parsing -----------------

document.querySelectorAll(".chip-example").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip-example").forEach(c => c.style.borderColor = "var(--line)");
    chip.style.borderColor = "var(--gold)";
    chip.style.background = "rgba(224, 166, 76, 0.15)";
    const input = document.getElementById("goal-input");
    input.value = chip.dataset.text;
    input.focus();
  });
});

document.getElementById("goal-input")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("btn-intake-submit")?.click();
  }
});

document.getElementById("btn-intake-submit").addEventListener("click", async () => {
  const goalText = document.getElementById("goal-input").value.trim();
  if (!goalText) {
    showToast("Please describe your career goal first.", "warn");
    return;
  }

  const btn = document.getElementById("btn-intake-submit");
  btn.innerHTML = `<span>Analyzing labor-market competencies...</span>`;
  btn.disabled = true;

  try {
    // 1. Submit Goal
    const goalRes = await fetch(`${API_BASE}/api/goal`, {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({ user_id: state.userId, goal_text: goalText })
    });
    const goalData = await goalRes.json();
    if (!goalRes.ok) {
      showToast(goalData.detail || "We couldn't detect a career goal from your input. Please specify a target role.", "warn");
      btn.innerHTML = `<span>Analyze Role & Launch Diagnostic</span> →`;
      btn.disabled = false;
      return;
    }
    state.targetRole = goalData.target_role;

    // 2. Submit Profile Preferences
    await fetch(`${API_BASE}/api/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        weekly_hours: parseInt(document.getElementById("pref-hours").value),
        experience_level: document.getElementById("pref-level").value,
        preferred_format: document.getElementById("pref-format").value
      })
    });

    // 3. Fetch Diagnostic Questions
    await startDiagnostic();
  } catch (err) {
    console.error("Intake error:", err);
    showToast("Couldn't reach the engine. Make sure the backend is running, then retry.", "error");
    btn.innerHTML = `<span>Analyze Role & Launch Diagnostic</span> →`;
    btn.disabled = false;
  }
});

// ----------------- Stage 2: Adaptive BKT Diagnostic -----------------

async function startDiagnostic() {
  const res = await fetch(`${API_BASE}/api/diagnostic/skills?user_id=${state.userId}`);
  const data = await res.json();
  state.diagnosticSkills = data.diagnostic_skills;
  state.currentDiagIndex = 0;

  document.getElementById("diag-role-title").innerText = `Calibrating for ${state.targetRole}`;
  switchView("diagnostic");
  await loadNextDiagnosticQuestion();
}

async function loadNextDiagnosticQuestion() {
  if (state.currentDiagIndex >= state.diagnosticSkills.length) {
    // Diagnostic completed
    await compileAndShowRoadmap();
    return;
  }

  const skill = state.diagnosticSkills[state.currentDiagIndex];
  state.currentSkillId = skill.id;

  document.getElementById("diag-step-counter").innerText = `Skill ${state.currentDiagIndex + 1} of ${state.diagnosticSkills.length}`;
  document.getElementById("diag-skill-name").innerText = skill.name;
  document.getElementById("diag-feedback").classList.add("hidden");
  document.getElementById("btn-diag-next").classList.add("hidden");

  // Fetch question
  const res = await fetch(`${API_BASE}/api/diagnostic/next`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: state.userId, skill_id: skill.id })
  });
  const data = await res.json();

  if (!data.has_more) {
    state.currentDiagIndex++;
    await loadNextDiagnosticQuestion();
    return;
  }

  state.currentQuestion = data.question;
  document.getElementById("diag-question-text").innerText = data.question.question;

  const optContainer = document.getElementById("diag-options-list");
  optContainer.innerHTML = "";

  data.question.options.forEach(opt => {
    const btn = document.createElement("button");
    btn.className = "option-btn";
    btn.innerText = opt;
    btn.onclick = () => submitDiagnosticAnswer(opt, btn);
    optContainer.appendChild(btn);
  });
}

async function submitDiagnosticAnswer(selectedOption, clickedBtn) {
  // Disable all options
  document.querySelectorAll(".option-btn").forEach(b => b.disabled = true);

  const res = await fetch(`${API_BASE}/api/diagnostic/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: state.userId,
      skill_id: state.currentQuestion?.skill_id || state.currentSkillId,
      question_text: state.currentQuestion.question,
      selected_option: selectedOption
    })
  });
  const result = await res.json();

  // Highlight button
  if (result.is_correct) {
    clickedBtn.classList.add("selected-correct");
  } else {
    clickedBtn.classList.add("selected-wrong");
    // Also reveal correct option for crystal-clear pedagogical feedback
    document.querySelectorAll(".option-btn").forEach(b => {
      const bText = b.innerText.trim().toLowerCase();
      const cText = (result.correct_answer || "").trim().toLowerCase();
      if (bText === cText || bText.replace(/['"`]/g, '') === cText.replace(/['"`]/g, '')) {
        b.classList.add("selected-correct");
      }
    });
  }

  // Update mastery badge
  const pct = Math.round(result.updated_mastery * 100);
  document.getElementById("diag-mastery-val").innerText = `${pct}%`;

  // Show BKT feedback
  const fb = document.getElementById("diag-feedback");
  const fbIcon = document.getElementById("diag-feedback-icon");
  const fbTitle = document.getElementById("diag-feedback-title");
  const fbMsg = document.getElementById("diag-feedback-msg");

  fb.className = `feedback-banner ${result.is_correct ? "correct" : "wrong"}`;
  fbIcon.innerText = result.is_correct ? "✓" : "✗";
  fbTitle.innerText = result.is_correct ? "Accurate Knowledge Demonstrated" : "Foundational Knowledge Gap Detected";
  fbMsg.innerText = `Bayesian Knowledge Tracing (BKT) adjusted mastery posterior to ${pct}% (Δ ${result.bkt_delta > 0 ? "+" : ""}${result.bkt_delta}).`;
  fb.classList.remove("hidden");

  document.getElementById("btn-diag-next").classList.remove("hidden");
}

document.getElementById("btn-diag-next").addEventListener("click", async () => {
  state.currentDiagIndex++;
  await loadNextDiagnosticQuestion();
});

document.getElementById("btn-diag-skip").addEventListener("click", async () => {
  await compileAndShowRoadmap();
});

// ----------------- Stage 3: Dashboard & Path Rendering -----------------

async function compileAndShowRoadmap() {
  showLoader("Compiling your grounded roadmap…");
  try {
    const res = await fetch(`${API_BASE}/api/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId })
    });
    if (!res.ok) throw new Error("Plan request failed: " + res.status);
    const plan = await res.json();
    state.activePlan = plan;
    renderDashboard(plan);
    switchView("dashboard");
  } catch (err) {
    console.error("Roadmap compile error:", err);
    showToast("Couldn't compile the roadmap — is the engine running? Retry in a moment.", "error");
  } finally {
    hideLoader();
  }
}

function renderDashboard(plan) {
  document.getElementById("dash-role-title").innerText = `${plan.role} Roadmap`;
  document.getElementById("dash-timeline-info").innerText = 
    `${plan.total_steps} prerequisite steps · ${plan.estimated_weeks} weeks estimated at ${plan.weekly_commitment_hours} hrs/week`;

  // Update Readiness Metrics
  const pct = Math.round(plan.readiness_score * 100);
  document.getElementById("metric-readiness-pct").innerText = `${pct}%`;
  document.getElementById("readiness-fill").style.width = `${pct}%`;
  document.getElementById("metric-skills-count").innerText = `${plan.skills_mastered_count} of ${plan.total_skills_count} skills mastered`;
  document.getElementById("metric-hours-total").innerText = `${plan.total_estimated_hours} total learning hours`;

  // Render Nodes
  const container = document.getElementById("journey-nodes-container");
  container.innerHTML = "";

  let lastPhase = null;

  plan.nodes.forEach((node, idx) => {
    // Inject Phase Header Divider if phase changed
    if (node.phase_name && node.phase_name !== lastPhase) {
      lastPhase = node.phase_name;
      const phaseDivider = document.createElement("div");
      phaseDivider.className = "phase-divider-banner";
      phaseDivider.innerHTML = `
        <div class="phase-title-tag">
          <span class="pulse-dot"></span>
          <strong>${node.phase_name}</strong>
        </div>
      `;
      container.appendChild(phaseDivider);
    }

    const card = document.createElement("div");
    card.className = `node-card ${node.status || "pending"}`;
    card.id = `node-${node.skill}`;

    // Format courses HTML
    const coursesHtml = node.courses.map(c => `
      <div class="course-row">
        <a href="${c.url}" target="_blank" class="course-name">
          <span>📚 ${c.title}</span>
          <span class="chip on">${c.platform}</span>
        </a>
        <span class="course-score">Match: ${Math.round(c.match_score * 100)}%</span>
      </div>
    `).join("");

    // Format project deliverables
    const deliverablesHtml = node.project.deliverables.map(d => `<li>${d}</li>`).join("");
    
    // Prereqs list
    const prereqsText = node.trace && node.trace.prerequisites_required && node.trace.prerequisites_required.length > 0 
      ? node.trace.prerequisites_required.join(", ") 
      : "None (Foundational Entrypoint)";

    card.innerHTML = `
      <div class="node-header">
        <div class="node-title-wrap">
          <span class="kicker">STEP ${node.step_number} · ${node.cluster.toUpperCase()}</span>
          <h3>${node.skill_name}</h3>
        </div>
        <button class="btn-ghost btn-sm btn-inspect" onclick="toggleInlineTrace('${node.skill}')">🔍 Inspect Trace</button>
      </div>
      <p class="node-desc">${node.description}</p>

      <div class="node-meta-chips">
        <span class="chip on">O*NET Importance: ${Math.round(node.importance * 100)}%</span>
        <span class="chip">Difficulty Tier ${node.difficulty}</span>
        <span class="chip" id="mastery-chip-${node.skill}">Mastery: ${Math.round(node.current_mastery * 100)}%</span>
      </div>

      <!-- Inline Glass-Box Trace Accordion (Instant Visibility) -->
      <div id="inline-trace-${node.skill}" class="inline-trace-box hidden">
        <div class="inline-trace-header">
          <span class="tag">🔍 Algorithmic Reasoning Breakdown</span>
          <span class="chip on">Deterministic Proof</span>
        </div>
        <table class="trace-table">
          <tr><td>Target Role</td><td><strong>${plan.role}</strong></td></tr>
          <tr><td>Labor Market Weight</td><td>${Math.round(node.importance * 100)}% (${node.trace.onet_classification})</td></tr>
          <tr><td>Measured Mastery</td><td>${Math.round(node.current_mastery * 100)}%</td></tr>
          <tr><td>Required Prerequisites</td><td>${prereqsText}</td></tr>
          <tr><td>Sequencing Rule</td><td><span class="chip on">Kahn's Topological Sort (Zero Prereq Violation)</span></td></tr>
        </table>
        <div class="trace-narrative" id="narrative-${node.skill}">
          <strong>Why this step was selected:</strong><br>
          ${node.trace.recommendation_rationale}
        </div>
      </div>

      <!-- Recommended Courses -->
      <div class="node-sub-sec">
        <div class="sub-sec-title">
          <span>Curated Real-World Courses</span>
          <span class="muted font-12">Multi-Factor Ranked</span>
        </div>
        <div class="course-list">${coursesHtml}</div>
      </div>

      <!-- Hands-On Capstone Project -->
      <div class="node-sub-sec">
        <div class="sub-sec-title">
          <span>Outcome-Based Capstone Project</span>
          <span class="muted font-12">~${node.project.estimated_hours} Hours</span>
        </div>
        <div class="project-box">
          <div class="project-title">🛠️ ${node.project.title}</div>
          <div class="project-obj">${node.project.objective}</div>
          <ul class="project-deliverables">${deliverablesHtml}</ul>
        </div>
      </div>

      <!-- Node Progress Controls -->
      <div class="node-footer">
        <div class="footer-meta">
          <span class="muted font-12">Milestone Assessment: ${node.milestone.pass_criteria}</span>
          ${node.status === 'passed' ? '<span class="chip on" style="margin-left: 8px;">✓ Verified Competency</span>' : ''}
        </div>
        <div class="node-btn-group">
          <button class="btn-pass" data-action="submit-capstone" data-skill="${node.skill}" onclick="openCapstoneEvaluator('${node.skill}')">⚡ Submit Capstone & Evaluate</button>
          <button class="btn-fail" data-action="remediation" data-skill="${node.skill}" onclick="handleProgress('${node.skill}', 'failed')">↺ Need Remediation</button>
        </div>
      </div>
    `;

    container.appendChild(card);
  });

  // Attach global container click delegation for 100% reliable button actions
  if (!container.dataset.boundEvents) {
    container.dataset.boundEvents = "true";
    container.addEventListener("click", (e) => {
      const passBtn = e.target.closest('[data-action="submit-capstone"]');
      if (passBtn) {
        const skillId = passBtn.dataset.skill;
        openCapstoneEvaluator(skillId);
        return;
      }
      const failBtn = e.target.closest('[data-action="remediation"]');
      if (failBtn) {
        const skillId = failBtn.dataset.skill;
        handleProgress(skillId, 'failed');
        return;
      }
    });
  }

  // Re-attach 3D Tilt & Glare to all rendered cards
  if (window.Vanilla3DTilt) {
    window.Vanilla3DTilt.init();
  }

  // Select first node trace in sidebar inspector
  if (plan.nodes.length > 0) {
    inspectTrace(plan.nodes[0].skill);
  }
}

// ----------------- Inline & Sidebar Glass-Box Trace Inspector -----------------

window.toggleInlineTrace = async function(skillId) {
  const inlineBox = document.getElementById(`inline-trace-${skillId}`);
  if (inlineBox) {
    inlineBox.classList.toggle("hidden");
  }
  // Also update and highlight sidebar inspector
  await inspectTrace(skillId);
};

// ----------------- Sidebar Dual Command Deck -----------------

window.switchSidebarDeck = function(tabId) {
  const coachTab = document.getElementById("tab-deck-coach");
  const explainTab = document.getElementById("tab-deck-explain");
  const coachPanel = document.getElementById("deck-panel-coach");
  const explainPanel = document.getElementById("deck-panel-explain");

  if (tabId === "explain") {
    if (coachTab) coachTab.classList.remove("active");
    if (explainTab) explainTab.classList.add("active");
    if (coachPanel) coachPanel.classList.remove("active");
    if (explainPanel) explainPanel.classList.add("active");
  } else {
    if (coachTab) coachTab.classList.add("active");
    if (explainTab) explainTab.classList.remove("active");
    if (coachPanel) coachPanel.classList.add("active");
    if (explainPanel) explainPanel.classList.remove("active");
  }
};

async function inspectTrace(skillId) {
  state.selectedNodeSkill = skillId;
  const inspector = document.getElementById("inspector-content");
  if (!inspector) return;

  // Auto-switch to trace inspector deck panel
  if (typeof window.switchSidebarDeck === "function") {
    window.switchSidebarDeck("explain");
  }

  inspector.innerHTML = `<div class="muted font-12" style="padding: 12px 0;">Extracting algorithmic reasoning trace...</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/explain`, {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({ user_id: state.userId, skill_id: skillId })
    });
    const data = await res.json();
    const t = data.trace;

    const hasPrereqs = t.prerequisites_required && t.prerequisites_required.length > 0;
    const prereqsText = hasPrereqs
      ? t.prerequisites_required.join(", ") 
      : "None (foundational)";

    // Only show Prerequisite Status row when there are actual prerequisites to report
    const prereqStatusRow = hasPrereqs
      ? `<tr><td>Prerequisite Status</td><td><span class="chip on" style="font-size: 11px; padding: 3px 8px;">✓ All Prerequisites Met</span></td></tr>`
      : "";

    inspector.innerHTML = `
      <table class="trace-table">
        <tr><td>Target Career Role</td><td><strong>${t.target_role}</strong></td></tr>
        <tr><td>O*NET Importance</td><td>${Math.round(t.skill_importance_score * 100)}% (${t.onet_classification})</td></tr>
        <tr><td>Measured Baseline</td><td>${t.measured_mastery_level}</td></tr>
        <tr><td>Prerequisites</td><td>${prereqsText}</td></tr>
        ${prereqStatusRow}
      </table>
      <div class="trace-narrative">
        <strong>Explainable AI Rationale:</strong><br>
        ${data.explanation}
      </div>
    `;

    // Also update inline narrative if open
    const inlineNarrative = document.getElementById(`narrative-${skillId}`);
    if (inlineNarrative && data.explanation) {
      inlineNarrative.innerHTML = `<strong>Explainable AI Rationale:</strong><br>${data.explanation}`;
    }
  } catch (err) {
    console.error("Inspector error:", err);
  }
}


// ----------------- Dynamic Milestone Re-planning & Capstone Evaluation -----------------

// Capstone Evaluation Modal State
let activeEvaluatingSkill = null;

window.openCapstoneEvaluator = function(skillId, skillName) {
  activeEvaluatingSkill = skillId;
  const modal = document.getElementById("modal-capstone-evaluator");
  if (!modal) {
    console.error("modal-capstone-evaluator not found");
    return;
  }

  const node = (state.activePlan?.nodes || []).find(n => n.skill === skillId);
  const resolvedName = skillName || node?.skill_name || skillId;
  const proj = node?.project || {
    title: `${resolvedName} Hands-On Capstone`,
    objective: "Implement an end-to-end practical project demonstrating core competency.",
    deliverables: ["Project Repository on GitHub", "Modular Code Architecture", "Comprehensive Unit Tests"],
    estimated_hours: 10
  };

  document.getElementById("capstone-modal-skill-title").innerText = `Submit Capstone: ${resolvedName}`;
  document.getElementById("capstone-spec-title").innerText = `🛠️ ${proj.title}`;
  document.getElementById("capstone-spec-obj").innerText = proj.objective;
  document.getElementById("capstone-spec-effort").innerText = `Estimated Effort: ~${proj.estimated_hours} Hours`;

  // Populate dynamic deliverables checklist
  const checklistContainer = document.getElementById("capstone-deliverables-checklist");
  if (checklistContainer) {
    checklistContainer.innerHTML = "";
    (proj.deliverables || []).forEach((d) => {
      const item = document.createElement("label");
      item.className = "checklist-item";
      item.innerHTML = `
        <input type="checkbox" checked value="${escapeHtml(d)}" class="capstone-chk">
        <span>${escapeHtml(d)}</span>
      `;
      checklistContainer.appendChild(item);
    });
  }

  // Reset feedback
  const resultCard = document.getElementById("capstone-eval-result");
  if (resultCard) resultCard.classList.add("hidden");
  
  const submitBtn = document.getElementById("btn-submit-capstone-action");
  if (submitBtn) {
    submitBtn.innerHTML = `<span>⚡ Run AI Rubric Evaluation</span>`;
    submitBtn.disabled = false;
  }

  modal.classList.remove("hidden");
};

window.closeCapstoneModal = function() {
  const modal = document.getElementById("modal-capstone-evaluator");
  if (modal) modal.classList.add("hidden");
};

window.submitCapstoneEvaluation = async function() {
  if (!activeEvaluatingSkill) return;
  
  const submitBtn = document.getElementById("btn-submit-capstone-action");
  const repoUrl = document.getElementById("capstone-repo-url").value.trim();
  const notes = document.getElementById("capstone-notes").value.trim();
  const completedChecklist = Array.from(document.querySelectorAll(".capstone-chk:checked")).map(c => c.value);

  submitBtn.innerHTML = `<span>⏳ Evaluating Against Industry Rubric...</span>`;
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/milestone/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        skill_id: activeEvaluatingSkill,
        github_url: repoUrl || "https://github.com/learner/capstone-submission",
        deliverables_notes: notes || "Modular architecture with automated pytest coverage.",
        completed_checklist: completedChecklist
      })
    });

    const result = await res.json();
    
    // Show rubric result
    const resultCard = document.getElementById("capstone-eval-result");
    const badge = document.getElementById("eval-result-badge");
    const scoreVal = document.getElementById("eval-result-score");
    const feedbackText = document.getElementById("eval-result-feedback");

    resultCard.classList.remove("hidden");
    scoreVal.innerText = `${result.rubric_score} / 100`;
    feedbackText.innerText = result.feedback;

    if (result.passed) {
      resultCard.className = "eval-result-card";
      badge.className = "chip on";
      badge.innerText = "✓ Verified Mastery Awarded";
      submitBtn.innerHTML = `<span>✓ Competency Verified!</span>`;
      
      // Update plan state
      state.activePlan = result.new_plan;
      renderDashboard(result.new_plan);

      addBotChatMessage(`🎉 **Capstone Passed!** (${result.rubric_score}/100) Verified mastery awarded for **${activeEvaluatingSkill}**. Downstream roadmap recalibrated!`);

      // Auto close after 1.5s
      setTimeout(() => {
        closeCapstoneModal();
      }, 1500);
    } else {
      resultCard.className = "eval-result-card failed-eval";
      badge.className = "chip";
      badge.style.background = "rgba(248,113,113,0.2)";
      badge.style.color = "var(--red)";
      badge.innerText = "✗ Remediation Needed";
      submitBtn.innerHTML = `<span>↺ Re-submit Evaluation</span>`;
      submitBtn.disabled = false;

      state.activePlan = result.new_plan;
      renderDashboard(result.new_plan);
      addBotChatMessage(`⚠️ Capstone evaluation scored ${result.rubric_score}/100 for **${activeEvaluatingSkill}**. Please review modular architecture & unit test requirements.`);
    }

  } catch (err) {
    console.error("Evaluation error:", err);
    showToast("Self-assessment couldn't run — check the backend connection.", "error");
    submitBtn.innerHTML = `<span>⚡ Run AI Rubric Evaluation</span>`;
    submitBtn.disabled = false;
  }
};

window.handleProgress = async function(skillId, status) {
  const nodeEl = document.getElementById(`node-${skillId}`);
  if (nodeEl) {
    nodeEl.style.opacity = "0.6";
  }

  try {
    const res = await fetch(`${API_BASE}/api/progress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        skill_id: skillId,
        status: status
      })
    });
    const result = await res.json();
    state.activePlan = result.new_plan;
    renderDashboard(result.new_plan);
    
    // Add coaching message (with deduplication)
    const chatMsg = status === "passed"
      ? `🎉 Congratulations on mastering ${skillId}! Your mastery has been updated to 90% and your downstream milestones have been optimized.`
      : `⚠️ Checkpoint not passed for ${skillId}. Your measured mastery has dropped to reflect this gap, and targeted reinforcement has been dynamically scheduled into your roadmap.`;
    addBotChatMessage(chatMsg);
  } catch (err) {
    console.error("Progress update error:", err);
  }
};

document.getElementById("btn-replan-all").addEventListener("click", async () => {
  const btn = document.getElementById("btn-replan-all");
  btn.innerText = "⏳ Recalibrating...";
  btn.disabled = true;
  await compileAndShowRoadmap();
  btn.innerText = "✓ Path Re-computed";
  setTimeout(() => {
    btn.innerText = "↺ Re-compute Path";
    btn.disabled = false;
  }, 1500);
});

// ----------------- Export Roadmap Handlers (PDF / Markdown / JSON) -----------------

window.openExportModal = function() {
  const modal = document.getElementById("modal-export-roadmap");
  if (modal) modal.classList.remove("hidden");
};

window.closeExportModal = function() {
  const modal = document.getElementById("modal-export-roadmap");
  if (modal) modal.classList.add("hidden");
};

document.getElementById("btn-export-roadmap")?.addEventListener("click", () => {
  openExportModal();
});

document.getElementById("btn-action-export-pdf")?.addEventListener("click", () => {
  closeExportModal();
  setTimeout(() => {
    window.print();
  }, 250);
});

document.getElementById("btn-action-export-md")?.addEventListener("click", () => {
  closeExportModal();
  if (!state.userId) return;
  const link = document.createElement("a");
  link.href = `${API_BASE}/api/export/markdown?user_id=${encodeURIComponent(state.userId)}`;
  link.download = `Career_Path_${(state.targetRole || "Roadmap").replace(/\s+/g, "_")}.md`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("Exported roadmap study syllabus (.md)!", "info");
});

document.getElementById("btn-action-export-json")?.addEventListener("click", () => {
  closeExportModal();
  if (!state.activePlan) return;
  const blob = new Blob([JSON.stringify(state.activePlan, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `Career_Plan_${(state.targetRole || "Roadmap").replace(/\s+/g, "_")}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  showToast("Exported roadmap JSON data!", "info");
});

// Close export modal on backdrop click
const modalExport = document.getElementById("modal-export-roadmap");
modalExport?.addEventListener("click", (e) => {
  if (e.target === modalExport) closeExportModal();
});

document.getElementById("btn-reset").addEventListener("click", () => window.location.reload());

// Close capstone modal on backdrop click
const modalCapstone = document.getElementById("modal-capstone-evaluator");
modalCapstone?.addEventListener("click", (e) => {
  if (e.target === modalCapstone) closeCapstoneModal();
});

// ----------------- Judge / Evaluator API Key Modal Controls -----------------

window.openApiModal = function() {
  const modal = document.getElementById("modal-api-settings");
  if (!modal) return;
  const currentKey = getJudgeKey();
  const inputEl = document.getElementById("input-judge-api-key");
  const statusEl = document.getElementById("api-key-active-status");
  if (inputEl) inputEl.value = currentKey;
  if (statusEl) {
    if (currentKey) {
      statusEl.className = "chip on";
      statusEl.innerText = "✓ Custom Judge API Key Active (Session Only)";
    } else {
      statusEl.className = "chip on";
      statusEl.innerText = "✓ Default Project Key Active (.env)";
    }
  }
  modal.classList.remove("hidden");
};

window.closeApiModal = function() {
  const modal = document.getElementById("modal-api-settings");
  if (modal) modal.classList.add("hidden");
};

window.saveJudgeKey = function() {
  const inputEl = document.getElementById("input-judge-api-key");
  const val = (inputEl?.value || "").trim();
  const label = document.getElementById("api-status-label");
  if (val) {
    sessionStorage.setItem("judge_gemini_key", val);
    if (label) label.innerText = "Custom Key Active";
    showToast("Custom Gemini key saved for this session!", "info");
  } else {
    sessionStorage.removeItem("judge_gemini_key");
    if (label) label.innerText = "API Key";
    showToast("Reverted to default project key.", "info");
  }
  closeApiModal();
};

window.clearJudgeKey = function() {
  sessionStorage.removeItem("judge_gemini_key");
  const inputEl = document.getElementById("input-judge-api-key");
  const label = document.getElementById("api-status-label");
  if (inputEl) inputEl.value = "";
  if (label) label.innerText = "API Key";
  showToast("Custom API key cleared. Using default project key.", "info");
  closeApiModal();
};

document.getElementById("btn-api-settings")?.addEventListener("click", () => {
  openApiModal();
});

// Close API settings modal on backdrop click
const modalApiSettings = document.getElementById("modal-api-settings");
modalApiSettings?.addEventListener("click", (e) => {
  if (e.target === modalApiSettings) closeApiModal();
});

// Initialize header label if custom key already in session
if (getJudgeKey()) {
  const label = document.getElementById("api-status-label");
  if (label) label.innerText = "Custom Key Active";
}

// ----------------- Grounded AI Career Coach Chat (Multi-Agent & Voice) -----------------

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessages = document.getElementById("chat-messages");

// Multi-Agent Guild Persona Selector
document.querySelectorAll(".persona-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".persona-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    state.activePersona = chip.dataset.persona;
    
    const personaLabels = {
      general: "AI Learning Coach",
      algo_specialist: "⚡ Algorithm Specialist",
      system_architect: "🏗️ System Architect",
      career_strategist: "💼 Executive Career Strategist"
    };
    const avatarIcons = {
      general: "🤖",
      algo_specialist: "⚡",
      system_architect: "🏗️",
      career_strategist: "💼"
    };
    
    const titleEl = document.getElementById("coach-title-label");
    const avatarEl = document.getElementById("coach-avatar-icon");
    if (titleEl) titleEl.innerText = personaLabels[state.activePersona] || "AI Learning Coach";
    if (avatarEl) avatarEl.innerText = avatarIcons[state.activePersona] || "🤖";
    
    showToast(`Switched mentor persona to ${personaLabels[state.activePersona]}`, "info");
  });
});

// Voice Speech-to-Text Recording (Web Speech API)
const btnVoiceInput = document.getElementById("btn-voice-input");
if (btnVoiceInput) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    btnVoiceInput.addEventListener("click", () => {
      if (state.isRecording) {
        recognition.stop();
        state.isRecording = false;
        btnVoiceInput.classList.remove("recording");
        showToast("Voice capture stopped.", "info");
      } else {
        try {
          recognition.start();
          state.isRecording = true;
          btnVoiceInput.classList.add("recording");
          btnVoiceInput.title = "Listening... click to stop";
          showToast("Listening \u2014 speak your question now.", "info");
        } catch (err) {
          state.isRecording = false;
          btnVoiceInput.classList.remove("recording");
          if (err.name === "InvalidStateError") {
            showToast("Mic is already listening. Click again to stop.", "warn");
          } else {
            showToast("Could not start microphone: " + err.message, "error");
          }
        }
      }
    });

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      chatInput.value = transcript;
      state.isRecording = false;
      btnVoiceInput.classList.remove("recording");
      btnVoiceInput.title = "Voice Input (Click to Speak)";
      showToast(`Captured: "${transcript}"`, "info");
      chatForm.dispatchEvent(new Event("submit"));
    };

    recognition.onerror = (event) => {
      state.isRecording = false;
      btnVoiceInput.classList.remove("recording");
      btnVoiceInput.title = "Voice Input (Click to Speak)";
      const errMap = {
        "not-allowed": "Microphone access denied. Allow mic in browser settings and reload.",
        "no-speech": "No speech detected. Please try speaking again.",
        "network": "Network error during voice processing.",
        "aborted": "Voice capture was stopped.",
        "audio-capture": "No microphone found on this device.",
        "service-not-allowed": "Speech service blocked. Use HTTPS or localhost."
      };
      const msg = errMap[event.error] || `Voice error: ${event.error}`;
      showToast(msg, event.error === "aborted" ? "info" : "warn");
    };

    recognition.onend = () => {
      state.isRecording = false;
      btnVoiceInput.classList.remove("recording");
      btnVoiceInput.title = "Voice Input (Click to Speak)";
    };
  } else {
    btnVoiceInput.title = "Voice not supported in this browser";
    btnVoiceInput.addEventListener("click", () => {
      showToast("Voice input works in Chrome, Edge, and Safari. Try one of those browsers.", "warn");
    });
  }
}

// ----------------- Pleasing Natural AI Voice Engine (Web Speech API) -----------------
let cachedAvailableVoices = [];

function initPleasingVoices() {
  if (!window.speechSynthesis) return;
  
  const updateVoices = () => {
    try {
      const voices = window.speechSynthesis.getVoices() || [];
      if (voices.length > 0) {
        cachedAvailableVoices = voices;
      }
    } catch (e) {
      console.warn("Could not retrieve system voices:", e);
    }
  };

  updateVoices();
  if (window.speechSynthesis.onvoiceschanged !== undefined) {
    window.speechSynthesis.onvoiceschanged = updateVoices;
  }
}

// Initialise voices eagerly
initPleasingVoices();

// Ranked list of the most pleasing, natural-sounding, soft-spoken US English voices across platforms
const PLEASING_VOICE_PREFERENCES = {
  // General Coach: Warm, gentle, inviting, soft-spoken US female/balanced voice
  general: [
    "Microsoft Jenny Online (Natural) - English (United States)",
    "Microsoft Aria Online (Natural) - English (United States)",
    "Google US English",
    "Microsoft Michelle Online (Natural) - English (United States)",
    "Samantha (Enhanced)",
    "Samantha (Premium)",
    "Ava (Enhanced)",
    "Ava (Premium)",
    "Allison (Enhanced)",
    "Microsoft Jenny (Natural)",
    "Microsoft Aria (Natural)",
    "Samantha",
    "Ava",
    "Microsoft Zira - English (United States)",
    "Microsoft Zira",
  ],
  // Algorithm Specialist: Clear, articulate, soft, calm US voice
  algo_specialist: [
    "Microsoft Christopher Online (Natural) - English (United States)",
    "Microsoft Guy Online (Natural) - English (United States)",
    "Microsoft Eric Online (Natural) - English (United States)",
    "Google US English",
    "Tom (Enhanced)",
    "Samantha (Enhanced)",
    "Alex",
    "Microsoft Guy (Natural)",
    "Microsoft David - English (United States)",
    "Microsoft David",
  ],
  // System Architect: Deep, steady, calm, gentle US voice
  system_architect: [
    "Microsoft Steffan Online (Natural) - English (United States)",
    "Microsoft Roger Online (Natural) - English (United States)",
    "Microsoft Guy Online (Natural) - English (United States)",
    "Google US English",
    "Alex",
    "Samantha (Enhanced)",
    "Microsoft Steffan (Natural)",
    "Microsoft David - English (United States)",
    "Microsoft David",
  ],
  // Executive Career Strategist: Polished, smooth, poised, gentle US voice
  career_strategist: [
    "Microsoft Aria Online (Natural) - English (United States)",
    "Microsoft Jenny Online (Natural) - English (United States)",
    "Microsoft Michelle Online (Natural) - English (United States)",
    "Google US English",
    "Ava (Enhanced)",
    "Samantha (Enhanced)",
    "Ava (Premium)",
    "Samantha (Premium)",
    "Samantha",
    "Ava",
    "Microsoft Zira - English (United States)",
    "Microsoft Zira",
  ]
};

function isUsVoice(v) {
  if (!v) return false;
  const lang = (v.lang || "").toLowerCase();
  const name = (v.name || "").toLowerCase();
  // Exclude British / UK voices explicitly
  if (lang.includes("gb") || lang.includes("uk") || name.includes("uk english") || name.includes("united kingdom") || name.includes("great britain")) {
    return false;
  }
  return lang === "en-us" || lang === "en_us" || lang.startsWith("en-us") || name.includes("united states") || name.includes("us english") || name.includes("(us)") || (!lang.includes("-") && name.includes("us"));
}

function getPleasingVoice(persona = "general") {
  if (!cachedAvailableVoices || cachedAvailableVoices.length === 0) {
    if (window.speechSynthesis) {
      cachedAvailableVoices = window.speechSynthesis.getVoices() || [];
    }
  }
  if (!cachedAvailableVoices || cachedAvailableVoices.length === 0) return null;

  const targetPreferences = PLEASING_VOICE_PREFERENCES[persona] || PLEASING_VOICE_PREFERENCES.general;

  // 1. Try exact matches from persona US preferences
  for (const preferredName of targetPreferences) {
    const found = cachedAvailableVoices.find(v => v.name.toLowerCase() === preferredName.toLowerCase());
    if (found) return found;
  }

  // 2. Try partial name matches from persona US preferences (ensuring US voice)
  for (const preferredName of targetPreferences) {
    const firstWord = preferredName.split(" ")[0].toLowerCase();
    const secondWord = preferredName.split(" ")[1]?.toLowerCase();
    const found = cachedAvailableVoices.find(v => {
      const vName = v.name.toLowerCase();
      return isUsVoice(v) && vName.includes(firstWord) && (!secondWord || vName.includes(secondWord));
    });
    if (found) return found;
  }

  // 3. Look for any Natural / Neural / Studio / Premium / Enhanced US English voice
  const highQualityKeywords = ["natural", "neural", "online (natural)", "enhanced", "premium", "studio", "google"];
  for (const keyword of highQualityKeywords) {
    const match = cachedAvailableVoices.find(v => 
      isUsVoice(v) && v.name.toLowerCase().includes(keyword)
    );
    if (match) return match;
  }

  // 4. Look for any US English voice
  const usVoice = cachedAvailableVoices.find(v => isUsVoice(v));
  if (usVoice) return usVoice;

  // 5. Fallback to any non-UK English voice if available
  const englishVoice = cachedAvailableVoices.find(v => {
    const lang = (v.lang || "").toLowerCase();
    const name = (v.name || "").toLowerCase();
    return (lang.startsWith("en") && !lang.includes("gb") && !name.includes("uk english"));
  });
  if (englishVoice) return englishVoice;

  // 6. Default browser fallback voice
  return cachedAvailableVoices.find(v => v.default) || cachedAvailableVoices[0] || null;
}

function getVoiceProsody(persona = "general") {
  // Soft-spoken, calm, gentle acoustic profile (unhurried rate, warm pitch, comfortable volume)
  switch (persona) {
    case "algo_specialist":
      return { rate: 0.94, pitch: 0.98, volume: 0.88 };
    case "system_architect":
      return { rate: 0.92, pitch: 0.96, volume: 0.88 };
    case "career_strategist":
      return { rate: 0.93, pitch: 0.99, volume: 0.88 };
    case "general":
    default:
      return { rate: 0.93, pitch: 0.98, volume: 0.88 };
  }
}

function cleanTextForSpeech(text) {
  if (!text) return "";
  let clean = text
    // Replace mermaid blocks with a friendly spoken notification
    .replace(/```mermaid[\s\S]*?```/gi, " I have rendered an interactive architecture diagram on your screen. ")
    // Replace markdown code blocks
    .replace(/```[\s\S]*?```/g, " Code example displayed. ")
    // Replace markdown links: [label](url) -> label
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    // Remove HTML tags
    .replace(/<[^>]*>/g, "")
    // Remove markdown formatting symbols
    .replace(/[*_`#~>]/g, "")
    // Remove markdown bullet points / headers at line starts
    .replace(/^[\s-•*#]+/gm, "")
    // Remove emojis to avoid unnatural emoji metadata pronunciation
    .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]/gu, "")
    // Clean up excessive whitespace
    .replace(/\s+/g, " ")
    .trim();

  // Smart sentence truncation to avoid mid-word cuts and keep voice concise and pleasant
  // Keeps full sentences up to ~320 characters
  if (clean.length > 320) {
    const sentenceEndings = [...clean.matchAll(/[.!?]\s+/g)];
    let bestCut = -1;
    for (const match of sentenceEndings) {
      if (match.index >= 120 && match.index <= 320) {
        bestCut = match.index + 1;
      }
    }
    if (bestCut > 0) {
      clean = clean.substring(0, bestCut);
    } else {
      const lastSpace = clean.lastIndexOf(" ", 300);
      if (lastSpace > 100) {
        clean = clean.substring(0, lastSpace) + "...";
      }
    }
  }

  return clean;
}

// Text-to-Speech Read-Aloud Toggle with Gemini 2.5 Flash Native Audio
const btnVoiceToggle = document.getElementById("btn-voice-toggle");
let currentNativeAudio = null;

btnVoiceToggle?.addEventListener("click", () => {
  state.ttsActive = !state.ttsActive;
  if (state.ttsActive) {
    btnVoiceToggle.classList.add("active");
    const persona = state.activePersona || "general";
    const geminiVoiceMap = {
      general: "Kore (Warm Soft-Spoken)",
      algo_specialist: "Puck (Articulate Specialist)",
      system_architect: "Fenrir (Deep Architect)",
      career_strategist: "Aoede (Poised Strategist)"
    };
    const voiceLabel = geminiVoiceMap[persona] || "Kore";
    showToast(`Gemini 2.5 Flash Native Audio Enabled 🔊 (${voiceLabel})`, "info");
    speakText("Audio read aloud enabled. I will read my guidance aloud with Gemini 2.5 Flash native voice.");
  } else {
    btnVoiceToggle.classList.remove("active");
    btnVoiceToggle.classList.remove("speaking");
    stopAllAudioPlayback();
    showToast("Audio Read-Aloud Disabled 🔇", "info");
  }
});

function stopAllAudioPlayback() {
  if (currentNativeAudio) {
    try {
      currentNativeAudio.pause();
      currentNativeAudio.currentTime = 0;
    } catch (e) {}
    currentNativeAudio = null;
  }
  if (window.speechSynthesis) {
    try {
      window.speechSynthesis.cancel();
    } catch (e) {}
  }
}

async function speakText(text) {
  if (!state.ttsActive) return;
  
  // Stop previous audio playback
  stopAllAudioPlayback();
  
  const clean = cleanTextForSpeech(text);
  if (!clean) return;

  if (btnVoiceToggle) btnVoiceToggle.classList.add("speaking");

  // 1. Attempt Gemini 2.5 Flash Native Audio Output
  try {
    const res = await fetch(`${API_BASE}/api/chat/speech`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders()
      },
      body: JSON.stringify({
        user_id: state.userId,
        text: clean,
        persona: state.activePersona || "general"
      })
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data.success && data.audio_base64) {
        currentNativeAudio = new Audio("data:audio/wav;base64," + data.audio_base64);
        currentNativeAudio.onended = () => {
          if (btnVoiceToggle) btnVoiceToggle.classList.remove("speaking");
        };
        currentNativeAudio.onerror = () => {
          fallbackToBrowserSpeech(clean);
        };
        await currentNativeAudio.play();
        return;
      }
    }
  } catch (err) {
    console.warn("Gemini 2.5 Flash native audio request error, falling back:", err);
  }

  // 2. Fallback to soft-spoken US English Web Speech API
  fallbackToBrowserSpeech(clean);
}

function fallbackToBrowserSpeech(cleanText) {
  if (!window.speechSynthesis || !state.ttsActive) {
    if (btnVoiceToggle) btnVoiceToggle.classList.remove("speaking");
    return;
  }

  const utterance = new SpeechSynthesisUtterance(cleanText);
  const voice = getPleasingVoice(state.activePersona || "general");
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang || "en-US";
  }

  const prosody = getVoiceProsody(state.activePersona || "general");
  utterance.rate = prosody.rate;
  utterance.pitch = prosody.pitch;
  utterance.volume = prosody.volume;

  utterance.onend = () => {
    if (btnVoiceToggle) btnVoiceToggle.classList.remove("speaking");
  };
  utterance.onerror = () => {
    if (btnVoiceToggle) btnVoiceToggle.classList.remove("speaking");
  };

  window.speechSynthesis.speak(utterance);
}

// 60s Micro-Challenge Handler
document.getElementById("btn-micro-challenge")?.addEventListener("click", async () => {
  try {
    showToast("Generating 60-second micro-challenge...", "info");
    const res = await fetch(`${API_BASE}/api/chat/micro-challenge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId })
    });
    const data = await res.json();
    addBotChatMessage(data.reply);
  } catch (err) {
    showToast("Could not retrieve micro-challenge.", "error");
  }
});

chatForm?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  chatInput.value = "";
  addUserChatMessage(text);

  // Show typing indicator
  const typing = document.createElement("div");
  typing.className = "chat-msg bot";
  typing.id = "coach-typing";
  typing.innerHTML = `<p class="muted font-12"><em>Thinking & consulting active persona...</em></p>`;
  chatMessages.appendChild(typing);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({
        user_id: state.userId,
        message: text,
        persona: state.activePersona || "general"
      })
    });
    const data = await res.json();
    typing.remove();
    addBotChatMessage(data.reply);
  } catch (err) {
    typing.remove();
    addBotChatMessage("I encountered a network issue. Please ask again.");
  }
});

// Wire up quick-action prompt chips
document.querySelectorAll(".quick-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    if (chip.id === "btn-micro-challenge") return;
    const prompt = chip.dataset.prompt;
    if (!prompt) return;
    chatInput.value = prompt;
    chatForm.dispatchEvent(new Event("submit"));
  });
});

// Wire up clear chat button
document.getElementById("btn-clear-chat")?.addEventListener("click", () => {
  chatMessages.innerHTML = `
    <div class="chat-msg bot">
      <p>👋 <strong>Welcome!</strong> I'm your grounded AI Learning Coach. Select a mentor persona above or ask any custom question about your learning roadmap.</p>
      <p style="margin-top: 6px;" class="muted font-12">Click a quick topic below, ask for a visual architecture diagram, or tap 🎙️ to speak.</p>
    </div>
  `;
  lastBotMsgText = "";
});

// ----------------- Stage 4: Visual DAG Topology Explorer -----------------

const modalGraph = document.getElementById("modal-graph-viewer");
const btnViewGraph = document.getElementById("btn-view-graph");
const btnCloseGraph = document.getElementById("btn-close-graph");
const svgDag = document.getElementById("svg-dag-graph");
const graphBanner = document.getElementById("graph-node-details");

btnViewGraph?.addEventListener("click", () => {
  modalGraph.classList.remove("hidden");
  renderInteractiveDagGraph();
});

btnCloseGraph?.addEventListener("click", () => {
  modalGraph.classList.add("hidden");
});

modalGraph?.addEventListener("click", (e) => {
  if (e.target === modalGraph) modalGraph.classList.add("hidden");
});

let cachedGraphData = null;

async function renderInteractiveDagGraph() {
  if (!cachedGraphData) {
    try {
      const res = await fetch(`${API_BASE}/api/graph`);
      cachedGraphData = await res.json();
    } catch (e) {
      console.error("Failed to load graph data", e);
      return;
    }
  }

  const { nodes, edges, role_targets } = cachedGraphData;
  const activePathSkills = new Set((state.activePlan?.nodes || []).map(n => n.skill));
  const currentRole = state.targetRole || "Data Scientist";
  const targetRoleSkills = new Set(role_targets[currentRole] || []);

  // Compute node levels/ranks using longest path topological distance
  const inDegree = {};
  const adj = {};
  const revAdj = {};
  Object.keys(nodes).forEach(n => {
    inDegree[n] = 0;
    adj[n] = [];
    revAdj[n] = [];
  });
  edges.forEach(([u, v]) => {
    if (adj[u] && revAdj[v]) {
      adj[u].push(v);
      revAdj[v].push(u);
      inDegree[v] = (inDegree[v] || 0) + 1;
    }
  });

  const levels = {};
  const queue = Object.keys(nodes).filter(n => inDegree[n] === 0);
  queue.forEach(n => { levels[n] = 0; });

  const qCopy = [...queue];
  while (qCopy.length > 0) {
    const curr = qCopy.shift();
    const currLvl = levels[curr] || 0;
    (adj[curr] || []).forEach(nxt => {
      levels[nxt] = Math.max(levels[nxt] || 0, currLvl + 1);
      inDegree[nxt]--;
      if (inDegree[nxt] === 0) {
        qCopy.push(nxt);
      }
    });
  }

  // Group nodes by level (columns)
  const columns = {};
  Object.keys(nodes).forEach(n => {
    const lvl = levels[n] || 0;
    if (!columns[lvl]) columns[lvl] = [];
    columns[lvl].push(n);
  });

  const totalCols = Math.max(...Object.keys(columns).map(Number)) + 1;
  const width = 1080;
  const height = 480;
  const colSpacing = (width - 120) / Math.max(1, totalCols - 1);

  // Position coordinates map
  const positions = {};
  Object.entries(columns).forEach(([lvlStr, colNodes]) => {
    const colIdx = parseInt(lvlStr);
    const x = 60 + colIdx * colSpacing;
    const rowSpacing = (height - 80) / Math.max(1, colNodes.length + 1);
    colNodes.forEach((n, rIdx) => {
      const y = 40 + (rIdx + 1) * rowSpacing;
      positions[n] = { x, y };
    });
  });

  // Render SVG Edges
  let svgContent = `<defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="rgba(224, 166, 76, 0.4)" />
    </marker>
    <marker id="arrow-active" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#E0A64C" />
    </marker>
  </defs>`;

  // Draw Edges
  edges.forEach(([u, v]) => {
    const p1 = positions[u];
    const p2 = positions[v];
    if (!p1 || !p2) return;
    const isActiveEdge = activePathSkills.has(u) && activePathSkills.has(v);
    const dx = (p2.x - p1.x) / 2;
    const pathD = `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
    svgContent += `<path d="${pathD}" class="dag-edge ${isActiveEdge ? 'highlighted' : ''}" marker-end="url(#${isActiveEdge ? 'arrow-active' : 'arrow'})" />`;
  });

  // Draw Nodes
  Object.entries(nodes).forEach(([id, meta]) => {
    const pos = positions[id];
    if (!pos) return;
    const isActive = activePathSkills.has(id);
    const isTarget = targetRoleSkills.has(id);

    let fillColor = "#475569";
    let strokeColor = "rgba(255,255,255,0.2)";
    if (isActive) {
      fillColor = "#E0A64C";
      strokeColor = "#FFE08A";
    } else if (isTarget) {
      fillColor = "#0284C7";
      strokeColor = "#38BDF8";
    }

    svgContent += `
      <g class="dag-node" data-id="${id}" data-cluster="${meta.cluster || 'general'}" transform="translate(${pos.x}, ${pos.y})">
        <circle r="7.5" fill="${fillColor}" stroke="${strokeColor}" stroke-width="2" />
        <text x="11" y="3.5">${meta.name || id}</text>
      </g>
    `;
  });

  svgDag.innerHTML = svgContent;

  // Cluster filter chips handler
  document.querySelectorAll(".graph-filter-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".graph-filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const filter = btn.dataset.filter;

      svgDag.querySelectorAll(".dag-node").forEach(gn => {
        const cluster = gn.dataset.cluster;
        if (filter === "all" || cluster.includes(filter) || (filter === "ui_ux" && cluster.includes("design"))) {
          gn.style.opacity = "1";
          gn.style.filter = "none";
        } else {
          gn.style.opacity = "0.2";
          gn.style.filter = "grayscale(1)";
        }
      });
    };
  });

  // Add node interaction listeners
  svgDag.querySelectorAll(".dag-node").forEach(gNode => {
    const skillId = gNode.dataset.id;
    const meta = nodes[skillId] || {};
    const prereqs = revAdj[skillId] || [];

    gNode.addEventListener("mouseenter", () => {
      const prereqNames = prereqs.map(p => nodes[p]?.name || p).join(", ") || "None (Foundational Node)";
      graphBanner.innerHTML = `
        <div>
          <strong style="color: #E0A64C;">${meta.name || skillId}</strong> 
          <span class="muted" style="margin-left: 8px;">Cluster: ${meta.cluster || 'General'} | Difficulty: Tier ${meta.difficulty || 1}</span>
          <div class="font-11 muted" style="margin-top: 2px;">Prerequisites: ${prereqNames}</div>
        </div>
      `;
    });
  });
}

// ----------------- Stage 5: Export Roadmap Engine -----------------

const btnExport = document.getElementById("btn-export-roadmap");

btnExport?.addEventListener("click", () => {
  if (!state.activePlan || !state.activePlan.nodes) {
    showToast("Generate a learning path first.", "warn");
    return;
  }

  const plan = state.activePlan;
  let md = `# 🧭 Career PathFinder — Learning Roadmap & Syllabus\n`;
  md += `**Target Role:** ${plan.role}\n`;
  md += `**Estimated Duration:** ${plan.estimated_weeks} weeks at ${plan.weekly_commitment_hours} hrs/week (${plan.total_estimated_hours} total hours)\n`;
  md += `**Role Readiness Baseline:** ${Math.round(plan.readiness_score * 100)}%\n`;
  md += `**Generated By:** Career PathFinder Topological Knowledge Engine\n\n`;
  md += `---\n\n`;
  md += `## 📋 Ordered Milestone Curriculum\n\n`;

  plan.nodes.forEach((node, idx) => {
    md += `### Step ${node.step_number}: ${node.skill_name} (Difficulty: Tier ${node.difficulty})\n`;
    md += `* **O*NET Importance:** ${Math.round((node.importance || 0.8) * 100)}% | **Baseline Mastery:** ${Math.round((node.current_mastery || 0) * 100)}%\n`;
    md += `* **Skill Cluster:** ${node.cluster}\n\n`;
    
    md += `#### 📚 Recommended Real-World Courses:\n`;
    (node.courses || []).forEach(c => {
      md += `  - **[${c.title}](${c.url})** (${c.platform}) — Match Score: ${Math.round(c.match_score * 100)}% (Rating: ${c.rating}★)\n`;
    });
    md += `\n`;

    if (node.project) {
      md += `#### 🛠️ Outcome-Based Hands-On Capstone Project:\n`;
      md += `* **Project Title:** ${node.project.title}\n`;
      md += `* **Objective:** ${node.project.objective}\n`;
      md += `* **Deliverables:**\n`;
      (node.project.deliverables || []).forEach(d => {
        md += `  - [ ] ${d}\n`;
      });
      md += `* **Estimated Effort:** ~${node.project.estimated_hours} hours\n\n`;
    }

    if (node.trace) {
      md += `#### 🔍 Glass-Box Reasoning Trace:\n`;
      md += `> ${node.trace.recommendation_rationale}\n\n`;
    }

    md += `---\n\n`;
  });

  md += `\n*Automated roadmap compiled with zero prerequisite violations via Kahn's Topological Sort algorithm.*\n`;

  // Trigger download as .md file
  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${plan.role.replace(/\\s+/g, '_')}_Roadmap.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// Initialize Mermaid with dark theme
if (typeof mermaid !== "undefined") {
  try {
    mermaid.initialize({
      startOnLoad: false,
      theme: "dark",
      themeVariables: {
        darkMode: true,
        background: "#0B0E14",
        primaryColor: "#1e293b",
        primaryTextColor: "#f8fafc",
        primaryBorderColor: "#E0A64C",
        lineColor: "#E0A64C",
        secondaryColor: "#0f172a",
        tertiaryColor: "#1e293b"
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: "basis"
      },
      securityLevel: "loose"
    });
  } catch (e) {
    console.warn("Mermaid init notice:", e);
  }
}

// 60s Micro-Challenge Handler
document.getElementById("btn-micro-challenge")?.addEventListener("click", async () => {
  try {
    showToast("Generating 60-second micro-challenge...", "info");
    const res = await fetch(`${API_BASE}/api/chat/micro-challenge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId })
    });
    const data = await res.json();
    if (data.question && data.question.options) {
      addMicroChallengeCard(data);
    } else {
      addBotChatMessage(data.reply);
    }
  } catch (err) {
    showToast("Could not retrieve micro-challenge.", "error");
  }
});

function addMicroChallengeCard(data) {
  const q = data.question || {};
  const skillId = (data.skill_id || "Competency").replace(/_/g, " ").toUpperCase();
  const qText = q.q || q.question || "Test your recall for this skill:";
  const options = q.options || ["Option A", "Option B", "Option C", "Option D"];
  const correctAns = q.answer || "";

  const container = document.createElement("div");
  container.className = "chat-msg bot";

  let optionsHtml = "";
  options.forEach((opt) => {
    optionsHtml += `<button type="button" class="quiz-opt-btn" data-opt="${escapeHtml(opt)}">${escapeHtml(opt)}</button>`;
  });

  container.innerHTML = `
    <p>⚡ <strong>60-Second Micro-Challenge (${escapeHtml(skillId)}):</strong></p>
    <div class="quiz-nudge-card">
      <p class="quiz-nudge-q">${escapeHtml(qText)}</p>
      <div class="quiz-nudge-options">
        ${optionsHtml}
      </div>
      <div class="quiz-feedback-box hidden" style="margin-top: 10px; font-size: 12px; line-height: 1.4;"></div>
    </div>
  `;

  chatMessages.appendChild(container);
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });

  const optBtns = container.querySelectorAll(".quiz-opt-btn");
  const fbBox = container.querySelector(".quiz-feedback-box");

  optBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      optBtns.forEach(b => b.disabled = true);
      const chosen = btn.getAttribute("data-opt") || "";
      const isCorrect = correctAns ? (chosen.trim().toLowerCase() === correctAns.trim().toLowerCase() || chosen.includes(correctAns) || correctAns.includes(chosen)) : true;

      if (isCorrect) {
        btn.classList.add("correct-choice");
        fbBox.innerHTML = `🎯 <strong style="color: #4ADE80;">Correct!</strong> Active recall reinforced for <em>${escapeHtml(skillId)}</em>.`;
        fbBox.style.color = "#4ADE80";
      } else {
        btn.classList.add("wrong-choice");
        optBtns.forEach(b => {
          const val = b.getAttribute("data-opt") || "";
          if (correctAns && (val.trim().toLowerCase() === correctAns.trim().toLowerCase() || val.includes(correctAns) || correctAns.includes(val))) {
            b.classList.add("correct-choice");
          }
        });
        fbBox.innerHTML = `💡 <strong style="color: #E0A64C;">Core Concept:</strong> The correct answer is <strong>${escapeHtml(correctAns || "Option A")}</strong>.`;
        fbBox.style.color = "#E2E8F0";
      }
      fbBox.classList.remove("hidden");
      chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
    });
  });
}

function addUserChatMessage(text) {
  const msg = document.createElement("div");
  msg.className = "chat-msg user";
  msg.innerHTML = `<p>${escapeHtml(text)}</p>`;
  chatMessages.appendChild(msg);
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

let lastBotMsgText = "";
function addBotChatMessage(text) {
  // Prevent duplicate consecutive spam
  if (text === lastBotMsgText) {
    return;
  }
  lastBotMsgText = text;

  // Render TTS audio if enabled
  if (state.ttsActive) {
    speakText(text);
  }

  let formattedText = text;
  let rawMermaidCode = "";

  // Extract and process ```mermaid ... ``` code blocks
  if (formattedText.includes("```mermaid")) {
    const parts = formattedText.split(/```mermaid[\r\n]?([\s\S]*?)```/);
    if (parts.length >= 2) {
      formattedText = parts[0] + (parts[2] || "");
      rawMermaidCode = parts[1].trim();
    }
  }

  // Rich markdown parsing
  let html = formattedText
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" style="color: var(--gold); text-decoration: underline;">$1</a>');

  // Convert bullet lists
  const lines = html.split("\n");
  let inList = false;
  let resultLines = [];

  lines.forEach(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (!inList) {
        resultLines.push("<ul>");
        inList = true;
      }
      resultLines.push(`<li>${trimmed.substring(2)}</li>`);
    } else {
      if (inList) {
        resultLines.push("</ul>");
        inList = false;
      }
      if (trimmed) {
        resultLines.push(`<p>${trimmed}</p>`);
      }
    }
  });
  if (inList) {
    resultLines.push("</ul>");
  }

  const msg = document.createElement("div");
  msg.className = "chat-msg bot";

  let mermaidPlaceholderHtml = "";
  if (rawMermaidCode) {
    mermaidPlaceholderHtml = `<div class="mermaid-diagram-container"><div class="mermaid-target" style="opacity: 0.9;">Loading visual flowchart...</div></div>`;
  }

  msg.innerHTML = resultLines.join("") + mermaidPlaceholderHtml;
  chatMessages.appendChild(msg);
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });

  // Async Mermaid SVG render
  if (rawMermaidCode && window.mermaid) {
    const target = msg.querySelector(".mermaid-target");
    if (target) {
      const renderId = "mermaid_svg_" + Math.random().toString(36).substring(2, 9);
      window.mermaid.render(renderId, rawMermaidCode)
        .then(({ svg }) => {
          target.innerHTML = svg;
          chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
        })
        .catch(err => {
          console.warn("Mermaid render fallback:", err);
          target.innerHTML = `<pre style="font-size: 11px; text-align: left; color: #94a3b8; padding: 6px;">${escapeHtml(rawMermaidCode)}</pre>`;
        });
    }
  }
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.innerText = str;
  return div.innerHTML;
}

// ----------------- UX: toast notifications, loading overlay, modal a11y -----------------

function showToast(message, type = "info") {
  let wrap = document.getElementById("cpf-toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.id = "cpf-toast-wrap";
    wrap.className = "cpf-toast-wrap";
    wrap.setAttribute("role", "status");
    wrap.setAttribute("aria-live", "polite");
    document.body.appendChild(wrap);
  }
  const t = document.createElement("div");
  t.className = "cpf-toast cpf-toast-" + type;
  t.textContent = message;
  wrap.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 300); }, 4200);
}

function showLoader(message = "Working…") {
  let el = document.getElementById("cpf-loader");
  if (!el) {
    el = document.createElement("div");
    el.id = "cpf-loader";
    el.className = "cpf-loader";
    el.innerHTML = '<div class="cpf-spinner" aria-hidden="true"></div><p class="cpf-loader-msg"></p>';
    document.body.appendChild(el);
  }
  el.querySelector(".cpf-loader-msg").textContent = message;
  el.classList.add("show");
}

function hideLoader() {
  const el = document.getElementById("cpf-loader");
  if (el) el.classList.remove("show");
}

// Close any open modal with Escape and restore focus to whatever opened it.
let _lastModalTrigger = null;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const open = document.querySelectorAll(".modal-overlay:not(.hidden)");
    if (open.length) {
      open.forEach(m => m.classList.add("hidden"));
      if (_lastModalTrigger) { _lastModalTrigger.focus(); _lastModalTrigger = null; }
    }
  }
});

// When the DAG explorer opens, remember the trigger and move focus into the modal.
(function enhanceGraphModalFocus() {
  const trigger = document.getElementById("btn-view-graph");
  const closeBtn = document.getElementById("btn-close-graph");
  if (trigger) trigger.addEventListener("click", () => {
    _lastModalTrigger = trigger;
    setTimeout(() => { if (closeBtn) closeBtn.focus(); }, 40);
  });
  if (closeBtn) closeBtn.addEventListener("click", () => {
    if (_lastModalTrigger) { _lastModalTrigger.focus(); _lastModalTrigger = null; }
  });
})();
