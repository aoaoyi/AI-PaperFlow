const API_BASE = window.AI_PAPERFLOW_API_BASE || "http://127.0.0.1:8000";

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function badge(label, value) {
  return `<span class="badge">${escapeHtml(label)}: ${escapeHtml(value)}</span>`;
}

function paperCard(paper) {
  const sources = (paper.retrieval_sources || []).join(", ") || "retrieval";
  return `
    <article class="paper-card">
      <h3>${paper.rank ? `${paper.rank}. ` : ""}${escapeHtml(paper.title)}</h3>
      <div class="meta">
        ${badge("Final", Number(paper.final_score || 0).toFixed(2))}
        ${badge("Hybrid", Number(paper.hybrid_score || 0).toFixed(4))}
        ${paper.rerank_score !== null && paper.rerank_score !== undefined ? badge("Rerank", Number(paper.rerank_score).toFixed(4)) : ""}
        ${badge("Sources", sources)}
      </div>
      <div class="links">
        ${paper.source_url ? `<a href="${escapeHtml(paper.source_url)}" target="_blank" rel="noreferrer">Source</a>` : ""}
        ${paper.pdf_url ? `<a href="${escapeHtml(paper.pdf_url)}" target="_blank" rel="noreferrer">PDF</a>` : ""}
      </div>
    </article>
  `;
}

function renderPapers(papers) {
  if (!papers || papers.length === 0) return `<p>No retrieved papers.</p>`;
  return `<div class="paper-list">${papers.map(paperCard).join("")}</div>`;
}

function switchTab(tabName) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabName));
  document.querySelector("#qaPanel").classList.toggle("active", tabName === "qa");
  document.querySelector("#researchPanel").classList.toggle("active", tabName === "research");
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));

document.querySelector("#qaForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#questionInput").value.trim();
  const result = document.querySelector("#qaResult");
  result.textContent = "Loading...";
  try {
    const data = await postJson("/ask", { question, top_k: 5, use_rerank: true });
    result.innerHTML = `
      <h2>Answer</h2>
      <div class="meta">
        ${badge("Mode", data.retrieval_mode)}
        ${badge("Rerank", data.rerank_used)}
        ${badge("LLM", data.llm_used)}
        ${badge("Model", data.model || "none")}
        ${badge("Candidates", data.candidate_count)}
      </div>
      ${data.fallback_reason ? `<p class="notice">${escapeHtml(data.fallback_reason)}</p>` : ""}
      <pre class="report">${escapeHtml(data.answer)}</pre>
      <h3>Retrieved Papers</h3>
      ${renderPapers(data.retrieved_papers)}
    `;
  } catch (error) {
    result.textContent = error.message;
  }
});

document.querySelector("#researchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = document.querySelector("#topicInput").value.trim();
  const result = document.querySelector("#researchResult");
  result.textContent = "Generating...";
  try {
    const data = await postJson("/research", { topic, top_k: 5 });
    result.innerHTML = `
      <h2>Research Report</h2>
      <div class="meta">
        ${badge("LLM", data.llm_used)}
        ${badge("Verification", data.verification?.verification_passed)}
        ${badge("Candidates", data.candidate_count || 0)}
      </div>
      ${data.fallback_reason ? `<p class="notice">${escapeHtml(data.fallback_reason)}</p>` : ""}
      <h3>Sub Questions</h3>
      <ul>${(data.sub_questions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <pre class="report">${escapeHtml(data.report || "")}</pre>
      <h3>Retrieved Papers</h3>
      ${renderPapers(data.retrieved_papers)}
    `;
  } catch (error) {
    result.textContent = error.message;
  }
});
