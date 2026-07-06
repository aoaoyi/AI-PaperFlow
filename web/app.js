const THEME_STORAGE_KEY = "paper-daily-theme";
const THEMES = new Set(["dark", "light", "eye"]);

const state = {
  datasets: {
    daily: null,
    conference: null,
  },
  allPapers: [],
  theme: "dark",
  filters: {
    query: "",
    topic: "all",
    level: "all",
    collection: "all",
    view: "all",
    date: null,
  },
};

const nodes = {
  updatedAt: document.querySelector("#updatedAt"),
  dataDebug: document.querySelector("#dataDebug"),
  paperCount: document.querySelector("#paperCount"),
  weekCount: document.querySelector("#weekCount"),
  monthCount: document.querySelector("#monthCount"),
  topScore: document.querySelector("#topScore"),
  resultCount: document.querySelector("#resultCount"),
  viewTitle: document.querySelector("#viewTitle"),
  listTitle: document.querySelector("#listTitle"),
  scopeLabel: document.querySelector("#scopeLabel"),
  paperList: document.querySelector("#paperList"),
  topicFilter: document.querySelector("#topicFilter"),
  levelFilter: document.querySelector("#levelFilter"),
  dateFilter: document.querySelector("#dateFilter"),
  searchInput: document.querySelector("#searchInput"),
  themeOptions: document.querySelectorAll("[data-theme-option]"),
  collectionTabs: document.querySelectorAll("[data-collection]"),
  tabs: document.querySelectorAll(".tab"),
  template: document.querySelector("#paperTemplate"),
};

function emptyDataset() {
  return { generated_at_iso: new Date().toISOString(), topics: [], papers: [], stats: {} };
}

function normalizeData(payload, dataKind = "daily") {
  const papers = Array.isArray(payload) ? payload : payload?.papers || [];
  const topicMap = new Map();
  for (const paper of papers) {
    for (const name of paper.matched_topic_names || []) {
      const id = String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
      if (id) topicMap.set(id, { id, name });
    }
    for (const id of paper.matched_topic_ids || []) {
      if (id && !topicMap.has(String(id))) topicMap.set(String(id), { id: String(id), name: String(id) });
    }
  }
  return {
    ...(Array.isArray(payload) ? {} : payload || {}),
    data_kind: dataKind,
    generated_at_iso: payload?.generated_at_iso || payload?.generated_at || new Date().toISOString(),
    topics: Array.isArray(payload?.topics) && payload.topics.length ? payload.topics : [...topicMap.values()],
    papers: Array.isArray(papers) ? papers : [],
    stats: payload?.stats || {},
  };
}

function activeData() {
  if (state.filters.collection === "conference") return state.datasets.conference || emptyDataset();
  return state.datasets.daily || emptyDataset();
}

function currentFilters() {
  return {
    filterMode: state.filters.view,
    dateFilter: state.filters.date,
    topicFilter: state.filters.topic,
    matchFilter: state.filters.level,
    searchQuery: state.filters.query,
    collection: state.filters.collection,
  };
}

function storedTheme() {
  try {
    const theme = localStorage.getItem(THEME_STORAGE_KEY);
    return THEMES.has(theme) ? theme : "dark";
  } catch {
    return "dark";
  }
}

function applyTheme(theme) {
  state.theme = THEMES.has(theme) ? theme : "dark";
  document.body.dataset.theme = state.theme;
  for (const option of nodes.themeOptions) {
    const active = option.dataset.themeOption === state.theme;
    option.classList.toggle("active", active);
    option.setAttribute("aria-checked", String(active));
  }
  try {
    localStorage.setItem(THEME_STORAGE_KEY, state.theme);
  } catch {
    // localStorage may be blocked in privacy-focused browser modes.
  }
}

function parseDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatDate(value) {
  const date = parseDate(value);
  if (!date) return value ? String(value).slice(0, 10) : "-";
  return date.toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" });
}

function dateKey(value) {
  const date = parseDate(value);
  if (!date) return "";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function collectionTime(paper) {
  return paper.publication_date || paper.published || paper.last_seen_at || paper.first_seen_at || paper.updated || "";
}

function paperDate(paper) {
  return paper.publication_date || paper.published || collectionTime(paper);
}

function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function startOfWeek(date) {
  const day = startOfDay(date);
  const offset = (day.getDay() + 6) % 7;
  day.setDate(day.getDate() - offset);
  return day;
}

function endOfWeek(date) {
  const end = startOfWeek(date);
  end.setDate(end.getDate() + 7);
  return end;
}

function startOfMonth(date) {
  return new Date(date.getFullYear(), date.getMonth(), 1);
}

function endOfMonth(date) {
  return new Date(date.getFullYear(), date.getMonth() + 1, 1);
}

function inRange(value, start, end) {
  const date = parseDate(value);
  return Boolean(date && date >= start && date < end);
}

function selectedDate() {
  return state.filters.date ? parseDate(`${state.filters.date}T12:00:00`) || new Date() : new Date();
}

function scoreOf(paper) {
  return Number(paper.score ?? paper.best_match?.score ?? paper.final_score ?? 0);
}

function finalScoreOf(paper) {
  return Number(paper.final_score ?? paper.best_match?.score ?? 0);
}

function displayFinalScore(paper) {
  if (paper.final_score === undefined || paper.final_score === null) return "";
  const value = Number(paper.final_score);
  return Number.isFinite(value) ? `Final Score: ${value.toFixed(2)}` : "";
}

function levelOf(paper) {
  if (paper.best_match?.level) return String(paper.best_match.level).toLowerCase();
  const score = finalScoreOf(paper);
  if (score >= 15) return "high";
  if (score >= 8) return "medium";
  return "low";
}

function topicNamesOf(paper) {
  return paper.matched_topic_names || paper.topics || paper.categories || [];
}

function topicIdsOf(paper) {
  return paper.matched_topic_ids || [];
}

function abstractPreview(paper) {
  const text = String(paper.abstract || paper.summary || "");
  return text.length > 300 ? `${text.slice(0, 300)}...` : text;
}

function textIncludes(paper, query) {
  if (!query) return true;
  const haystack = [
    paper.title,
    paper.abstract,
    paper.summary,
    (paper.authors || []).join(" "),
    (paper.topics || []).join(" "),
    (paper.categories || []).join(" "),
    (paper.matched_topic_names || []).join(" "),
    paper.best_match?.reason,
    paper.chinese_summary?.innovation,
    paper.chinese_summary?.evidence,
    paper.chinese_summary?.limitations,
    paper.chinese_summary?.why_relevant,
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function matchesBaseFilters(paper) {
  if (!textIncludes(paper, state.filters.query)) return false;
  if (state.filters.topic !== "all") {
    const ids = topicIdsOf(paper).map(String);
    const names = topicNamesOf(paper).map((name) => String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""));
    if (!ids.includes(state.filters.topic) && !names.includes(state.filters.topic)) return false;
  }
  if (state.filters.level !== "all" && levelOf(paper) !== state.filters.level) return false;
  return true;
}

function matchesView(paper) {
  if (state.filters.view === "all") return true;
  const date = selectedDate();
  const publishedAt = paperDate(paper);
  if (state.filters.view === "daily") return dateKey(publishedAt) === state.filters.date;
  if (state.filters.view === "week") return inRange(publishedAt, new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), new Date(Date.now() + 24 * 60 * 60 * 1000));
  if (state.filters.view === "month") return inRange(publishedAt, startOfMonth(date), endOfMonth(date));
  if (state.filters.view === "highlights") {
    return inRange(publishedAt, new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), new Date(Date.now() + 24 * 60 * 60 * 1000)) && finalScoreOf(paper) >= 15;
  }
  return true;
}

function filteredPapers() {
  const data = activeData();
  const basePapers = state.filters.collection === "conference" ? data.papers || [] : state.allPapers;
  const rankScore = state.filters.collection === "conference" ? scoreOf : finalScoreOf;
  return (basePapers || [])
    .filter((paper) => matchesBaseFilters(paper) && matchesView(paper))
    // Daily papers are ranked by the personalized final_score while the displayed badge keeps the original score.
    .sort((a, b) => rankScore(b) - rankScore(a) || String(paperDate(b) || "").localeCompare(String(paperDate(a) || "")));
}

function setText(parent, selector, text) {
  parent.querySelector(selector).textContent = text || "暂无";
}

function safeFilename(paper) {
  const title = String(paper.title || paper.id || "paper")
    .replace(/[\\/:*?"<>|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
  return `${title || "paper"}.pdf`;
}

function renderPaper(paper) {
  const node = nodes.template.content.firstElementChild.cloneNode(true);
  const best = paper.best_match || {};
  const summary = paper.chinese_summary || paper.structured_summary || {};
  const badge = node.querySelector(".match-badge");
  const level = levelOf(paper);

  badge.textContent = `${level} ${scoreOf(paper).toFixed(2)}`;
  badge.classList.add(level);

  const finalScoreText = displayFinalScore(paper);
  if (finalScoreText) {
    const finalScoreNode = document.createElement("span");
    finalScoreNode.className = "paper-final-score";
    finalScoreNode.textContent = finalScoreText;
    badge.insertAdjacentElement("afterend", finalScoreNode);
  }

  setText(node, ".paper-date", `发布 ${formatDate(paperDate(paper))} · 年份 ${paper.publication_year || "-"}`);
  setText(node, ".paper-source", `${paper.source || "OpenAlex"} · 引用 ${Number(paper.cited_by_count || 0)}`);
  setText(node, ".paper-title", paper.title);
  setText(node, ".paper-authors", (paper.authors || []).slice(0, 8).join(", "));
  setText(node, ".summary-problem", summary.problem || abstractPreview(paper));
  setText(node, ".summary-method", summary.method);
  setText(node, ".summary-innovation", summary.innovation);
  setText(node, ".summary-evidence", summary.evidence || `Cited by ${Number(paper.cited_by_count || 0)}`);
  setText(node, ".summary-limitations", summary.limitations || summary.limitation);
  setText(node, ".summary-relevant", summary.why_relevant);
  setText(node, ".match-reason", `${topicNamesOf(paper).join(", ") || best.topic_name || "未分类"}：${best.reason || "OpenAlex matched topic"}`);

  const tags = node.querySelector(".paper-tags");
  for (const category of topicNamesOf(paper).slice(0, 8)) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = category;
    tags.appendChild(tag);
  }

  const absLink = node.querySelector(".abs-link");
  const pdfLink = node.querySelector(".pdf-link");
  const downloadLink = node.querySelector(".download-link");
  const sourceUrl = paper.source_url || paper.paper_url || "#";
  const pdfUrl = paper.pdf_url || "#";
  absLink.href = sourceUrl;
  pdfLink.href = pdfUrl;
  downloadLink.href = pdfUrl;
  pdfLink.style.display = paper.pdf_url ? "" : "none";
  downloadLink.style.display = paper.pdf_url ? "" : "none";
  downloadLink.setAttribute("download", safeFilename(paper));
  downloadLink.setAttribute("target", "_blank");
  downloadLink.setAttribute("rel", "noreferrer");
  return node;
}

function viewLabels() {
  const date = selectedDate();
  const dayLabel = formatDate(date.toISOString());
  const weekStart = formatDate(startOfWeek(date).toISOString());
  const weekEndDate = endOfWeek(date);
  weekEndDate.setDate(weekEndDate.getDate() - 1);
  const weekEnd = formatDate(weekEndDate.toISOString());
  const monthLabel = `${date.getFullYear()} 年 ${String(date.getMonth() + 1).padStart(2, "0")} 月`;
  return {
    all: [state.filters.collection === "conference" ? "顶会精品" : "全部论文", "全部已收录论文"],
    daily: ["当日论文", dayLabel],
    week: ["本周论文", `${weekStart} - ${weekEnd}`],
    month: ["月度论文", monthLabel],
    highlights: ["本周精选", `${weekStart} - ${weekEnd}`],
  };
}

function updateHeadings(papers) {
  const labels = viewLabels()[state.filters.view];
  nodes.viewTitle.textContent = labels[0];
  nodes.listTitle.textContent = labels[0];
  nodes.scopeLabel.textContent = labels[1];
  nodes.resultCount.textContent = `${papers.length} 篇`;
}

function syncDateFilterState() {
  nodes.dateFilter.disabled = state.filters.view !== "daily";
}

function render() {
  const papers = filteredPapers();
  console.log("Filtered papers count:", papers.length);
  console.log("Current filters:", currentFilters());
  syncDateFilterState();
  updateHeadings(papers);
  nodes.paperList.textContent = "";

  if (!papers.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = state.allPapers.length > 0 ? "当前筛选条件下没有论文，请尝试切换到全部论文。" : "当前没有可展示的论文数据。";
    nodes.paperList.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const paper of papers) fragment.appendChild(renderPaper(paper));
  nodes.paperList.appendChild(fragment);
}

function hydrateTopicFilter() {
  nodes.topicFilter.innerHTML = '<option value="all">全部方向</option>';
  for (const topic of activeData().topics || []) {
    const option = document.createElement("option");
    option.value = topic.id;
    option.textContent = topic.name;
    nodes.topicFilter.appendChild(option);
  }
}

function hydrateDateFilter() {
  const data = activeData();
  const dates = [...new Set((data.papers || []).map((paper) => dateKey(paperDate(paper))).filter(Boolean))].sort().reverse();
  const fallback = dateKey(data.generated_at_iso || new Date().toISOString());
  const options = dates.length ? dates : [fallback];
  nodes.dateFilter.textContent = "";
  for (const key of options) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = formatDate(`${key}T12:00:00`);
    nodes.dateFilter.appendChild(option);
  }
  if (state.filters.view === "daily" && (!state.filters.date || !options.includes(state.filters.date))) {
    state.filters.date = options[0];
  }
  if (state.filters.date && options.includes(state.filters.date)) nodes.dateFilter.value = state.filters.date;
}

function updateStats() {
  const papers = state.allPapers || [];
  const now = new Date();
  const recentWeekStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const weekPapers = papers.filter((paper) => inRange(paperDate(paper), recentWeekStart, new Date(now.getTime() + 24 * 60 * 60 * 1000)));
  const monthPapers = papers.filter((paper) => inRange(paperDate(paper), startOfMonth(now), endOfMonth(now)));
  const top = papers.reduce((max, paper) => Math.max(max, finalScoreOf(paper)), 0);
  nodes.paperCount.textContent = String(papers.length);
  nodes.weekCount.textContent = String(weekPapers.length);
  nodes.monthCount.textContent = String(monthPapers.length);
  nodes.topScore.textContent = top.toFixed(2);
}

function bindEvents() {
  for (const option of nodes.themeOptions) {
    option.addEventListener("click", () => {
      applyTheme(option.dataset.themeOption);
    });
  }
  nodes.searchInput.addEventListener("input", (event) => {
    state.filters.query = event.target.value.trim();
    render();
  });
  nodes.topicFilter.addEventListener("change", (event) => {
    state.filters.topic = event.target.value;
    render();
  });
  nodes.levelFilter.addEventListener("change", (event) => {
    state.filters.level = event.target.value;
    render();
  });
  for (const tab of nodes.collectionTabs) {
    tab.addEventListener("click", () => {
      state.filters.collection = tab.dataset.collection;
      state.filters.view = "all";
      state.filters.date = null;
      state.filters.topic = "all";
      for (const item of nodes.collectionTabs) item.classList.toggle("active", item === tab);
      for (const item of nodes.tabs) item.classList.toggle("active", item.dataset.view === state.filters.view);
      syncDateFilterState();
      hydrateTopicFilter();
      hydrateDateFilter();
      updateStats();
      updateUpdatedAt();
      render();
    });
  }
  nodes.dateFilter.addEventListener("change", (event) => {
    state.filters.date = event.target.value;
    state.filters.view = "daily";
    for (const item of nodes.tabs) item.classList.toggle("active", item.dataset.view === "daily");
    syncDateFilterState();
    updateStats();
    render();
  });
  for (const tab of nodes.tabs) {
    tab.addEventListener("click", () => {
      state.filters.view = tab.dataset.view;
      if (state.filters.view === "all") state.filters.date = null;
      if (state.filters.view === "daily" && !state.filters.date) state.filters.date = nodes.dateFilter.value || null;
      for (const item of nodes.tabs) item.classList.toggle("active", item === tab);
      syncDateFilterState();
      render();
    });
  }
}

async function loadData() {
  const response = await fetch("data/papers.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const normalized = normalizeData(data, "daily");
  const allPapers = Array.isArray(data) ? data : data.papers || [];
  console.log("Loaded papers raw data:", data);
  console.log("All papers count:", allPapers.length);
  console.log("First paper:", normalized.papers[0]);
  return normalized;
}

async function loadOptionalData(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) return emptyDataset();
  return normalizeData(await response.json(), "conference");
}

function updateUpdatedAt(message = "") {
  if (message) {
    nodes.updatedAt.textContent = message;
    return;
  }
  const data = activeData();
  const stats = data.stats || {};
  const mode = stats.collection_mode === "incremental" ? "增量" : "初始化";
  const kind = state.filters.collection === "conference" ? "顶会精品" : "全部论文";
  nodes.updatedAt.textContent = `${kind} · 更新于 ${formatDate(data.generated_at_iso)} · ${mode} · ${stats.llm_enabled ? "LLM" : "基础"}`;
}

async function main() {
  applyTheme(storedTheme());
  bindEvents();
  try {
    state.datasets.daily = await loadData();
    state.datasets.conference = await loadOptionalData("./data/conference_papers.json");
    state.allPapers = state.datasets.daily.papers || [];
    if (nodes.dataDebug) nodes.dataDebug.textContent = `Data loaded: ${state.allPapers.length} papers`;
  } catch (error) {
    state.datasets.daily = {
      generated_at_iso: new Date().toISOString(),
      topics: [],
      papers: [],
      stats: { llm_enabled: false },
    };
    state.datasets.conference = {
      generated_at_iso: new Date().toISOString(),
      topics: [],
      papers: [],
      stats: { llm_enabled: false },
    };
    state.allPapers = [];
    if (nodes.dataDebug) nodes.dataDebug.textContent = "Data loaded: 0 papers";
    updateUpdatedAt(`数据读取失败：${error.message}`);
  }

  updateUpdatedAt();
  hydrateTopicFilter();
  hydrateDateFilter();
  updateStats();
  render();
}

main();
