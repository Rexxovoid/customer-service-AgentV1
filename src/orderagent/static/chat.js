"use strict";

const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newChat = document.querySelector("#new-chat");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const notice = document.querySelector("#notice");
const scrollArea = document.querySelector("#scroll-area");
let csrf = "";
let busy = true;
let authenticated = false;
let conversation = null;
let writable = false;
let historyOffset = 0;
let storedMessages = [];
const statuses = {agent_consulting: "Agent咨询", human_active: "模拟人工处理中", waiting_human: "等待人工", transfer_failed: "交接失败", closed: "已结束"};

function setBusy(value) {
  busy = value;
  input.disabled = value || !writable;
  sendButton.disabled = value || !writable || !input.value.trim();
  newChat.disabled = value;
  document.querySelector("#end-service").disabled = value;
  document.querySelectorAll(".suggestion").forEach(button => { button.disabled = value; });
  messages.setAttribute("aria-busy", String(value));
}

function showNotice(text, reconnect = false) {
  notice.textContent = text;
  notice.hidden = !text;
  if (reconnect) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "重新连接";
    button.addEventListener("click", connect);
    notice.append(button);
  }
}

function addMessage(role, text, loading = false) {
  welcome.hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "U" : "✳";
  avatar.setAttribute("aria-hidden", "true");
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = ({user: "你", simulated_agent: "模拟人工客服", system: "系统通知"})[role] || "OrderAgent";
  const content = document.createElement("div");
  content.className = "message-text";
  if (loading) {
    content.classList.add("loading-dots");
    content.setAttribute("aria-label", "正在查询，请稍候");
    for (let i = 0; i < 3; i++) content.append(document.createElement("span"));
  } else {
    // 对模型和用户正文一律使用textContent，禁止HTML注入。
    content.textContent = text;
  }
  body.append(label, content);
  article.append(avatar, body);
  messages.append(article);
  scrollArea.scrollTop = scrollArea.scrollHeight;
  return article;
}

async function request(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    credentials: "same-origin",
    headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  });
  const data = await response.json();
  if (!response.ok) {
    const id = data.request_id || response.headers.get("X-Request-ID");
    const message = typeof data.detail === "string" ? data.detail : "请求未完成，请稍后再试。";
    const error = new Error(`${message}${id ? ` 问题编号：${id}` : ""}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

async function connect() {
  setBusy(true);
  showNotice("");
  try {
    const session = await request("/api/session");
    csrf = session.csrf_token;
    authenticated = session.authenticated;
    document.querySelector("#login-panel").hidden = authenticated;
    document.querySelector("#logout").hidden = !authenticated;
    document.querySelector("#user-id").textContent = authenticated ? session.username : "未登录";
    messages.replaceChildren();
    if (!authenticated) {
      conversation = null;
      writable = false;
      storedMessages = [];
      document.querySelector("#history-list").replaceChildren();
      document.querySelector("#history-more").hidden = true;
      document.querySelector("#messages-more").hidden = true;
      document.querySelector("#end-service").hidden = true;
      welcome.hidden = false;
      document.querySelector("#username").focus();
      return;
    }
    renderConversation(session);
    await loadHistory(true);
    setBusy(false);
    input.focus();
  } catch (error) {
    showNotice(error.message || "连接失败，请确认本地服务仍在运行。", true);
  }
}

function renderConversation(snapshot) {
  conversation = snapshot.conversation;
  writable = snapshot.writable;
  storedMessages = snapshot.messages;
  renderMessages();
  document.querySelector("#conversation-label").textContent =
    `${statuses[conversation.status] || conversation.status}${writable ? "" : " · 历史记录，仅供查看"}`;
  document.querySelector("#end-service").hidden = conversation.channel !== "human" || conversation.status === "closed";
  showNotice(writable ? "" : "此会话仅供查看。需要继续咨询时，请新建对话。");
}

function renderMessages() {
  messages.replaceChildren();
  welcome.hidden = storedMessages.length > 0;
  storedMessages.forEach(message => {
    const suffix = message.status === "unknown" ? "\n【结果待核查，不自动重试】" :
      message.status === "processing" ? "\n【处理中】" : "";
    addMessage(message.role, message.text + suffix);
  });
  document.querySelector("#messages-more").hidden = storedMessages.length < 100;
}

async function loadHistory(reset = false) {
  if (reset) {
    historyOffset = 0;
    document.querySelector("#history-list").replaceChildren();
  }
  const data = await request(`/api/conversations?offset=${historyOffset}&limit=20`);
  data.items.forEach(item => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${item.title} · ${statuses[item.status] || item.status}`;
    button.addEventListener("click", async () => {
      if (busy) return;
      setBusy(true);
      try { renderConversation(await request(`/api/conversations/${item.id}/select`, {})); }
      catch (error) { showNotice(error.message); }
      finally { setBusy(false); }
    });
    document.querySelector("#history-list").append(button);
  });
  historyOffset += data.items.length;
  document.querySelector("#history-more").hidden = !data.has_more;
}

document.querySelector("#history-more").addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try { await loadHistory(); } catch (error) { showNotice(error.message); }
  finally { setBusy(false); }
});
document.querySelector("#messages-more").addEventListener("click", async () => {
  if (busy || !conversation || !storedMessages.length) return;
  setBusy(true);
  try {
    const data = await request(`/api/conversations/${conversation.id}?before=${storedMessages[0].seq}`);
    const ids = new Set(storedMessages.map(m => m.id));
    storedMessages = [...data.messages.filter(m => !ids.has(m.id)), ...storedMessages];
    renderMessages();
    document.querySelector("#messages-more").hidden = data.messages.length < 100;
  } catch (error) { showNotice(error.message); }
  finally { setBusy(false); }
});
document.querySelector("#end-service").addEventListener("click", async () => {
  if (busy || !conversation || !window.confirm("结束本次模拟人工服务？历史仍可查看，未完成退货不会被取消。")) return;
  setBusy(true);
  try {
    renderConversation(await request(`/api/conversations/${conversation.id}/close`, {}));
    await loadHistory(true);
  } catch (error) { showNotice(error.message); }
  finally { setBusy(false); }
});

document.querySelector("#chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = input.value.trim();
  if (busy || !writable || !text || text.length > 4000 || !csrf) return;
  setBusy(true);
  showNotice("");
  addMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  const pending = addMessage("assistant", "", true);
  try {
    await request("/api/chat", {message: text, message_id: crypto.randomUUID(), conversation_id: conversation.id});
    pending.remove();
    await connect();
  } catch (error) {
    if (error.status === 401) await connect();
    pending.remove();
    showNotice(`${error.message} 如连接中断，处理可能仍在继续；请先查询进度，不要重复提交确认。`, true);
  } finally {
    setBusy(!authenticated);
    if (authenticated) input.focus();
  }
});

input.addEventListener("input", () => {
  sendButton.disabled = busy || !writable || !input.value.trim();
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    document.querySelector("#chat-form").requestSubmit();
  }
});
document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
});
newChat.addEventListener("click", async () => {
  if (busy || !csrf || !window.confirm("开启新咨询？历史记录及未结束人工服务仍会保留。")) return;
  setBusy(true);
  try {
    await request("/api/reset", {});
    await connect();
    messages.replaceChildren();
    welcome.hidden = false;
    input.value = "";
    input.style.height = "auto";
    showNotice("");
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    setBusy(false);
    input.focus();
  }
});
document.querySelector("#login-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = document.querySelector("#login-button");
  button.disabled = true;
  document.querySelector("#login-error").textContent = "";
  try {
    await request("/api/login", {
      username: document.querySelector("#username").value,
      password: document.querySelector("#password").value
    });
    document.querySelector("#password").value = "";
    await connect();
  } catch (error) {
    document.querySelector("#login-error").textContent = error.message;
    if (error.status === 401 || error.status === 403) await connect();
  } finally { button.disabled = false; }
});
document.querySelector("#logout").addEventListener("click", async () => {
  if (busy) return;
  try {
    await request("/api/logout", {});
    authenticated = false;
    await connect();
  } catch (error) { showNotice(error.message); }
});
connect();
