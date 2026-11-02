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

function setBusy(value) {
  busy = value;
  input.disabled = value;
  sendButton.disabled = value || !input.value.trim();
  newChat.disabled = value;
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
  label.textContent = role === "user" ? "你" : "OrderAgent";
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
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "请求未完成，请稍后再试。");
  return data;
}

async function connect() {
  setBusy(true);
  showNotice("");
  try {
    const session = await request("/api/session");
    csrf = session.csrf_token;
    document.querySelector("#user-id").textContent = session.user_id;
    messages.replaceChildren();
    welcome.hidden = session.messages.length > 0;
    session.messages.forEach(message => addMessage(message.role, message.text));
    setBusy(false);
    input.focus();
  } catch (error) {
    showNotice(error.message || "连接失败，请确认本地服务仍在运行。", true);
  }
}

document.querySelector("#chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = input.value.trim();
  if (busy || !text || text.length > 4000 || !csrf) return;
  setBusy(true);
  showNotice("");
  addMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  const pending = addMessage("assistant", "", true);
  try {
    const data = await request("/api/chat", {message: text});
    pending.remove();
    addMessage("assistant", data.answer);
  } catch (error) {
    pending.remove();
    showNotice(`${error.message} 如连接中断，处理可能仍在继续；请先查询进度，不要重复提交确认。`, true);
  } finally {
    setBusy(false);
    input.focus();
  }
});

input.addEventListener("input", () => {
  sendButton.disabled = busy || !input.value.trim();
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
  if (busy || !csrf || !window.confirm("开始新对话？聊天记录和咨询次数将清空，退货记录仍会保留。")) return;
  setBusy(true);
  try {
    await request("/api/reset", {});
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
connect();
