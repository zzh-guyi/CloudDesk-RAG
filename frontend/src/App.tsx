import { Composer } from "./components/Composer";
import { MessageList } from "./components/MessageList";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const {
    conversations,
    activeConversation,
    isStreaming,
    health,
    createConversation,
    selectConversation,
    sendMessage,
    stopGeneration
  } = useChat();

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeConversation?.id ?? null}
        health={health}
        onNewConversation={createConversation}
        onSelectConversation={selectConversation}
      />

      <main className="chat-workspace">
        <header className="chat-header">
          <div>
            <span className="header-kicker">CloudDesk AI</span>
            <h1>{activeConversation?.title ?? "智能客服"}</h1>
          </div>
          <span className={isStreaming ? "generation-state active" : "generation-state"}>
            {isStreaming ? "正在生成" : "在线咨询"}
          </span>
        </header>

        <section className="chat-body" aria-live="polite">
          <MessageList messages={activeConversation?.messages ?? []} />
        </section>

        <Composer
          isStreaming={isStreaming}
          onSend={(query) => void sendMessage(query)}
          onStop={stopGeneration}
        />
      </main>
    </div>
  );
}
