import { Database } from "lucide-react";
import type { ChatMessage } from "../types/api";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">
          <Database size={26} aria-hidden="true" />
        </div>
        <h2>企业知识库智能问答</h2>
        <p>输入问题，CloudDesk 将结合检索结果生成回答。</p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </div>
  );
}
