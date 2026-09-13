import {
  LoaderCircle,
  MessageSquare,
  Plus,
  Wifi,
  WifiOff
} from "lucide-react";
import type { Conversation, ServiceHealth } from "../types/api";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  health: ServiceHealth;
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
}

const healthText: Record<ServiceHealth, string> = {
  checking: "检查服务中",
  online: "服务正常",
  offline: "服务异常"
};

export function Sidebar({
  conversations,
  activeId,
  health,
  onNewConversation,
  onSelectConversation
}: SidebarProps) {
  const HealthIcon =
    health === "online"
      ? Wifi
      : health === "offline"
        ? WifiOff
        : LoaderCircle;

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">CD</div>
        <div>
          <strong>CloudDesk</strong>
          <span>Enterprise AI Support</span>
        </div>
      </div>

      <button
        className="new-conversation-button"
        type="button"
        onClick={onNewConversation}
      >
        <Plus size={17} aria-hidden="true" />
        新建会话
      </button>

      <div className="conversation-heading">
        <span>会话</span>
        <span>{conversations.length}</span>
      </div>

      <nav className="conversation-list" aria-label="本地会话列表">
        {conversations.map((conversation) => (
          <button
            className={
              conversation.id === activeId
                ? "conversation-item active"
                : "conversation-item"
            }
            key={conversation.id}
            type="button"
            onClick={() => onSelectConversation(conversation.id)}
            aria-current={conversation.id === activeId ? "page" : undefined}
          >
            <MessageSquare size={16} aria-hidden="true" />
            <span>{conversation.title}</span>
          </button>
        ))}
      </nav>

      <div className={`service-status ${health}`} title={healthText[health]}>
        <HealthIcon
          className={health === "checking" ? "status-spin" : ""}
          size={16}
          aria-hidden="true"
        />
        <span>{healthText[health]}</span>
      </div>
    </aside>
  );
}
