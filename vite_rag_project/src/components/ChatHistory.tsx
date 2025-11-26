import React from 'react';
import ReactMarkdown from 'react-markdown';

// App.tsx와 동일한 Message 인터페이스를 사용
interface Message {
    id: number;
    sender: 'user' | 'llm';
    text: string;
}

interface ChatHistoryProps {
    messages: Message[];
    //App.tsx에서 전달받을 Ref 타입
    messagesEndRef: React.RefObject<HTMLDivElement|null>; 
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ messages, messagesEndRef }) => { 
    return (
        <div className="chat-history">
            {messages.map(message => (
                <div 
                    key={message.id} 
                    className={`message ${message.sender}`}
                >   {message.sender === 'llm' ? (
                        /* LLM 메시지: 마크다운 렌더링 */
                        <div className="markdown-content">
                            <ReactMarkdown>{message.text}</ReactMarkdown>
                        </div>
                    ) : (
                        /* 사용자 메시지: 줄바꿈만 인식 */
                        <div style={{ whiteSpace: 'pre-wrap' }}>
                            {message.text}
                        </div>
                    )}
                </div>
            ))}
            
            {/*스크롤 Ref*/}
            <div ref={messagesEndRef} />
        </div>
    );
};

export default ChatHistory;