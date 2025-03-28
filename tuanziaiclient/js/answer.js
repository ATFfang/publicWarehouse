// 获取输入框和终端内容区域的引用
const terminalInput = document.getElementById('terminal-input');
const terminalContent = document.getElementById('terminal-content');

// 监听用户输入并在按下 Enter 键时作出响应
terminalInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        const userInput = terminalInput.value;  // 获取用户输入内容
        if (userInput.trim() !== '') {
            addMessage(`> ${userInput}`, 'output');  // 显示用户输入

            // 触发回复函数
            setTimeout(() => {
                addMessage('好的主人', 'output typewriter', true);  // 显示带打字效果和发光字的回复内容
            }, 500);  // 模拟回复延迟
        }

        terminalInput.value = '';  // 清空输入框
    }
});

// 将消息添加到终端内容中
function addMessage(message, className, isReply = false) {
    const messageElement = document.createElement('div');
    
    // 将多个类名分开传递
    const classes = className.split(' ');
    messageElement.classList.add(...classes);
    
    messageElement.textContent = message;

    // 如果是系统回复，将最后两个字加上发光效果
    if (isReply) {
        const glowingText = document.createElement('span');
        glowingText.classList.add('glowing');
        glowingText.textContent = message.slice(-2); // 最后两个字 "主人"
        messageElement.textContent = message.slice(0, -2); // 剩下的部分 "好的"
        messageElement.appendChild(glowingText); // 拼接发光效果的部分
    }

    terminalContent.appendChild(messageElement);
    terminalContent.scrollTop = terminalContent.scrollHeight;  // 滚动到最新消息
}



